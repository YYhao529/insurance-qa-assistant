"""
智能保险问答 Agent（RAG 检索 + 工具调用 + 多轮记忆）。

对外能力：
    - chat_stream()     流式问答（异步生成器，逐段产出文本），供 server.py 调用
    - clear_session()   清空指定会话的短期记忆
    - 会话自动回收：超过 TTL 或数量上限的会话会被清理，避免记忆无限增长
"""
import logging
import threading
import time
from datetime import datetime
from functools import lru_cache
from typing import AsyncIterator

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_chroma import Chroma
from langchain_core.messages import AIMessageChunk, HumanMessage
from langchain_core.tools import tool
from langchain_ollama import OllamaEmbeddings
from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import InMemorySaver

import qa_config

logger = logging.getLogger(__name__)

# 系统提示词：定义角色、工具调用策略与回答底线
SYSTEM_PROMPT = """
你是「苹安保险智能客服」——一名专业、务实的保险顾问，面向中国普通家庭回答保险相关问题。

## 回答风格
- 全程使用简体中文；先给结论，再给理由和要点，避免大段铺垫和套话。
- 默认简洁：先给 2~4 条要点；复杂问题（如产品对比、理赔争议）再分点展开。
- 适当使用 Markdown（小标题、有序/无序列表、**加粗**关键结论）；前端不渲染表格，不要输出表格或代码块。
- 问题含糊时，先给出最可能的答案，再用一句话追问关键信息（如年龄、预算、已有保障），不要连续反问。

## 工具使用（按优先级）
1. 保险专业问题——保障责任、条款解读、产品对比、健康告知与核保、理赔流程、免赔额、等待期、续保等——必须先调用 rag_search 检索知识库；把口语化问题改写成检索关键词（例如“重疾险 28 种重大疾病 赔付”），不要把整段话直接丢进去。
2. 涉及“今天 / 现在 / 当前 / 今年 / 还有多少天”等时效信息或日期计算时，先调用 get_current_time 获取当前日期，严禁自行猜测或编造日期。
3. 知识库之外、且有时效性的信息（最新监管政策、行业新闻、产品停售或调整、公司最新动态、监管处罚等）使用 web_search；检索词加上“保险”“金融监管总局”等限定词，优先采用官方或权威媒体来源，并在回答中注明来源与时间。
4. 一次回答中同一工具原则上只调用 1 次；信息足够就停止调用、直接作答，不要复述工具返回的原文。
5. 工具没有结果或调用失败时，如实说明“知识库 / 联网搜索暂未覆盖”，给出通用建议和核实途径（保险条款、官网、官方客服），绝不编造数据、条款、公司或政策。

## 内容底线
- 保额、免赔额、等待期、赔付比例、费率等数字必须来自检索资料；资料中没有就写“具体以保险条款为准”。
- 不承诺收益、不承诺“一定能赔”，不做具体公司或产品的优劣排名。
- 涉及健康告知、既往症、理赔争议等个性化判断时，说明最终以保险公司的核保结论和官方答复为准。
- 不提供投资、避税、法律等超出保险顾问范围的确定性建议。

## 上下文
- 结合对话历史理解“这个 / 它 / 那款产品”等指代；用户已提供的信息不要重复询问。
- 用户纠正或补充信息后，以最新信息为准。
"""


class InsuranceQAAgentStream:
    """
    智能保险问答助手
        - RAG 检索
        - 工具调用
        - 内存记忆（按 thread_id 隔离）
    """

    # 构造方法
    def __init__(self):
        if not qa_config.OPENAI_API_KEY:
            raise RuntimeError(
                "未检测到模型 API Key，请先设置环境变量 DEEPSEEK_API_KEY 后再启动服务。"
            )

        # 初始化联网搜索工具（工具名与描述用中文，便于模型判断何时该联网）
        try:
            tavily = TavilySearch(
                name="web_search",
                description=(
                    "联网搜索最新、权威的公开信息，返回网页标题、链接和摘要。"
                    "适用于知识库以外的时效性内容：最新监管政策、行业新闻、"
                    "产品停售或调整、保险公司动态等。参数 query 为中文检索词，"
                    "建议带上“保险”“金融监管总局”等限定词。"
                ),
                max_results=5,  # 返回搜索结果条数
                topic="general",  # 通用全网搜索
                search_depth="basic",  # 搜索深度：basic基础 / advanced深度
            )
        except Exception as exc:  # noqa: BLE001 - 缺少 Key 时给出明确提示
            raise RuntimeError(
                "联网搜索工具初始化失败：请先设置环境变量 TAVILY_API_KEY。"
            ) from exc

        # 1.初始化向量模型
        self.embeddings_model = OllamaEmbeddings(model=qa_config.EMBED_MODEL)
        # 2.连接向量数据库
        self.db = Chroma(
            persist_directory=qa_config.VECTORSTORE_PATH,
            embedding_function=self.embeddings_model,
        )
        # 3.创建对话模型
        self.model = init_chat_model(
            model=qa_config.MODEL_NAME,
            model_provider=qa_config.MODEL_PROVIDER,
            base_url=qa_config.OPENAI_BASE_URL,
            api_key=qa_config.OPENAI_API_KEY,
        )
        # 4.短期记忆存储器
        self.checkpointer = InMemorySaver()
        # 4-1 会话活跃时间表（用于回收过期会话，避免记忆无限增长）
        self._active_at: dict[str, float] = {}
        self._sessions_lock = threading.Lock()

        # 5.定义工具
        @tool
        def rag_search(query: str) -> str:
            """在本地保险知识库中检索专业知识。

            覆盖：保险条款、保障责任、产品对比、健康告知与核保、理赔流程、
            免赔额、等待期、续保等。回答保险专业问题时应优先使用本工具，
            query 请写成简洁的中文检索关键词，而不是整句口语。
            """
            docs_with_scores = self.db.similarity_search_with_relevance_scores(
                query=query, k=qa_config.TOP_K
            )
            if not docs_with_scores:
                return "未检索到相关知识"
            return "\n".join(doc.page_content for doc, _score in docs_with_scores)

        @tool
        def get_current_time() -> str:
            """获取当前的系统日期和时间，格式为 YYYY-MM-DD HH:MM:SS（服务器本地时区）。

            当用户询问今天几号、当前时间，或需要进行日期计算
            （如距离某个日期还有多少天、某年龄对应的年份）时调用。
            """
            return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 6.创建智能体
        self.agent = create_agent(
            model=self.model,
            tools=[rag_search, get_current_time, tavily],
            checkpointer=self.checkpointer,
            system_prompt=SYSTEM_PROMPT,
        )

    # ==================== 内部工具方法 ====================

    @staticmethod
    def _session_config(thread_id: str) -> dict:
        """按会话 ID 生成 LangGraph 运行配置（决定记忆归属）。"""
        return {"configurable": {"thread_id": thread_id or "user_default"}}

    @classmethod
    def _extract_text(cls, message) -> str:
        """从消息对象中提取纯文本，兼容字符串与多模态内容块两种格式。"""
        content = getattr(message, "content", message)
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text") or "")
            return "".join(parts)
        return ""

    @classmethod
    def _chunk_text(cls, chunk) -> str:
        """
        判断一个流式消息块是否属于“要展示给用户的回答”，并返回文本。

        工具调用参数块（tool_call_chunks）和工具返回结果不展示，
        只保留模型产出的正文内容，避免前端出现多余字符。
        """
        if not isinstance(chunk, AIMessageChunk):
            return ""
        if getattr(chunk, "tool_call_chunks", None):
            return ""
        return cls._extract_text(chunk)

    # ==================== 会话（短期记忆）管理 ====================

    def _touch_session(self, thread_id: str) -> None:
        """记录会话最近活跃时间。"""
        with self._sessions_lock:
            self._active_at[thread_id] = time.monotonic()

    def _cleanup_sessions(self) -> int:
        """
        回收长期不活跃 / 超出数量上限的会话，避免记忆无限增长。

        返回本次回收的会话数量。
        """
        now = time.monotonic()
        with self._sessions_lock:
            expired = {
                tid
                for tid, ts in self._active_at.items()
                if now - ts > qa_config.SESSION_TTL_SECONDS
            }
            # 超出上限时，优先淘汰最久未活跃的会话
            alive = sorted(
                (ts, tid) for tid, ts in self._active_at.items() if tid not in expired
            )
            overflow = len(alive) - qa_config.MAX_SESSIONS
            if overflow > 0:
                expired.update(tid for _ts, tid in alive[:overflow])
            for tid in expired:
                self._active_at.pop(tid, None)

        # 删除记忆放到锁外执行，避免持锁做 I/O
        for tid in expired:
            try:
                self.checkpointer.delete_thread(tid)
            except Exception:  # noqa: BLE001 - 清理失败不应影响正常问答
                logger.warning("回收会话记忆失败 thread_id=%s", tid, exc_info=True)

        if expired:
            logger.info("已回收 %d 个过期会话", len(expired))
        return len(expired)

    @property
    def session_count(self) -> int:
        """当前保存了记忆的会话数量。"""
        with self._sessions_lock:
            return len(self._active_at)

    # ==================== 对外接口 ====================

    # 流式问答（异步生成器，逐段产出文本）
    async def chat_stream(
            self, question: str, thread_id: str = "user_default"
    ) -> AsyncIterator[str]:
        session_id = thread_id or "user_default"
        self._cleanup_sessions()
        self._touch_session(session_id)
        try:
            stream = self.agent.astream(
                {"messages": [HumanMessage(content=question)]},
                stream_mode="messages",
                config=self._session_config(session_id),
            )
            async for chunk, _metadata in stream:
                text = self._chunk_text(chunk)
                if text:
                    yield text
        finally:
            # 生成结束（含客户端中断）后刷新活跃时间，便于后续按 TTL 回收
            self._touch_session(session_id)

    # 清空指定会话的短期记忆
    def clear_session(self, thread_id: str) -> None:
        tid = thread_id or "user_default"
        with self._sessions_lock:
            self._active_at.pop(tid, None)
        self.checkpointer.delete_thread(tid)


@lru_cache(maxsize=1)
def get_agent() -> InsuranceQAAgentStream:
    """获取全局唯一的 Agent 实例（懒加载，供接口函数依赖注入）。"""
    return InsuranceQAAgentStream()


if __name__ == "__main__":
    # 命令行流式体验：python insurance_qa_agent_stream.py
    import asyncio


    async def _cli() -> None:
        qa_agent = get_agent()
        print("苹安保险智能客服已就绪，输入问题开始对话（输入 exit 退出）\n")
        while True:
            try:
                question = input("你：").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not question or question.lower() in {"exit", "quit"}:
                break
            print("苹安保险智能客服：", end="", flush=True)
            async for piece in qa_agent.chat_stream(question):
                print(piece, end="", flush=True)
            print("\n")


    asyncio.run(_cli())
