"""
苹安保险智能客服 API 服务

FastAPI 入口 + 接口实现：
    - POST /chat/stream     流式问答（SSE），调用 Agent 的异步流式方法
    - DELETE /chat/{id}     清空会话短期记忆
    - GET /health           健康检查
    - GET /suggestions      首页推荐问题
"""
import asyncio
import contextlib
import json
import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

import qa_config
from insurance_qa_agent_stream import InsuranceQAAgentStream, get_agent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("server")

# 首页推荐问题
SUGGESTIONS = [
    "终身寿险和定期寿险有什么区别？",
    "重疾险一般包含哪些疾病？",
    "如何选择适合自己的医疗保险？",
    "保险理赔流程是怎样的？",
    "什么是免赔额？",
    "意外险有必要买吗？",
]

# SSE 结束标记（与前端约定）
SSE_DONE = "[DONE]"


# ==================== 数据模型 ====================

class ChatRequest(BaseModel):
    """聊天请求模型"""

    message: str = Field(
        ...,
        description="用户问题",
        min_length=1,
        max_length=qa_config.MAX_QUESTION_LENGTH,
        examples=["重疾险一般包含哪些疾病？"],
    )
    thread_id: str = Field(
        default="default",
        description="会话 ID，用于隔离多轮对话上下文",
        pattern=r"^[A-Za-z0-9_-]{1,64}$",
        examples=["web-1a2b3c-xyz"],
    )


class ClearSessionResponse(BaseModel):
    """清空会话记忆的响应模型"""

    status: str = "ok"
    thread_id: str


class HealthResponse(BaseModel):
    """健康检查响应模型"""

    status: str = "ok"
    message: str = "服务运行正常"
    model: str
    vectorstore: str


class SuggestionsResponse(BaseModel):
    """建议问题响应模型"""

    suggestions: list[str]


# ==================== 应用初始化 ====================

@asynccontextmanager
async def lifespan(_: FastAPI):
    """启动时预热 Agent（连接向量库、创建模型），失败则终止启动并给出明确提示。"""
    await asyncio.to_thread(get_agent)
    logger.info("Agent 初始化完成：model=%s", qa_config.MODEL_NAME)
    yield
    get_agent.cache_clear()
    logger.info("服务已关闭")


app = FastAPI(
    title="苹安保险智能客服 API",
    description="基于 LangChain + FastAPI 的保险智能问答系统",
    version="1.1.0",
    lifespan=lifespan,
)

# 配置跨域访问（CORS）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应改为具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== 内部工具函数 ====================

def _sse_event(payload: dict | str) -> str:
    """把数据序列化为一个标准 SSE 事件块（以空行结尾）。"""
    if isinstance(payload, str):
        data = payload
    else:
        # ensure_ascii=False 保证中文正常显示；JSON 内的换行会被转义，不会破坏 SSE 分帧
        data = json.dumps(payload, ensure_ascii=False)
    return f"data: {data}\n\n"


def _friendly_error(exc: Exception) -> str:
    """把底层异常转换成可读的提示信息，避免把堆栈暴露给前端。"""
    name = type(exc).__name__
    text = str(exc).strip()

    if "Authentication" in name or "401" in text or "invalid api key" in text.lower():
        return "模型鉴权失败，请检查 DEEPSEEK_API_KEY 是否配置正确。"
    if "RateLimit" in name or "429" in text:
        return "请求过于频繁，请稍后再试。"
    if "Timeout" in name or "Connect" in name:
        return "模型服务连接异常，请稍后重试。"
    return text[:200] if text else "服务内部异常，请稍后重试。"


async def _heartbeat(queue: "asyncio.Queue[tuple[str, str]]") -> None:
    """定时向队列投放心跳事件，避免长时间无输出导致连接被代理断开。"""
    while True:
        await asyncio.sleep(qa_config.STREAM_HEARTBEAT_SECONDS)
        queue.put_nowait(("ping", ""))


async def _stream_events(
    agent: InsuranceQAAgentStream, question: str, thread_id: str
) -> AsyncIterator[str]:
    """
    把 Agent 的异步 token 流转换成 SSE 事件流。

    事件格式（与前端 static/苹安保险智能客服.html 约定一致）：
        data: {"content": "增量文本"}
        data: {"error": "错误信息"}
        data: [DONE]
    """
    queue: "asyncio.Queue[tuple[str, str]]" = asyncio.Queue()
    heartbeat_task = asyncio.create_task(_heartbeat(queue))

    async def produce() -> None:
        """后台任务：消费 Agent 流，把结果投递到队列，保证与心跳互不阻塞。"""
        try:
            async for text in agent.chat_stream(question, thread_id):
                queue.put_nowait(("token", text))
        except asyncio.CancelledError:
            # 客户端中断（例如前端点击“停止”），直接向上抛出让任务退出
            raise
        except Exception as exc:  # noqa: BLE001 - 统一转为 SSE 错误事件
            logger.exception("流式回答生成失败 thread_id=%s", thread_id)
            queue.put_nowait(("error", _friendly_error(exc)))
        finally:
            queue.put_nowait(("end", ""))

    producer = asyncio.create_task(produce())
    finished = False
    failed = False
    try:
        while not finished:
            kind, payload = await queue.get()

            if kind == "end":
                finished = True
            elif kind == "error":
                yield _sse_event({"error": payload})
                failed = True
                finished = True
            elif kind == "token":
                yield _sse_event({"content": payload})
            elif kind == "ping":  # 心跳：SSE 注释行，前端会忽略
                yield ": keep-alive\n\n"

        # 只有正常结束才发送结束标记；出错时前端已收到 error 事件
        if finished and not failed:
            yield _sse_event(SSE_DONE)
    finally:
        # 无论正常结束还是客户端断开，都要回收后台任务，避免线程/连接泄漏
        producer.cancel()
        heartbeat_task.cancel()
        for task in (producer, heartbeat_task):
            with contextlib.suppress(asyncio.CancelledError):
                await task


# ==================== 接口定义 ====================

@app.get("/", tags=["元信息"], summary="API 概览")
async def root():
    """根路径 - 返回 API 基本信息"""
    return {
        "service": "苹安保险智能客服",
        "version": "1.1.0",
        "docs": "/docs",
        "endpoints": {
            "健康检查": "GET /health",
            "建议问题": "GET /suggestions",
            "流式问答": "POST /chat/stream",
            "清空会话": "DELETE /chat/{thread_id}",
        },
    }


@app.get("/health", response_model=HealthResponse, summary="健康检查")
async def health_check() -> HealthResponse:
    """验证服务是否正常运行（不依赖模型与向量库，便于探活）。"""
    return HealthResponse(
        model=qa_config.MODEL_NAME,
        vectorstore=qa_config.VECTORSTORE_PATH,
    )


@app.get("/suggestions", response_model=SuggestionsResponse, summary="获取建议问题")
async def get_suggestions() -> SuggestionsResponse:
    """返回首页推荐问题列表。"""
    return SuggestionsResponse(suggestions=SUGGESTIONS)


@app.post(
    "/chat/stream",
    summary="流式问答（SSE）",
    description=(
        "以 SSE 方式逐字返回回答。每个事件为 `data: {...}`，"
        "文本增量在 content 字段中，结束时返回 `data: [DONE]`。"
    ),
    response_class=StreamingResponse,
    responses={
        200: {
            "content": {"text/event-stream": {}},
            "description": "SSE 事件流",
        },
        400: {"description": "问题内容为空"},
        422: {"description": "请求参数校验失败"},
    },
)
async def chat_stream(
    payload: ChatRequest,
    agent: InsuranceQAAgentStream = Depends(get_agent),
) -> StreamingResponse:
    question = payload.message.strip()
    if not question:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="问题内容不能为空"
        )

    logger.info("开始流式问答 thread_id=%s", payload.thread_id)

    return StreamingResponse(
        _stream_events(agent, question, payload.thread_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            # 关闭 Nginx 等反向代理的缓冲，保证逐字输出
            "X-Accel-Buffering": "no",
        },
    )


@app.delete(
    "/chat/{thread_id}",
    response_model=ClearSessionResponse,
    summary="清空会话记忆",
    description="删除指定 thread_id 的多轮对话上下文。",
)
async def clear_session(
    thread_id: str,
    agent: InsuranceQAAgentStream = Depends(get_agent),
) -> ClearSessionResponse:
    await asyncio.to_thread(agent.clear_session, thread_id)
    return ClearSessionResponse(thread_id=thread_id)


# ==================== 启动服务 ====================
if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    reload_enabled = os.getenv("RELOAD", "1") == "1"

    print("\n" + "=" * 60)
    print("🚀 苹安保险智能客服 API")
    print("=" * 60)
    print(f"📍 前端页面: 打开 static/苹安保险智能客服.html（默认请求 http://localhost:{port}）")
    print(f"📍 接口文档: http://localhost:{port}/docs")
    print(f"📍 健康检查: http://localhost:{port}/health")
    print("=" * 60 + "\n")

    # 启动 Uvicorn 服务器
    uvicorn.run(
        "server:app",
        host=host,
        port=port,
        reload=reload_enabled,  # 开发模式：代码修改自动重载
        log_level="info",
    )
