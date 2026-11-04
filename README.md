# 苹安保险智能客服 · 保险问答助手

基于 **FastAPI + LangChain（LangGraph）** 的保险问答服务：RAG 知识检索 + 工具调用 + 多轮记忆，
并支持 **SSE 流式输出**，可直接对接前端页面 `static/苹安保险智能客服.html`。

## 项目结构

```
insurance-qa-assistant/
├── server.py                    # FastAPI 入口 + 接口实现（/chat/stream、/health、/suggestions、清空会话）
├── insurance_qa_agent_stream.py # Agent 层：系统提示词、三个工具、模型、短期记忆、流式问答
├── build_vectorstore.py         # 构建向量库（data/ -> chroma_vsdb/）
├── qa_config.py                 # 配置（模型、检索参数、流式参数，均支持环境变量覆盖）
├── data/                        # 知识库原始文件（txt / pdf / csv）
├── chroma_vsdb/                 # 向量库持久化目录
├── static/                      # 前端资源
│   ├── 苹安保险智能客服.html      # 前端页面
│   └── img/                     # 图片资源（客服头像等）
└── requirements.txt
```

调用关系：`HTTP 请求 → server.py（参数校验、组织 SSE 响应）→ Agent（对话、检索、记忆）`。

## 内置工具

Agent 由系统提示词（`SYSTEM_PROMPT`）约束工具的使用时机：

| 工具 | 作用 | 典型场景 |
| --- | --- | --- |
| `rag_search` | 检索本地保险知识库（data/ 构建的向量库） | 条款、保障责任、产品对比、核保、理赔流程、免赔额、等待期 |
| `get_current_time` | 获取服务器当前日期时间 | “今天几号”“还有多少天”“今年多少岁”等日期计算 |
| `web_search` | Tavily 联网搜索（5 条结果） | 知识库以外的时效信息：最新监管政策、行业新闻、产品停售或调整等 |

工具选择、回答风格、数字与合规底线都写在 `SYSTEM_PROMPT` 里，改提示词只需改这一处常量。

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置 Key（必填）
#    Windows PowerShell:
#      $env:DEEPSEEK_API_KEY="sk-xxx"      # 对话模型
#      $env:TAVILY_API_KEY="tvly-xxx"      # 联网搜索
#    macOS / Linux:
#      export DEEPSEEK_API_KEY="sk-xxx"
#      export TAVILY_API_KEY="tvly-xxx"

# 3. 本地需要 Ollama，并已拉取向量模型
ollama pull bge-m3

# 4. 首次使用先构建向量库
python build_vectorstore.py

# 5. 启动服务
python server.py            # 默认 http://localhost:8000
```

启动后可访问：

- 接口文档：<http://localhost:8000/docs>
- 前端页面：直接用浏览器打开 `static/苹安保险智能客服.html`（默认请求 `http://localhost:8000`）
- 命令行体验：`python insurance_qa_agent_stream.py`

## 接口一览

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/health` | 健康检查（返回模型名、向量库路径） |
| GET | `/suggestions` | 首页推荐问题 |
| POST | `/chat/stream` | **流式问答（SSE）** |
| DELETE | `/chat/{thread_id}` | 清空指定会话的记忆 |

### 请求体

```json
{
  "message": "重疾险一般包含哪些疾病？",
  "thread_id": "web-abc-123"
}
```

`thread_id` 用于隔离多轮对话上下文（仅允许字母、数字、下划线和短横线，最长 64 位）。
前端每次打开 / 刷新页面都会随机生成一个新的 `thread_id`（`web-<uuid>`），
因此**刷新页面即等于开启一个新会话，短期记忆自然清空**；同一页面内的多轮提问共用同一个 ID，上下文连续。

### 流式接口 `POST /chat/stream`

响应头为 `Content-Type: text/event-stream`，每个事件以空行分隔：

```
data: {"content": "重疾险"}

data: {"content": "通常包含"}

data: [DONE]
```

- 文本增量在 `content` 字段，前端累加即可逐字渲染；
- 生成结束发送 `data: [DONE]`；
- 生成中途出错发送 `data: {"error": "错误信息"}`，随后连接关闭；
- 长时间无输出时会发送 `: keep-alive` 注释行（默认 15 秒），防止代理断开连接，前端忽略即可。

### 调用示例

```bash
curl -N -X POST http://localhost:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"message":"什么是免赔额？","thread_id":"web-1"}'
```

## 关键设计

- **Agent 单例**：`get_agent()` 使用 `lru_cache` 懒加载，通过 FastAPI `Depends` 注入接口函数，
  整个进程只创建一个 Agent（含向量库连接与记忆存储器）。
- **异步流式**：`server.py` 调用 `agent.chat_stream()`（异步生成器），用后台任务把 token 写入队列，
  由 SSE 生成器逐条推送，心跳与断连清理都在后台任务中完成，不阻塞事件循环。
- **只透出正文**：工具调用参数块、工具返回结果、推理内容都会被过滤，前端不会出现多余字符。
- **客户端断开即停止生成**：前端点击“停止”后，SSE 连接关闭，后台生成任务随之取消，避免继续消耗模型额度。
- **会话按需回收**：每来一次问答就顺带清理一次记忆——超过 `SESSION_TTL_SECONDS` 未活跃的会话会被删除，
  会话数超过 `MAX_SESSIONS` 时优先淘汰最久未活跃的，避免“每次刷新一个新会话”导致内存无限增长。
- **刷新即释放**：前端在 `pagehide` 时用 `fetch(..., { keepalive: true })` 调用
  `DELETE /chat/{thread_id}`，页面刷新 / 关闭后服务端记忆立即释放（走 bfcache 返回时 `persisted === true`，不清理）。

## 环境变量

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `DEEPSEEK_API_KEY` | 无（必填） | 模型 API Key，也兼容 `OPENAI_API_KEY` |
| `TAVILY_API_KEY` | 无（必填） | 联网搜索工具 Tavily 的 API Key |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com` | 模型服务地址 |
| `MODEL_NAME` | `deepseek-flash` | 对话模型名称 |
| `EMBED_MODEL` | `bge-m3` | 向量模型名称 |
| `STREAM_HEARTBEAT_SECONDS` | `15` | SSE 心跳间隔（秒） |
| `MAX_QUESTION_LENGTH` | `2000` | 单次提问最大长度 |
| `SESSION_TTL_SECONDS` | `1800` | 会话空闲多久后自动清理记忆（秒） |
| `MAX_SESSIONS` | `500` | 最多保留的会话数，超出淘汰最久未活跃的 |
| `HOST` / `PORT` / `RELOAD` | `0.0.0.0` / `8000` / `1` | 服务监听地址、端口、热重载开关 |
