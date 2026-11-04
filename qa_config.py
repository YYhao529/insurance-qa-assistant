# 项目配置文件
import os

# 路径配置
DATA_PATH = "./data"
VECTORSTORE_PATH = "./chroma_vsdb"
# 支持的文件类型
SUPPORTED_EXTENSIONS = ['.txt', '.pdf', '.csv']

# 分割参数
CHUNK_SIZE = 150
CHUNK_OVERLAP = 20

# 检索参数
TOP_K = 3

# ==================== LLM 模型配置 ====================
# 优先读取 DEEPSEEK_API_KEY，其次兼容 OPENAI_API_KEY
OPENAI_API_KEY = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
# 模型服务地址（可被环境变量覆盖，便于切换到代理 / 本地兼容服务）
OPENAI_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
MODEL_NAME = os.getenv("MODEL_NAME", "deepseek-flash")
MODEL_PROVIDER = "openai"
EMBED_MODEL = os.getenv("EMBED_MODEL", "bge-m3")

# ==================== 会话（短期记忆）配置 ====================
# 会话空闲多久后被自动清理（秒），默认 30 分钟
SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", "1800"))
# 最多保留的会话数量，超出后淘汰最久未活跃的会话
MAX_SESSIONS = int(os.getenv("MAX_SESSIONS", "500"))

# ==================== 流式输出配置 ====================
# SSE 心跳间隔（秒）：长时间无输出时发送注释行，避免连接被代理断开
STREAM_HEARTBEAT_SECONDS = int(os.getenv("STREAM_HEARTBEAT_SECONDS", "15"))
# 单次提问的最大长度，防止超长请求打爆模型上下文
MAX_QUESTION_LENGTH = int(os.getenv("MAX_QUESTION_LENGTH", "2000"))
