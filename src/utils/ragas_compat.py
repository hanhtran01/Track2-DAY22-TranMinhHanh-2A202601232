"""
Vá tương thích cho RAGAS 0.4.x trên langchain-community 0.4.x.

Vấn đề: ragas/llms/base.py vẫn chạy
    from langchain_community.chat_models.vertexai import ChatVertexAI
nhưng module đó đã bị gỡ khỏi langchain-community 0.4.0 trở lên → ImportError
ngay khi `import ragas`.

Cách vá: đăng ký một module giả vào sys.modules để câu import trên đi qua được.
Lab này không dùng Vertex AI, nên lớp giả bên dưới không bao giờ được khởi tạo.
Các phép `isinstance(llm, ChatVertexAI)` bên trong ragas sẽ trả về False — đúng
với thực tế là ta đang dùng OpenAI/OpenRouter.

CÁCH DÙNG: import module này TRƯỚC dòng `from ragas import ...`
    from utils import ragas_compat  # noqa: F401
    from ragas import evaluate
"""
import sys
import types

_MODULE_NAME = "langchain_community.chat_models.vertexai"


def _install_shim() -> bool:
    """Trả về True nếu đã phải cài module giả, False nếu bản gốc còn dùng được."""
    try:
        __import__(_MODULE_NAME)
        return False
    except ImportError:
        pass

    class ChatVertexAI:
        """Lớp giữ chỗ — lab không dùng Vertex AI."""

        def __init__(self, *args, **kwargs):
            """Luôn raise — lớp này chỉ để giữ chỗ cho câu import của ragas."""
            raise NotImplementedError(
                "Vertex AI không được hỗ trợ trong lab này. "
                "Đổi PROVIDER trong .env sang openai/gemini/anthropic/ollama/openrouter."
            )

    shim = types.ModuleType(_MODULE_NAME)
    shim.ChatVertexAI = ChatVertexAI
    sys.modules[_MODULE_NAME] = shim
    return True


SHIM_INSTALLED = _install_shim()
