"""
使用本地 Embedding 模型的工具

这个文件提供了使用本地开源 embedding 模型的替代方案，
可以替代 tools.py 中的 OpenAIEmbeddings。
"""
import os
from typing import Any

from langchain_chroma import Chroma
from langchain_core.tools import BaseTool, tool


def get_local_embeddings(
    model_type: str = "huggingface",
    model_name: str = "BAAI/bge-small-en-v1.5",
) -> Any:
    """
    获取本地 embedding 模型

    Args:
        model_type: 模型类型 ('huggingface', 'ollama', 'sentence-transformers')
        model_name: 模型名称

    Returns:
        Embeddings 实例
    """
    cache_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "embedding.model")
    
    if model_type == "huggingface":
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings

            return HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={"device": "cpu"},  # 使用 CPU，如果有 GPU 可以改为 "cuda"
                cache_folder=cache_folder, 
                encode_kwargs={"normalize_embeddings": True},
            )
        except ImportError:
            raise ImportError(
                "需要安装 langchain-community: pip install langchain-community"
            )

    elif model_type == "ollama":
        try:
            from langchain_community.embeddings import OllamaEmbeddings

            ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            return OllamaEmbeddings(
                model=model_name,
                base_url=ollama_base_url,
            )
        except ImportError:
            raise ImportError(
                "需要安装 langchain-community: pip install langchain-community"
            )

    elif model_type == "sentence-transformers":
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings

            return HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
        except ImportError:
            raise ImportError(
                "需要安装 langchain-community 和 sentence-transformers: "
                "pip install langchain-community sentence-transformers"
            )

    else:
        raise ValueError(f"不支持的模型类型: {model_type}")


def load_chroma_db_local(
    model_type: str = "huggingface",
    model_name: str = "BAAI/bge-small-en-v1.5",
):
    """
    使用本地 embedding 模型加载 ChromaDB

    Args:
        model_type: 模型类型 ('huggingface', 'ollama', 'sentence-transformers')
        model_name: 模型名称

    Returns:
        Retriever 实例
    """
    try:
        # 获取本地 embedding 模型
        embeddings = get_local_embeddings(model_type, model_name)

        # Load the stored vector database
        chroma_db = Chroma(
            persist_directory="./chroma_db", embedding_function=embeddings
        )
        retriever = chroma_db.as_retriever(search_kwargs={"k": 5})
        return retriever
    except Exception as e:
        raise RuntimeError(
            f"Failed to initialize local embeddings ({model_type}): {e}. "
            f"确保已安装必要的依赖: pip install langchain-community sentence-transformers"
        ) from e


def database_search_local_func(
    query: str,
    model_type: str = "huggingface",
    model_name: str = "BAAI/bge-small-en-v1.5",
) -> str:
    """
    使用本地 embedding 模型搜索数据库

    Args:
        query: 查询字符串
        model_type: 模型类型
        model_name: 模型名称

    Returns:
        相关文档内容
    """
    # Get the chroma retriever
    retriever = load_chroma_db_local(model_type, model_name)

    # Search the database for relevant documents
    documents = retriever.invoke(query)

    # Format the documents into a string
    from agents.tools import format_contexts

    context_str = format_contexts(documents)

    return context_str


# 创建工具（使用环境变量配置模型）
def create_database_search_tool():
    """创建数据库搜索工具，使用环境变量配置的本地模型"""
    model_type = os.getenv("LOCAL_EMBEDDING_TYPE", "huggingface")
    model_name = os.getenv(
        "LOCAL_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"
    )

    def search_func(query: str) -> str:
        return database_search_local_func(query, model_type, model_name)

    database_search: BaseTool = tool(search_func)
    database_search.name = "Database_Search_Local"
    return database_search


# 如果设置了使用本地模型，则创建工具
if os.getenv("USE_LOCAL_EMBEDDINGS", "false").lower() == "true":
    database_search_local = create_database_search_tool()
else:
    database_search_local = None

