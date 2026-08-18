import json
from typing import cast

import chromadb
from chromadb.utils import embedding_functions
from langchain_core.tools import tool
from prompt_toolkit import document

# 1. Використовуємо PersistentClient і вказуємо шлях до папки
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# 2. Опціонально: задаємо модель ембедингів (наприклад, OpenAI)
efunc = cast(
    chromadb.EmbeddingFunction,
    embedding_functions.SentenceTransformerEmbeddingFunction(model_name="BAAI/bge-m3"),
)


def new_kb(name: str = "domain_knowledge"):
    knowledge_base = chroma_client.get_or_create_collection(
        name=name, embedding_function=efunc
    )
    return knowledge_base


@tool("search_info")
def search_info(query: str) -> str:
    """Пошук інформації у базі знань.

    Використовуйте цей інструмент,
    коли потрібна інформація з предметної області.

    Args:
        query: Пошуковий запит.

    Returns:
        Топ-3 релевантних документів з бази знань.
    """
    knowledge_base = new_kb()
    results = knowledge_base.query(query_texts=[query], n_results=3)
    if results["documents"]:
        docs = results["documents"][0]
        return "\n---\n".join(docs)
    else:
        return "не знайдено релевантних документів"
