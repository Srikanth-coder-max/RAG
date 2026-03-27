import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_aws import ChatBedrock


STRICT_QA_SYSTEM_PROMPT = """
You are a document-grounded assistant.
Use only the context provided below to answer the question.
Treat chat history as conversational context only, not as factual evidence.
If the answer is not explicitly present in the context, say:
"I don't know based on the provided documents."
Do not use outside knowledge. Do not guess.

Context:
{context}
""".strip()


def get_embedding_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    return HuggingFaceEmbeddings(model_name=model_name)


def load_vectorstore(index_dir: str, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
    embeddings = get_embedding_model(embedding_model)
    return FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)


def get_llm(provider: str = "gemini", model: str | None = None, temperature: float = 0.0):
    load_dotenv()
    provider = provider.lower()

    if provider == "openai":
        chosen_model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        return ChatOpenAI(model=chosen_model, temperature=temperature)

    if provider == "gemini":
        chosen_model = model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        deprecated = {
            "gemini-1.5-flash",
            "models/gemini-1.5-flash",
            "gemini-2.0-flash",
            "models/gemini-2.0-flash",
        }
        if chosen_model in deprecated:
            chosen_model = "gemini-2.5-flash"
        return ChatGoogleGenerativeAI(model=chosen_model, temperature=temperature)

    if provider == "bedrock":
        chosen_model = model or os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-5-sonnet-20240620-v1:0")
        region = os.getenv("AWS_REGION", "us-east-1")
        return ChatBedrock(model_id=chosen_model, region_name=region, model_kwargs={"temperature": temperature})

    raise ValueError("provider must be one of: gemini, openai, bedrock")


def build_retrieval_qa_chain(vectorstore, llm, k: int = 4, source_filters: list[str] | None = None):
    search_kwargs = {"k": k}

    if source_filters:
        allowed_sources = {Path(s).name for s in source_filters}
        search_kwargs["filter"] = lambda md: Path(str(md.get("source", ""))).name in allowed_sources

    retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", STRICT_QA_SYSTEM_PROMPT),
            (
                "human",
                "Conversation so far:\n{chat_history}\n\nQuestion: {input}",
            ),
        ]
    )

    def _format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    answer_chain = prompt | llm | StrOutputParser()

    # LCEL chain: retrieve docs, then generate grounded answer
    rag_chain = RunnablePassthrough.assign(
        context=lambda x: retriever.invoke(x["input"]),
    ) | RunnablePassthrough.assign(
        answer=lambda x: answer_chain.invoke(
            {
                "context": _format_docs(x["context"]),
                "input": x["input"],
                "chat_history": x.get("chat_history", ""),
            }
        )
    )
    return rag_chain


def format_citations(context_docs):
    citations = []
    for d in context_docs or []:
        source = d.metadata.get("source", "unknown")
        page = d.metadata.get("page", "?")
        snippet = " ".join(d.page_content.split())[:220]
        citations.append(
            {
                "source": source,
                "page": page,
                "snippet": snippet,
            }
        )
    return citations
