import os
import json
from datetime import datetime
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS

from kb import build_vectorstore_from_pdfs, save_vectorstore
from rag_core import build_retrieval_qa_chain, format_citations, get_embedding_model, get_llm


st.set_page_config(page_title="RAG PDF Chat", page_icon="R", layout="wide")
load_dotenv()

KB_DIR = Path("kb_store")
KB_PDF_DIR = KB_DIR / "pdfs"
KB_INDEX_DIR = KB_DIR / "faiss_index"


def _ensure_kb_dirs():
    KB_PDF_DIR.mkdir(parents=True, exist_ok=True)
    KB_INDEX_DIR.mkdir(parents=True, exist_ok=True)


def _list_kb_sources():
    return sorted(p.name for p in KB_PDF_DIR.glob("*.pdf"))


def _save_uploaded_pdf(uploaded_file):
    target = KB_PDF_DIR / uploaded_file.name
    target.write_bytes(uploaded_file.getvalue())
    return str(target)


def _load_persistent_vectorstore():
    index_file = KB_INDEX_DIR / "index.faiss"
    if not index_file.exists():
        return None
    embeddings = get_embedding_model("sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local(str(KB_INDEX_DIR), embeddings, allow_dangerous_deserialization=True)


def _format_chat_history(messages):
    lines = []
    for msg in messages[-10:]:
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines) if lines else "No prior conversation."


def _export_chat_markdown(messages):
    rows = ["# Chat History", f"Generated: {datetime.now().isoformat()}", ""]
    for msg in messages:
        role = "User" if msg["role"] == "user" else "Assistant"
        rows.append(f"## {role}")
        rows.append(msg["content"])
        rows.append("")
    return "\n".join(rows)

st.title("RAG PDF Chat")
st.caption("Upload one or more PDFs and ask grounded questions with citations.")

_ensure_kb_dirs()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = _load_persistent_vectorstore()
if "source_filters" not in st.session_state:
    st.session_state.source_filters = []

with st.sidebar:
    st.header("Model Settings")
    provider = st.selectbox("LLM Provider", ["gemini", "openai", "bedrock"], index=0)

    if provider == "gemini":
        default_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    elif provider == "openai":
        default_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    else:
        default_model = os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-5-sonnet-20240620-v1:0")

    model_id = st.text_input("Model", value=default_model)
    if provider == "gemini" and model_id in {"gemini-1.5-flash", "models/gemini-1.5-flash", "gemini-2.0-flash", "models/gemini-2.0-flash"}:
        st.info("Selected model is deprecated/unsupported for this endpoint. Using gemini-2.5-flash.")
        model_id = "gemini-2.5-flash"

    uploaded_pdfs = st.file_uploader("Upload PDF(s)", type=["pdf"], accept_multiple_files=True)
    rebuild_clicked = st.button("Add to Knowledge Base")

    st.divider()
    st.subheader("Metadata Filters")
    current_sources = _list_kb_sources()
    selected_sources = st.multiselect(
        "Restrict retrieval to selected documents",
        options=current_sources,
        default=current_sources,
    )
    st.session_state.source_filters = selected_sources

    st.divider()
    st.subheader("Chat Export")
    export_data = _export_chat_markdown(st.session_state.messages)
    st.download_button(
        "Download Chat (.md)",
        data=export_data,
        file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
        mime="text/markdown",
    )
    st.download_button(
        "Download Chat (.json)",
        data=json.dumps(st.session_state.messages, indent=2),
        file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json",
    )

if rebuild_clicked:
    if not uploaded_pdfs:
        st.warning("Please upload at least one PDF.")
    else:
        with st.spinner("Updating persistent knowledge base..."):
            saved_paths = [_save_uploaded_pdf(f) for f in uploaded_pdfs]
            all_pdf_paths = [str(p) for p in KB_PDF_DIR.glob("*.pdf")]

            vectorstore, chunks = build_vectorstore_from_pdfs(all_pdf_paths)
            save_vectorstore(vectorstore, str(KB_INDEX_DIR))

            st.session_state.vectorstore = vectorstore
            st.session_state.messages = []
        st.success(f"Knowledge base updated with {len(saved_paths)} new PDF(s), {len(chunks)} chunks indexed.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("citations"):
            with st.expander("Sources"):
                for c in msg["citations"]:
                    st.markdown(f"- **{c['source']}** (page {c['page']}): {c['snippet']}...")

user_prompt = st.chat_input("Ask a question about your knowledge base")
if user_prompt:
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        if st.session_state.vectorstore is None:
            answer = "Please add at least one PDF to the knowledge base first."
            citations = []
        elif _list_kb_sources() and not st.session_state.source_filters:
            answer = "Please select at least one document in Metadata Filters."
            citations = []
        else:
            with st.spinner("Thinking..."):
                try:
                    llm = get_llm(provider=provider, model=model_id, temperature=0.0)
                    rag_chain = build_retrieval_qa_chain(
                        st.session_state.vectorstore,
                        llm,
                        k=4,
                        source_filters=st.session_state.source_filters,
                    )
                    result = rag_chain.invoke(
                        {
                            "input": user_prompt,
                            "chat_history": _format_chat_history(st.session_state.messages),
                        }
                    )
                    answer = result.get("answer", "I don't know based on the provided documents.")
                    citations = format_citations(result.get("context", []))
                except Exception as exc:
                    answer = "Model call failed. Check your provider/model/API key configuration."
                    citations = []
                    st.error(str(exc))

        st.markdown(answer)
        if citations:
            with st.expander("Sources"):
                for c in citations:
                    st.markdown(f"- **{c['source']}** (page {c['page']}): {c['snippet']}...")

        st.session_state.messages.append({"role": "assistant", "content": answer, "citations": citations})
