# RAG App (Python + LangChain + FAISS + Streamlit)

This project now supports:

1. Persistent multi-PDF knowledge base (local disk)
2. Metadata filtering by document source
3. Embedding generation with `all-MiniLM-L6-v2`
4. FAISS vector index for similarity search
5. Strict anti-hallucination RetrievalQA prompt
6. Source citations (file + page + snippet)
7. Conversation-aware responses and chat export

## Pre-requisite: Environment Setup (Windows PowerShell)

```powershell
cd "c:\Users\SRIKANTH\OneDrive\Documents\RAG"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If script execution is blocked in PowerShell:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## Configuration

Copy `.env.example` to `.env` and set keys for the provider you want.

Gemini (recommended):
- `GOOGLE_API_KEY`
- Optional: `GEMINI_MODEL` (default: `gemini-1.5-flash`)

OpenAI (optional):
- `OPENAI_API_KEY`
- Optional: `OPENAI_MODEL`

Bedrock (optional):
- Standard AWS credentials (`aws configure` or environment variables)
- `AWS_REGION`
- Optional: `BEDROCK_MODEL_ID`

## Phase 1 + 2: Build Local FAISS Index From PDFs

Option A (CLI ingestion):

```powershell
python ingest.py --pdf-folder data --index-dir faiss_index
```

Option B (recommended UI flow):
- Open the Streamlit app.
- Upload one or more PDFs.
- Click `Add to Knowledge Base`.
- PDFs are persisted under `kb_store/pdfs`.
- Index is persisted under `kb_store/faiss_index`.

## Phase 3: Retrieval QA Chain (Strict Grounding)

`rag_core.py` contains a strict system prompt:
- Uses only retrieved context.
- If answer is missing in context, responds:
  `I don't know based on the provided documents.`
- Supports source metadata filters at retrieval time.
- Supports Gemini, OpenAI, and Bedrock providers.

## Phase 4: Streamlit UI

```powershell
streamlit run app.py
```

Then upload a PDF and chat with it in real time.

In the app sidebar, you can:
- Filter retrieval by selected document names.
- Download conversation history as a Markdown file.
- Switch LLM provider and model (Gemini is default).

## Docker (Local)

Build and run with Docker Compose:

```powershell
docker compose up --build
```

App URL:
- `http://localhost:8501`

Notes:
- `compose.yaml` mounts `./kb_store` into the container so your uploaded PDFs/index persist.
- `.env` is loaded via `env_file`, so keep your provider API keys there.

Stop containers:

```powershell
docker compose down
```

## CI/CD + Auto Deploy (GitHub Actions)

This repo includes workflow:
- `.github/workflows/ci-cd.yml`

Pipeline behavior:
1. On PR to `main`: run tests.
2. On push to `main`: run tests, build Docker image, push to GHCR.
3. If deploy secrets are configured: SSH to VM and auto-deploy latest image.

### Required GitHub Secrets (for auto deploy)

Set these in `Settings -> Secrets and variables -> Actions`:

- `DEPLOY_HOST`: VM public IP/DNS
- `DEPLOY_USER`: SSH user
- `DEPLOY_SSH_KEY`: Private key content for SSH auth
- `DEPLOY_APP_DIR`: Remote folder path (example: `/opt/rag-app`)
- `REGISTRY_USERNAME`: GHCR username (usually your GitHub username)
- `REGISTRY_TOKEN`: GitHub token/PAT with `read:packages`

### Remote VM one-time setup

On your Linux VM, run once:

```bash
sudo mkdir -p /opt/rag-app/kb_store
sudo chown -R $USER:$USER /opt/rag-app
cat > /opt/rag-app/.env << 'EOF'
GOOGLE_API_KEY=your_key
GEMINI_MODEL=gemini-2.5-flash
EOF
```

After that, every push to `main` will update the running container automatically.
