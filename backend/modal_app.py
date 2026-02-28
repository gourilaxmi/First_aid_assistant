"""
Modal deployment for First Aid RAG Assistant
Run:  modal serve modal_app.py   (dev/testing)
      modal deploy modal_app.py  (production)
"""

import modal

# ---------------------------------------------------------------------------
# 1. App definition
# ---------------------------------------------------------------------------

app = modal.App("first-aid-rag")

# ---------------------------------------------------------------------------
# 2. BioBERT download (runs once at image build time)
# ---------------------------------------------------------------------------

def download_biobert():
    """Bake BioBERT into the image so it never re-downloads at runtime."""
    from transformers import AutoTokenizer, AutoModel
    print("Downloading BioBERT...")
    AutoTokenizer.from_pretrained("dmis-lab/biobert-v1.1")
    AutoModel.from_pretrained("dmis-lab/biobert-v1.1")
    print("BioBERT downloaded successfully!")


# ---------------------------------------------------------------------------
# 3. Container image
# ---------------------------------------------------------------------------

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "fastapi",
        "uvicorn[standard]",
        "python-multipart",
        "python-jose[cryptography]",
        "passlib[bcrypt]",
        "pydantic[email]",
        "python-dotenv",
        "torch",
        "transformers",
        "numpy",
        "pinecone",          # ← changed from pinecone-client to pinecone
        "groq",
        "pymongo",
        "certifi",
    ])
    .run_function(download_biobert)
    .add_local_dir(".", remote_path="/app", ignore=[
        "__pycache__",
        "**/__pycache__",
        "*.pyc",
        "*.pyo",
        ".git",
        ".gitignore",
        ".env",
        "venv",
        ".venv",
        "env",
        "*.log",
        "node_modules",
        "*.db",
        "*.sqlite3",
        ".pytest_cache",
        "**/.pytest_cache",
        ".mypy_cache",
        "dist",
        "build",
        "*.egg-info",
    ])
)

# ---------------------------------------------------------------------------
# 4. Secrets
# ---------------------------------------------------------------------------

secrets = [modal.Secret.from_name("first-aid-secrets")]

# ---------------------------------------------------------------------------
# 5. Function definition
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    secrets=secrets,
    memory=4096,
    cpu=2.0,
    timeout=120,
    min_containers=1   
)
@modal.asgi_app()
def fastapi_app():
    import sys
    sys.path.insert(0, "/app")
    from main import app
    return app