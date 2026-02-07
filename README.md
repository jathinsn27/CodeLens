# CodeLens - Feature Intelligence Explorer

An interactive tool for non-technical team members to explore feature history from Jira, Slack, and code. Ask natural language questions and get narrative explanations backed by evidence.

## Stack

- **Qdrant** (Docker) - Vector database for semantic search
- **Qwen3-4B** (local GGUF) - LLM for RAG answers
- **nomic-embed-text** (local GGUF) - Embeddings
- **FastAPI** - Backend API
- **D3.js** - Knowledge graph visualization

## Quick Start

### 1. Start Qdrant

```bash
cd codelens
docker compose up -d
```

Verify it's running:
```bash
curl http://localhost:6333/collections
```

### 2. Install Dependencies

```bash
uv sync
```

Or with pip:
```bash
pip install fastapi uvicorn qdrant-client llama-cpp-python python-dotenv numpy requests
```

### 3. Ingest Data

```bash
uv run python pipeline/ingest.py
```

This loads the sample feature data (Jira tickets, Slack conversations) into Qdrant.

### 4. Start the App

```bash
uv run python app.py
```

Open http://localhost:8888 in your browser.

## Usage

### Ask Questions

Type natural language questions like:
- "Why was score boosting added?"
- "Who worked on hybrid queries?"
- "What user feedback led to this feature?"

The app retrieves relevant context from Qdrant and generates a narrative answer using the local LLM.

### Explore the Timeline

The left panel shows a chronological timeline of all events (Jira updates, Slack messages). Click any event to see details.

### Knowledge Graph

The right panel shows a force-directed graph of entities (people, tickets, features) and their relationships. Drag nodes to explore.

## Adding Your Own Data

Create JSON files in `data/features/` following this format:

```json
{
  "metadata": {
    "dataset_name": "my-feature-id",
    "description": "Description of the feature",
    "created_at": "2026-02-07T00:00:00Z"
  },
  "jira_tickets": [
    {
      "id": "JIRA-123",
      "type": "Story",
      "title": "Feature title",
      "description": "Full description...",
      "acceptance_criteria": ["AC1", "AC2"],
      "status": "Done",
      "created_at": "2026-02-01T10:00:00Z",
      "updated_at": "2026-02-05T10:00:00Z",
      "references": ["https://docs.example.com/feature"]
    }
  ],
  "slack_conversations": [
    {
      "channel": "#engineering",
      "messages": [
        {
          "timestamp": "2026-02-01T09:00:00Z",
          "user": "alice",
          "message": "Starting work on the new feature..."
        }
      ]
    }
  ]
}
```

Then re-run the ingestion:
```bash
uv run python pipeline/ingest.py
```

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /` | Interactive UI |
| `GET /api/search?q=...` | Semantic search with Prefetch + RRF Fusion |
| `GET /api/ask?q=...` | RAG Q&A with LLM |
| `GET /api/timeline` | Chronological events |
| `GET /api/graph` | Entity-relationship graph data |
| `GET /api/features` | List all features |
| `GET /api/evidence/{id}` | Get document details |

## Requirements

- Python 3.11+
- Docker (for Qdrant)
- ~8GB RAM (for local LLM)

## Models

The app uses local GGUF models from `../models/`:
- `Qwen3-4B-Q4_K_M/Qwen3-4B-Q4_K_M.gguf` - LLM
- `nomic-embed-text/nomic-embed-text-v1.5.f16.gguf` - Embeddings
