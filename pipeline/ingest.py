"""
CodeLens ingestion pipeline: process feature JSON data into Qdrant vectors.

This pipeline:
1. Loads feature JSON files (Jira tickets, Slack conversations)
2. Formats them as structured text documents with metadata
3. Embeds them using local nomic-embed-text
4. Stores vectors + payloads in local Qdrant

Usage:
    cd codelens
    docker compose up -d  # Start Qdrant
    uv run python pipeline/ingest.py
"""

import os
import sys
import json
import uuid
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv

# Add parent directory to path for shared imports
sys.path.insert(0, str(Path(__file__).parent.parent))

load_dotenv(Path(__file__).parent.parent / ".env")

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    PayloadSchemaType,
)

from shared.embeddings import init_embeddings, get_embedding

# Paths
DATA_DIR = Path(__file__).parent.parent / "data" / "features"
MODELS_DIR = Path(__file__).parent.parent.parent / "models"
EMBED_MODEL_PATH = MODELS_DIR / "nomic-embed-text" / "nomic-embed-text-v1.5.f16.gguf"

# Qdrant config
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = "codelens_documents"
VECTOR_SIZE = 768  # nomic-embed-text dimension


def get_qdrant_client() -> QdrantClient:
    """Create Qdrant client."""
    return QdrantClient(url=QDRANT_URL)


def ensure_collection(client: QdrantClient):
    """Create collection if it doesn't exist."""
    collections = client.get_collections().collections
    exists = any(c.name == COLLECTION_NAME for c in collections)
    
    if not exists:
        print(f"Creating collection: {COLLECTION_NAME}")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
        # Create payload indexes for filtering
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="source_type",
            field_schema=PayloadSchemaType.KEYWORD,
        )
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="feature_id",
            field_schema=PayloadSchemaType.KEYWORD,
        )
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="entity_type",
            field_schema=PayloadSchemaType.KEYWORD,
        )
    else:
        print(f"Collection {COLLECTION_NAME} already exists")


def format_jira_ticket(ticket: dict, feature_id: str) -> dict:
    """Format a Jira ticket as a document with metadata."""
    acceptance_criteria = "\n".join(f"  - {ac}" for ac in ticket.get("acceptance_criteria", []))
    references = "\n".join(f"  - {ref}" for ref in ticket.get("references", []))
    
    text = f"""JIRA Ticket: {ticket['id']}
Type: {ticket['type']}
Title: {ticket['title']}
Status: {ticket['status']}

Description:
{ticket['description']}

Acceptance Criteria:
{acceptance_criteria}

References:
{references}

Created: {ticket['created_at']}
Updated: {ticket['updated_at']}"""

    return {
        "id": str(uuid.uuid4()),
        "text": text,
        "source_type": "jira",
        "source_id": ticket["id"],
        "feature_id": feature_id,
        "title": ticket["title"],
        "timestamp": ticket["created_at"],
        "status": ticket["status"],
        "entity_type": "ticket",
        "references": ticket.get("references", []),
    }


def format_slack_message(message: dict, channel: str, feature_id: str) -> dict:
    """Format a Slack message as a document with metadata."""
    text = f"""Slack Message in {channel}
From: {message['user']}
Time: {message['timestamp']}

{message['message']}"""

    return {
        "id": str(uuid.uuid4()),
        "text": text,
        "source_type": "slack",
        "source_id": f"{channel}_{message['timestamp']}",
        "feature_id": feature_id,
        "channel": channel,
        "user": message["user"],
        "timestamp": message["timestamp"],
        "entity_type": "message",
        "message_text": message["message"],
    }


def extract_entities(data: dict, feature_id: str) -> list[dict]:
    """Extract entities (people, features) from the data for graph visualization."""
    entities = []
    seen_users = set()
    
    # Extract feature entity
    entities.append({
        "id": str(uuid.uuid4()),
        "text": f"Feature: {data['metadata']['dataset_name']}\n\n{data['metadata']['description']}",
        "source_type": "feature",
        "source_id": feature_id,
        "feature_id": feature_id,
        "entity_type": "feature",
        "name": data["metadata"]["dataset_name"],
        "timestamp": data["metadata"]["created_at"],
    })
    
    # Extract people from Slack messages
    for convo in data.get("slack_conversations", []):
        for msg in convo.get("messages", []):
            user = msg["user"]
            if user not in seen_users:
                seen_users.add(user)
                entities.append({
                    "id": str(uuid.uuid4()),
                    "text": f"Team Member: {user}\n\nContributor to {feature_id} feature discussions.",
                    "source_type": "person",
                    "source_id": user,
                    "feature_id": feature_id,
                    "entity_type": "person",
                    "name": user,
                    "timestamp": msg["timestamp"],
                })
    
    return entities


def load_feature_data(json_path: Path) -> dict:
    """Load feature JSON data."""
    with open(json_path, "r") as f:
        return json.load(f)


def ingest_feature(client: QdrantClient, json_path: Path):
    """Ingest a single feature JSON file into Qdrant."""
    print(f"\nIngesting: {json_path.name}")
    
    data = load_feature_data(json_path)
    feature_id = data["metadata"]["dataset_name"]
    
    documents = []
    
    # Process Jira tickets
    for ticket in data.get("jira_tickets", []):
        doc = format_jira_ticket(ticket, feature_id)
        documents.append(doc)
        print(f"  + Jira: {ticket['id']} - {ticket['title'][:50]}...")
    
    # Process Slack conversations
    for convo in data.get("slack_conversations", []):
        channel = convo["channel"]
        for msg in convo.get("messages", []):
            doc = format_slack_message(msg, channel, feature_id)
            documents.append(doc)
            print(f"  + Slack: {msg['user']} in {channel}")
    
    # Extract entities for graph
    entities = extract_entities(data, feature_id)
    documents.extend(entities)
    print(f"  + Extracted {len(entities)} entities (feature, people)")
    
    # Embed and store
    print(f"\nEmbedding {len(documents)} documents...")
    points = []
    
    for i, doc in enumerate(documents):
        text = doc.pop("text")
        doc_id = doc.pop("id")
        
        # Get embedding
        vector = get_embedding(text)
        
        # Store text in payload
        doc["text"] = text
        
        points.append(PointStruct(
            id=doc_id,
            vector=vector,
            payload=doc,
        ))
        
        if (i + 1) % 5 == 0:
            print(f"  Embedded {i + 1}/{len(documents)}")
    
    # Upsert to Qdrant
    print(f"\nUpserting {len(points)} points to Qdrant...")
    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"Done! Collection now has {client.count(COLLECTION_NAME).count} points.")


def main():
    """Main ingestion entry point."""
    print("=" * 60)
    print("CodeLens Ingestion Pipeline")
    print("=" * 60)
    
    # Initialize embedding model
    print(f"\nInitializing embedding model: {EMBED_MODEL_PATH}")
    init_embeddings(str(EMBED_MODEL_PATH))
    
    # Connect to Qdrant
    print(f"\nConnecting to Qdrant: {QDRANT_URL}")
    client = get_qdrant_client()
    
    # Ensure collection exists
    ensure_collection(client)
    
    # Find all feature JSON files
    json_files = list(DATA_DIR.glob("*.json"))
    if not json_files:
        print(f"\nNo JSON files found in {DATA_DIR}")
        return
    
    print(f"\nFound {len(json_files)} feature file(s)")
    
    # Ingest each feature
    for json_path in json_files:
        ingest_feature(client, json_path)
    
    print("\n" + "=" * 60)
    print("Ingestion complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
