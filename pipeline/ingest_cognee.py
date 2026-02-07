"""
CodeLens ingestion pipeline using cognee for knowledge graph generation.

This pipeline:
1. Loads feature JSON files (Jira tickets, Slack conversations)
2. Feeds them to cognee.add()
3. Runs cognee.cognify() to extract entities, relationships, and build knowledge graph
4. Stores everything in Qdrant via cognee's vector adapter

Usage:
    cd codelens
    docker compose up -d  # Start Qdrant
    # Set OPENAI_API_KEY in .env for cognify LLM
    uv run python pipeline/ingest_cognee.py
"""

import os
import sys
import json
import asyncio
from pathlib import Path

from dotenv import load_dotenv

# Add parent directory to path for shared imports
sys.path.insert(0, str(Path(__file__).parent.parent))

load_dotenv(Path(__file__).parent.parent / ".env")

# Paths
DATA_DIR = Path(__file__).parent.parent / "data" / "features"

# Qdrant config
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")


def format_jira_ticket(ticket: dict, feature_name: str) -> str:
    """Format a Jira ticket as structured text for cognee ingestion."""
    acceptance_criteria = "\n".join(f"  - {ac}" for ac in ticket.get("acceptance_criteria", []))
    references = "\n".join(f"  - {ref}" for ref in ticket.get("references", []))
    
    return f"""JIRA Ticket: {ticket['id']}
Feature: {feature_name}
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
Updated: {ticket['updated_at']}
"""


def format_slack_message(message: dict, channel: str, feature_name: str) -> str:
    """Format a Slack message as structured text for cognee ingestion."""
    return f"""Slack Message
Feature: {feature_name}
Channel: {channel}
From: {message['user']}
Timestamp: {message['timestamp']}

Message:
{message['message']}
"""


def format_feature_metadata(metadata: dict) -> str:
    """Format feature metadata for cognee ingestion."""
    return f"""Feature Overview
Name: {metadata['dataset_name']}
Description: {metadata['description']}
Created: {metadata['created_at']}
"""


async def setup_cognee():
    """Configure cognee to use local Qdrant."""
    import cognee
    from cognee.infrastructure.databases.vector import get_vector_engine
    
    print("Configuring cognee...")
    
    # Set vector DB to Qdrant
    cognee.config.set_vector_db_provider("qdrant")
    cognee.config.set_vector_db_url(QDRANT_URL)
    print(f"  Vector DB: Qdrant at {QDRANT_URL}")
    
    # Check for LLM API key (required for cognify)
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
    if api_key:
        cognee.config.set_llm_api_key(api_key)
        provider = os.getenv("LLM_PROVIDER", "openai")
        model = os.getenv("LLM_MODEL", "gpt-4o-mini")
        cognee.config.set_llm_provider(provider)
        cognee.config.set_llm_model(model)
        print(f"  LLM: {provider} / {model}")
        return True
    else:
        print("  WARNING: No LLM_API_KEY or OPENAI_API_KEY found in .env")
        print("  cognee.cognify() requires an LLM to extract entities/relationships")
        print("  Add OPENAI_API_KEY=sk-xxx to .env file")
        return False


async def ingest_with_cognee(json_path: Path, has_llm: bool):
    """Ingest a feature JSON file using cognee's ECL pipeline."""
    import cognee
    
    print(f"\n{'='*60}")
    print(f"Ingesting: {json_path.name}")
    print(f"{'='*60}")
    
    # Load JSON data
    with open(json_path) as f:
        data = json.load(f)
    
    feature_name = data["metadata"]["dataset_name"]
    
    # Collect all documents
    documents = []
    
    # Add feature metadata
    doc = format_feature_metadata(data["metadata"])
    documents.append(doc)
    print(f"  + Feature metadata: {feature_name}")
    
    # Add Jira tickets
    for ticket in data.get("jira_tickets", []):
        doc = format_jira_ticket(ticket, feature_name)
        documents.append(doc)
        print(f"  + Jira: {ticket['id']} - {ticket['title'][:40]}...")
    
    # Add Slack messages
    for convo in data.get("slack_conversations", []):
        channel = convo["channel"]
        for msg in convo.get("messages", []):
            doc = format_slack_message(msg, channel, feature_name)
            documents.append(doc)
            print(f"  + Slack: {msg['user']} in {channel}")
    
    print(f"\nTotal documents: {len(documents)}")
    
    # Add documents to cognee
    print("\nAdding documents to cognee...")
    
    # Combine into a single text for cognee
    combined_text = "\n\n---DOCUMENT SEPARATOR---\n\n".join(documents)
    
    await cognee.add(combined_text, dataset_name=feature_name)
    print(f"  Added {len(documents)} documents as dataset: {feature_name}")
    
    # Run cognify to extract knowledge graph
    if has_llm:
        print("\nRunning cognee.cognify() to extract entities and relationships...")
        print("  (This uses LLM to identify people, concepts, and their connections)")
        
        try:
            await cognee.cognify()
            print("  Knowledge graph generated successfully!")
        except Exception as e:
            print(f"  WARNING: cognify() failed: {e}")
            print("  Documents were added but knowledge graph was not extracted.")
    else:
        print("\nSkipping cognify() - no LLM API key configured.")
        print("  To enable knowledge graph extraction, add OPENAI_API_KEY to .env")


async def main():
    """Main entry point."""
    print("="*60)
    print("CodeLens Cognee Ingestion Pipeline")
    print("="*60)
    
    # Setup cognee
    has_llm = await setup_cognee()
    
    # Find all feature JSON files
    json_files = list(DATA_DIR.glob("*.json"))
    if not json_files:
        print(f"\nNo JSON files found in {DATA_DIR}")
        return
    
    print(f"\nFound {len(json_files)} feature file(s)")
    
    # Import cognee here after setup
    import cognee
    
    # Clear previous data
    print("\nClearing previous cognee data...")
    try:
        await cognee.prune.prune_data()
        await cognee.prune.prune_system(metadata=True)
        print("  Data cleared.")
    except Exception as e:
        print(f"  Note: {e}")
    
    # Ingest each feature
    for json_path in json_files:
        await ingest_with_cognee(json_path, has_llm)
    
    # Test search
    print("\n" + "="*60)
    print("Testing cognee search...")
    print("="*60)
    
    try:
        from cognee.api.v1.search import SearchType
        results = await cognee.search(
            query_text="score boosting",
            query_type=SearchType.CHUNKS,
        )
        print(f"  Found {len(results)} results for 'score boosting'")
        if results:
            print(f"  Sample: {str(results[0])[:150]}...")
    except Exception as e:
        print(f"  Search test: {e}")
    
    print("\n" + "="*60)
    print("Ingestion complete!")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())
