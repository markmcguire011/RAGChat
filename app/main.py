import os
import logging
from pathlib import Path

from config import (
    RAW_DATA_DIR, DEFAULT_EMBEDDING_FUNCTION, 
    DEFAULT_COLLECTION_NAME, DEFAULT_NUM_RESULTS,
    LOG_FORMAT, LOG_LEVEL
)
from database.chroma_client import ChromaClient
from database.vector_store import VectorStore

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format=LOG_FORMAT
)
logger = logging.getLogger(__name__)

def load_sample_documents(vector_store, num_samples=5):
    """
    Load sample documents from the raw data directory into the vector store.
    
    Args:
        vector_store: The VectorStore instance
        num_samples: Number of sample documents to load
        
    Returns:
        Number of documents loaded
    """
    raw_dir = Path(RAW_DATA_DIR)
    
    if not raw_dir.exists():
        logger.warning(f"Raw data directory not found: {raw_dir}")
        return 0
    
    md_files = list(raw_dir.glob("**/*.md"))
    logger.info(f"Found {len(md_files)} markdown files")
    
    # only get a couple
    sample_files = md_files[:num_samples]
    
    docs = []
    metadatas = []
    
    for file_path in sample_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                
                # try to get title from frontmatter
                title = file_path.stem
                if content.startswith('---'):
                    frontmatter_end = content.find('---', 3)
                    if frontmatter_end != -1:
                        frontmatter = content[3:frontmatter_end]
                        for line in frontmatter.split('\n'):
                            if line.startswith('title:'):
                                title = line.split('title:')[1].strip().strip('"\'')
                                break
                
                docs.append(content)
                metadatas.append({
                    "source": str(file_path),
                    "title": title,
                    "category": file_path.parent.name
                })
                logger.info(f"Added: {title} from {file_path}")
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
    
    if docs:
        vector_store.add_documents(documents=docs, metadatas=metadatas)
    
    return len(docs)

def run_test_queries(vector_store):
    """
    Run test queries against the vector store.
    
    Args:
        vector_store: The VectorStore instance
    """
    test_queries = [
        "How do I handle irate customers?",
        "What is the CSAP process?",
        "Tell me about customer service best practices",
        "What is the escalation procedure for support tickets?"
    ]
    
    logger.info("\nRunning test queries:")
    
    for query in test_queries:
        results = vector_store.query(
            query_text=query,
            n_results=DEFAULT_NUM_RESULTS
        )
        
        print(f"\nQuery: '{query}'")
        for i, (doc, metadata, distance) in enumerate(zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        )):
            doc_preview = doc[:200] + "..." if len(doc) > 200 else doc
            print(f"{i+1}. Title: {metadata.get('title', 'Unknown')}")
            print(f"   Source: {metadata.get('source', 'Unknown')}")
            print(f"   Category: {metadata.get('category', 'Unknown')}")
            print(f"   Distance: {distance}")
            print(f"   Preview: {doc_preview}")
            print()

def main():
    """Main function to demonstrate the vector store functionality."""
    logger.info("Initializing ChromaDB client...")
    chroma_client = ChromaClient(
        embedding_function_name=DEFAULT_EMBEDDING_FUNCTION
    )
    
    logger.info("Initializing vector store...")
    vector_store = VectorStore(
        chroma_client=chroma_client,
        collection_name=DEFAULT_COLLECTION_NAME
    )
    
    # check if we have documents already
    doc_count = vector_store.count()
    logger.info(f"Current document count: {doc_count}")
    
    if doc_count == 0:
        num_loaded = load_sample_documents(vector_store)
        logger.info(f"Loaded {num_loaded} sample documents")
    
    run_test_queries(vector_store)
    
    logger.info("Demo done")

if __name__ == "__main__":
    main()
