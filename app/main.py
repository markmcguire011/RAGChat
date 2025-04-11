import os
import logging
from pathlib import Path
import argparse
from dotenv import load_dotenv

load_dotenv()

from app.config import (
    RAW_DATA_DIR, DEFAULT_EMBEDDING_FUNCTION, 
    DEFAULT_COLLECTION_NAME, DEFAULT_NUM_RESULTS,
    LOG_FORMAT, LOG_LEVEL
)
from app.database.chroma_client import ChromaClient
from app.database.vector_store import VectorStore
from app.chat.bot import WiLineBot

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

def setup_vector_store():
    """Set up and return the vector store."""
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
    
    return vector_store

def setup_chatbot(vector_store):
    """Set up and return the chatbot with the vector store."""
    embedding_model = None
    
    return WiLineBot.from_vector_store(
        vector_store=vector_store,
        embedding_model=embedding_model,
        model_id="retrieval_bot",
        temperature=0.7
    )

def interactive_chat(chatbot):
    """Run an interactive chat session with the chatbot."""
    print("\n🤖 Welcome to the RAG Chatbot!")
    print("Type 'exit', 'quit', or 'q' to end the conversation.\n")
    
    chat_history = []
    
    while True:
        user_input = input("You: ")
        
        # exit commands
        if user_input.lower() in ['exit', 'quit', 'q']:
            print("🤖 Goodbye! Have a great day!")
            break
        
        # gen response
        response = chatbot.generate_response(
            query=user_input,
            chat_history=chat_history
        )
        
        print("\n🤖:", response["answer"])
        
        # sources if available
        if response.get("sources") and len(response["sources"]) > 0:
            print("\nSources:")
            for i, source in enumerate(response["sources"]):
                if source.get("metadata"):
                    print(f"  {i+1}. {source['metadata'].get('title', 'Unknown')}")
                    print(f"     Source: {source['metadata'].get('source', 'Unknown')}")
                else:
                    preview = source['content'][:100] + "..." if len(source['content']) > 100 else source['content']
                    print(f"  {i+1}. {preview}")
        
        print()
        
        # update chat history
        chat_history.append({"role": "user", "content": user_input})
        chat_history.append({"role": "assistant", "content": response["answer"]})

def main():
    parser = argparse.ArgumentParser(description="WiLine RAG Chatbot with ChromaDB and LangChain")
    parser.add_argument("--demo", action="store_true", help="Run the vector store demo")
    parser.add_argument("--chat", action="store_true", help="Start interactive chat")
    
    args = parser.parse_args()
    
    vector_store = setup_vector_store()
    
    if args.demo:
        # demo
        run_test_queries(vector_store)
        logger.info("Demo completed")
    elif args.chat:
        # chat
        chatbot = setup_chatbot(vector_store)
        interactive_chat(chatbot)
    else:
        # default: show help
        parser.print_help()
    
    logger.info("Application terminated")

if __name__ == "__main__":
    main()
