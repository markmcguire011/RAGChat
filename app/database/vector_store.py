"""
Vector store operations for the chatbot.
This will handle document storage, retrieval, etc.
"""

import uuid
import logging
from typing import Optional, List, Dict, Any
import chromadb

from config import (
    DEFAULT_COLLECTION_NAME, DEFAULT_EMBEDDING_FUNCTION,
    EMBEDDINGS_DIR, DEFAULT_NUM_RESULTS
)
from .chroma_client import ChromaClient

logger = logging.getLogger(__name__)

class VectorStore:
    """
    Handles vector store operations for document storage and retrieval.
    """
    
    def __init__(
        self,
        chroma_client: Optional[ChromaClient] = None,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        persist_directory: Optional[str] = None,
        embedding_function_name: str = DEFAULT_EMBEDDING_FUNCTION,
    ):
        """
        Initialize the vector store.
        
        Args:
            chroma_client: An existing ChromaClient instance
            collection_name: Name of the collection to use
            persist_directory: Directory to persist the ChromaDB data (if no client provided)
            embedding_function_name: Name of the embedding function to use (if no client provided)
        """
        self.collection_name = collection_name
        
        # Use provided client or create a new one
        if chroma_client:
            self.chroma_client = chroma_client
        else:
            self.chroma_client = ChromaClient(
                persist_directory=persist_directory or EMBEDDINGS_DIR,
                embedding_function_name=embedding_function_name
            )
        
        self.collection = self.get_or_create_collection()
        logger.info(f"Vector store initialized with collection '{collection_name}'")
    
    def get_or_create_collection(self):
        """Get an existing collection or create a new one."""
        try:
            return self.chroma_client.client.get_collection(
                name=self.collection_name,
                embedding_function=self.chroma_client.embedding_function
            )
        except chromadb.errors.NotFoundError:
            return self.chroma_client.client.create_collection(
                name=self.collection_name,
                embedding_function=self.chroma_client.embedding_function
            )
    
    def add_documents(
        self,
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Add documents to the collection.
        
        Args:
            documents: list of document texts
            metadatas: list of metadata dictionaries for each document
            ids: list of IDs for each document
            
        Returns:
            list of document IDs
        """
        if ids is None:
            # generate IDs if not provided
            ids = [str(uuid.uuid4()) for _ in range(len(documents))]
        
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
        logger.info(f"Added {len(documents)} documents to collection '{self.collection_name}'")
        return ids
    
    def query(
        self,
        query_text: str,
        n_results: int = DEFAULT_NUM_RESULTS,
        where: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Query the collection for similar documents.
        
        Args:
            query_text: the query text
            n_results: number of results to return
            where: filter conditions for metadata
            where_document: filter conditions for document content
            
        Returns:
            query results containing documents, metadatas, distances, and ids
        """
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where=where,
            where_document=where_document
        )
        
        logger.debug(f"Query '{query_text}' returned {len(results['documents'][0])} results")
        return results
    
    def get_document(self, doc_id: str) -> Dict[str, Any]:
        """
        Get a document by ID.
        
        Args:
            doc_id: document ID
            
        Returns:
            document data
        """
        result = self.collection.get(ids=[doc_id])
        if not result["documents"]:
            raise ValueError(f"Document with ID {doc_id} not found")
        
        return {
            "document": result["documents"][0],
            "metadata": result["metadatas"][0] if result["metadatas"] else None,
            "id": doc_id
        }
    
    def delete_document(self, doc_id: str) -> None:
        """
        Delete a document by ID.
        
        Args:
            doc_id: Document ID
        """
        self.collection.delete(ids=[doc_id])
        logger.info(f"Deleted document with ID {doc_id}")
    
    def delete_collection(self) -> None:
        """Delete the entire collection."""
        self.chroma_client.client.delete_collection(name=self.collection_name)
        logger.warning(f"Deleted collection '{self.collection_name}'")
        
        # recreate collection
        self.collection = self.get_or_create_collection()

    def count(self):
        """Count the number of documents in the collection."""
        return self.collection.count()


if __name__ == "__main__":
    pass
