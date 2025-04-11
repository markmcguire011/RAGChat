from typing import List, Optional, Dict, Any
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain.vectorstores import VectorStore
from langchain.embeddings.base import Embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
import logging

from app.config import (
    DEFAULT_NUM_RESULTS, SIMILARITY_THRESHOLD, EMBEDDINGS_DIR
)

logger = logging.getLogger(__name__)

class DocumentRetriever:
    """
    Handles document retrieval operations using langchain.
    """
    
    def __init__(
        self,
        vector_store: VectorStore,
        embeddings: Embeddings,
        k: int = DEFAULT_NUM_RESULTS,
        search_type: str = "similarity",
        score_threshold: Optional[float] = SIMILARITY_THRESHOLD,
    ):
        """
        Initialize the document retriever.
        
        Args:
            vector_store: the vector store containing document embeddings
            embeddings: the embeddings model to use
            k: number of documents to retrieve (default from config)
            search_type: type of search to perform ("similarity", "mmr", etc.)
            score_threshold: minimum relevance score for retrieved documents (default from config)
        """
        self.vector_store = vector_store
        self.embeddings = embeddings
        self.k = k
        self.search_type = search_type
        self.score_threshold = score_threshold
        self.retriever = self._create_retriever()
    
    def _create_retriever(self) -> BaseRetriever:
        """Create a retriever from the vector store."""
        search_kwargs = {"k": self.k}
        
        # Only add score_threshold if it's supported by the vector store
        # Chroma doesn't support score_threshold in the query
        if self.score_threshold is not None and not isinstance(self.vector_store, Chroma):
            search_kwargs["score_threshold"] = self.score_threshold
            
        return self.vector_store.as_retriever(
            search_type=self.search_type,
            search_kwargs=search_kwargs
        )
    
    def retrieve(self, query: str, top_k: int = None) -> List[Document]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: The query string
            top_k: Number of documents to retrieve (uses default if None)
            
        Returns:
            List of relevant documents
        """
        try:
            documents = self.retriever.invoke(
                query,
                {"k": top_k} if top_k else None
            )
            
            logger.info(f"Retrieved {len(documents)} documents for query")
            return documents
        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            return []
    
    def retrieve_with_scores(self, query: str) -> List[tuple[Document, float]]:
        """
        Retrieve relevant documents with their relevance scores.
        
        Args:
            query: The search query
            
        Returns:
            List of tuples containing (document, score)
        """
        if hasattr(self.vector_store, "similarity_search_with_score"):
            return self.vector_store.similarity_search_with_score(
                query, k=self.k
            )
        else:
            logger.warning("Vector store doesn't support retrieval with scores")
            return [(doc, 1.0) for doc in self.retrieve(query)]

    @classmethod
    def from_documents(
        cls,
        documents: List[Document],
        embeddings: Embeddings,
        vector_store_cls: type[VectorStore],
        vector_store_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> "DocumentRetriever":
        """
        Create a DocumentRetriever from a list of documents.
        
        Args:
            documents: List of documents to index
            embeddings: Embeddings model to use
            vector_store_cls: Vector store class to use
            vector_store_kwargs: Additional arguments for vector store initialization
            **kwargs: Additional arguments for DocumentRetriever initialization
            
        Returns:
            Initialized DocumentRetriever
        """
        vector_store_kwargs = vector_store_kwargs or {}
        
        # embeddings directory from config
        if "persist_directory" not in vector_store_kwargs and hasattr(vector_store_cls, "from_documents"):
            vector_store_kwargs["persist_directory"] = EMBEDDINGS_DIR
            
        vector_store = vector_store_cls.from_documents(
            documents, embeddings, **vector_store_kwargs
        )
        return cls(vector_store=vector_store, embeddings=embeddings, **kwargs)
    
    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embeddings: Embeddings,
        vector_store_cls: type[VectorStore],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        **kwargs
    ) -> "DocumentRetriever":
        """
        Create a DocumentRetriever from a list of text strings.
        
        Args:
            texts: list of text strings to index
            embeddings: embeddings model to use
            vector_store_cls: vector store class to use
            metadatas: optional metadata for each text
            chunk_size: size of text chunks for splitting
            chunk_overlap: overlap between chunks
            **kwargs: additional arguments for DocumentRetriever initialization
            
        Returns:
            Initialized DocumentRetriever
        """
        # chunking
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        if metadatas:
            documents = []
            for i, text in enumerate(texts):
                chunks = text_splitter.split_text(text)
                for chunk in chunks:
                    documents.append(Document(
                        page_content=chunk,
                        metadata=metadatas[i]
                    ))
        else:
            documents = text_splitter.create_documents(texts)
            
        return cls.from_documents(
            documents=documents,
            embeddings=embeddings,
            vector_store_cls=vector_store_cls,
            **kwargs
        )
