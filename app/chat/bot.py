"""
Main chatbot logic
weaves together LLM, retrieval, and prompt components.
"""

import logging
from typing import List, Dict, Any, Optional, Union

from app.llm.model import model_manager
from app.llm.prompt import prompt_manager
from app.retrieval.retriever import DocumentRetriever
from app.database.vector_store import VectorStore
from langchain_core.documents import Document
from langchain_chroma import Chroma

logger = logging.getLogger(__name__)

class WiLineBot:
    """
    Main chatbot class that handles user interactions and generates responses.
    """
    
    def __init__(
        self,
        model_id: str = "default",
        retriever: Optional[DocumentRetriever] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        provider: str = "openai",
        model_name: str = "gpt-3.5-turbo-instruct",
    ):
        """
        Initialize the chatbot.
        
        Args:
            model_id: Identifier for the model to use
            retriever: Document retriever for context augmentation
            system_prompt: Custom system prompt (uses default if None)
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
            provider: Provider for the model
            model_name: Name of the model
        """
        self.model_id = model_id
        self.retriever = retriever
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # initialize model if not already loaded
        if model_id not in model_manager.models:
            logger.info(f"Loading model {model_id}")
            model_manager.load_model(
                model_id=model_id,
                provider=provider,
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens
            )
        
        if system_prompt:
            # custom system prompt if provided
            prompt_manager.register_chat_template(
                f"custom_{model_id}",
                system_message=system_prompt,
                human_message="{query}"
            )
            self.prompt_id = f"custom_{model_id}"
        else:
            # default chat prompt
            self.prompt_id = "chat_with_context" if self.retriever else "default_chat"
            
            # Register chat_with_context template if using retrieval
            if self.retriever and "chat_with_context" not in prompt_manager.templates:
                prompt_manager.register_chat_template(
                    "chat_with_context",
                    system_message="You are a helpful assistant. Use the following context to answer the user's question.",
                    human_message="Context:\n{context}\n\nQuestion: {query}"
                )
            
            # default chat prompt if not using retrieval
            if not self.retriever and "default_chat" not in prompt_manager.templates:
                prompt_manager.register_chat_template(
                    "default_chat",
                    system_message="You are a helpful assistant.",
                    human_message="{query}"
                )
        
        # create the chain
        self.chain_id = f"chain_{model_id}"
        model_manager.create_chain(
            chain_id=self.chain_id,
            model_id=model_id,
            prompt_template=prompt_manager.get_template(self.prompt_id),
            output_parser=prompt_manager.get_parser("str"),
            verbose=False
        )
        
        logger.info(f"ChatBot initialized with model {model_id}")
        
    def get_context(self, query: str) -> List[Document]:
        """
        Retrieve relevant context for the query.
        
        Args:
            query: User query
            
        Returns:
            List of relevant documents
        """
        if not self.retriever:
            return []
        
        try:
            return self.retriever.retrieve(query)
        except Exception as e:
            logger.error(f"Error retrieving context: {str(e)}")
            return []
    
    def format_context(self, documents: List[Document]) -> str:
        """
        Format retrieved documents into a context string.
        
        Args:
            documents: List of retrieved documents
            
        Returns:
            Formatted context string
        """
        if not documents:
            return ""
        
        context_parts = []
        for i, doc in enumerate(documents):
            # metadata if available
            metadata_str = ""
            if hasattr(doc, "metadata") and doc.metadata:
                metadata_str = " | ".join(f"{k}: {v}" for k, v in doc.metadata.items())
                metadata_str = f"[{metadata_str}]\n"
            
            # document content
            context_parts.append(f"Document {i+1}:\n{metadata_str}{doc.page_content}\n")
        
        return "\n".join(context_parts)
    
    def generate_response(
        self,
        query: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        use_retrieval: bool = True
    ) -> Dict[str, Any]:
        """
        Generate a response to the user query.
        
        Args:
            query: User query
            chat_history: Optional chat history
            use_retrieval: Whether to use retrieval for context
            
        Returns:
            Response object containing the answer and metadata
        """
        logger.info(f"Generating response for query: {query}")
        
        # context if retrieval is enabled
        context = ""
        retrieved_docs = []
        if use_retrieval and self.retriever:
            retrieved_docs = self.get_context(query)
            context = self.format_context(retrieved_docs)
            logger.debug(f"Retrieved context: {context[:100]}...")
        
        # Prepare input variables based on the prompt template
        input_vars = {"query": query}
        
        # Only add context if we have it and the prompt template expects it
        if context and "context" in prompt_manager.get_template(self.prompt_id).input_variables:
            input_vars["context"] = context
        
        # chat history if provided
        if chat_history and "chat_history" in prompt_manager.get_template(self.prompt_id).input_variables:
            formatted_history = self._format_chat_history(chat_history)
            input_vars["chat_history"] = formatted_history
        
        # get the chain
        chain = model_manager.get_chain(self.chain_id)
        
        # generate response
        try:
            response = chain.invoke(input_vars)
            
            # response object
            result = {
                "answer": response,
                "query": query,
                "sources": [self._document_to_source(doc) for doc in retrieved_docs] if retrieved_docs else []
            }
            
            logger.info(f"Generated response: {response[:100]}...")
            return result
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return {
                "answer": "I'm sorry, I encountered an error while generating a response.",
                "query": query,
                "error": str(e)
            }
    
    def _format_chat_history(self, chat_history: List[Dict[str, str]]) -> str:
        """
        Format chat history for inclusion in the prompt.
        
        Args:
            chat_history: List of message dictionaries
            
        Returns:
            Formatted chat history string
        """
        formatted = []
        for message in chat_history:
            role = message.get("role", "unknown")
            content = message.get("content", "")
            
            if role == "user":
                formatted.append(f"User: {content}")
            elif role == "assistant":
                formatted.append(f"Assistant: {content}")
            else:
                formatted.append(f"{role.capitalize()}: {content}")
        
        return "\n".join(formatted)
    
    def _document_to_source(self, doc: Document) -> Dict[str, Any]:
        """
        Convert a Document to a source dictionary.
        
        Args:
            doc: Document object
            
        Returns:
            Source dictionary
        """
        source = {
            "content": doc.page_content,
        }
        
        # metadata if available
        if hasattr(doc, "metadata") and doc.metadata:
            source["metadata"] = doc.metadata
        
        return source
    
    @classmethod
    def from_vector_store(
        cls,
        vector_store: VectorStore,
        embedding_model: Any,
        model_id: str = "default",
        **kwargs
    ) -> "WiLineBot":
        """
        Create a WiLineBot instance with a retriever from a vector store.
        
        Args:
            vector_store: Vector store for document retrieval
            embedding_model: Embedding model for the retriever
            model_id: Identifier for the LLM
            **kwargs: Additional arguments for ChatBot initialization
            
        Returns:
            Initialized ChatBot instance
        """
        # wrapper for embedding function
        class EmbeddingWrapper:
            def __init__(self, embedding_function):
                self.embedding_function = embedding_function
                
            def embed_query(self, text):
                return self.embedding_function([text])[0]
                
            def embed_documents(self, documents):
                return self.embedding_function(documents)
        
        if not hasattr(vector_store.chroma_client.embedding_function, 'embed_query'):
            wrapped_embeddings = EmbeddingWrapper(vector_store.chroma_client.embedding_function)
        else:
            wrapped_embeddings = vector_store.chroma_client.embedding_function
        
        # VectorStore to a langchain VectorStore
        langchain_vector_store = Chroma(
            client=vector_store.chroma_client.client,
            collection_name=vector_store.collection_name,
            embedding_function=wrapped_embeddings
        )
        
        retriever = DocumentRetriever(
            vector_store=langchain_vector_store,
            embeddings=wrapped_embeddings,  # wrapped embeddings
            score_threshold=None 
        )
        
        return cls(
            model_id=model_id,
            retriever=retriever,
            **kwargs
        )

default_bot = WiLineBot(model_id="default", provider="huggingface", model_name="gpt2")
