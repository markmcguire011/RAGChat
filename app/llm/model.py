"""
LLM setup and configuration.
This module handles the initialization and configuration of language models.
"""

import os
import logging
from typing import Dict, Any, Optional, List, Union
from langchain_core.language_models import BaseLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import BaseOutputParser
from langchain.chains import LLMChain
from langchain_community.llms import OpenAI, HuggingFacePipeline
from langchain.callbacks.manager import CallbackManager

logger = logging.getLogger(__name__)

class LLMFactory:
    """
    Factory class for creating and configuring language models.
    """
    
    @staticmethod
    def create_llm(
        provider: str = "openai",
        model_name: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> BaseLLM:
        """
        Create a language model based on the specified provider.
        
        Args:
            provider: The LLM provider ("openai", "huggingface", etc.)
            model_name: Name of the model to use
            temperature: Temperature for generation (higher = more creative)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional arguments for the specific LLM
            
        Returns:
            Configured LLM instance
        """
        if provider == "openai":
            if not os.environ.get("OPENAI_API_KEY"):
                raise ValueError("OPENAI_API_KEY environment variable is required")
            
            model_name = model_name or "gpt-3.5-turbo-instruct"
            
            return OpenAI(
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            
        elif provider == "huggingface":
            try:
                from transformers import pipeline
            except ImportError:
                raise ImportError("Please install transformers: pip install transformers")
            
            model_name = model_name or "google/flan-t5-base"
            
            # Create HF pipeline
            pipe = pipeline(
                "text-generation",
                model=model_name,
                **kwargs
            )
            
            return HuggingFacePipeline(
                pipeline=pipe,
                model_kwargs={"temperature": temperature, "max_length": max_tokens}
            )
            
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
    
    @staticmethod
    def create_chain(
        llm: BaseLLM,
        prompt_template: Union[str, PromptTemplate],
        output_parser: Optional[BaseOutputParser] = None,
        verbose: bool = False
    ) -> LLMChain:
        """
        Create an LLMChain with the specified LLM and prompt.
        
        Args:
            llm: The language model
            prompt_template: Prompt template string or PromptTemplate object
            output_parser: Optional parser for structured output
            verbose: Whether to enable verbose output
            
        Returns:
            Configured LLMChain
        """
        # string to PromptTemplate if needed
        if isinstance(prompt_template, str):
            # extract input variables from the template
            import re
            input_variables = re.findall(r'\{([^{}]*)\}', prompt_template)
            prompt_template = PromptTemplate(
                template=prompt_template,
                input_variables=input_variables
            )
        
        return LLMChain(
            llm=llm,
            prompt=prompt_template,
            output_parser=output_parser,
            verbose=verbose
        )

class ModelManager:
    """
    Manages LLM instances and provides utility methods for working with models.
    """
    
    def __init__(self):
        self.models: Dict[str, BaseLLM] = {}
        self.chains: Dict[str, LLMChain] = {}
    
    def load_model(
        self,
        model_id: str,
        provider: str = "openai",
        model_name: Optional[str] = None,
        **kwargs
    ) -> BaseLLM:
        """
        Load and cache a language model.
        
        Args:
            model_id: Identifier for the model
            provider: The LLM provider
            model_name: Name of the model to use
            **kwargs: Additional arguments for the LLM
            
        Returns:
            The loaded LLM
        """
        if model_id not in self.models:
            logger.info(f"Loading model {model_id} ({provider}/{model_name})")
            self.models[model_id] = LLMFactory.create_llm(
                provider=provider,
                model_name=model_name,
                **kwargs
            )
        return self.models[model_id]
    
    def create_chain(
        self,
        chain_id: str,
        model_id: str,
        prompt_template: Union[str, PromptTemplate],
        output_parser: Optional[BaseOutputParser] = None,
        verbose: bool = False
    ) -> LLMChain:
        """
        Create and cache an LLMChain.
        
        Args:
            chain_id: Identifier for the chain
            model_id: Identifier for the model to use
            prompt_template: Prompt template
            output_parser: Optional output parser
            verbose: Whether to enable verbose output
            
        Returns:
            The created LLMChain
        """
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not loaded")
        
        if chain_id not in self.chains:
            logger.info(f"Creating chain {chain_id} with model {model_id}")
            self.chains[chain_id] = LLMFactory.create_chain(
                llm=self.models[model_id],
                prompt_template=prompt_template,
                output_parser=output_parser,
                verbose=verbose
            )
        
        return self.chains[chain_id]
    
    def get_model(self, model_id: str) -> BaseLLM:
        """Get a loaded model by ID."""
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not loaded")
        return self.models[model_id]
    
    def get_chain(self, chain_id: str) -> LLMChain:
        """Get a created chain by ID."""
        if chain_id not in self.chains:
            raise ValueError(f"Chain {chain_id} not created")
        return self.chains[chain_id]

# global model manager instance
model_manager = ModelManager()
