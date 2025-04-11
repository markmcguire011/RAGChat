"""
Prompt templates and management.
This module provides utilities for creating and managing prompt templates.
"""

import logging
from typing import Dict, List, Any, Optional, Union
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import BaseOutputParser, StrOutputParser

logger = logging.getLogger(__name__)

# system prompt
DEFAULT_SYSTEM_PROMPT = """You are a helpful AI assistant. Answer the user's questions 
based on the provided context. If you don't know the answer, say so - don't make up information."""

class PromptManager:
    """
    Manages prompt templates for the application.
    """
    
    def __init__(self):
        self.templates: Dict[str, Union[PromptTemplate, ChatPromptTemplate]] = {}
        self.parsers: Dict[str, BaseOutputParser] = {}
        
        # default parsers
        self.parsers["str"] = StrOutputParser()
    
    def register_template(
        self,
        template_id: str,
        template: Union[str, PromptTemplate, ChatPromptTemplate],
        input_variables: Optional[List[str]] = None,
        template_format: str = "f-string",
        validate_template: bool = True
    ) -> Union[PromptTemplate, ChatPromptTemplate]:
        """
        Register a prompt template.
        
        Args:
            template_id: Identifier for the template
            template: Template string or PromptTemplate object
            input_variables: List of input variables (if template is a string)
            template_format: Format of the template string
            validate_template: Whether to validate the template
            
        Returns:
            The registered template
        """
        if isinstance(template, str):
            if not input_variables:
                # try to extract input variables from f-string format
                import re
                input_variables = re.findall(r'\{([^{}]*)\}', template)
            
            template = PromptTemplate(
                template=template,
                input_variables=input_variables,
                template_format=template_format,
                validate_template=validate_template
            )
        
        self.templates[template_id] = template
        logger.info(f"Registered prompt template: {template_id}")
        return template
    
    def register_chat_template(
        self,
        template_id: str,
        system_message: str = DEFAULT_SYSTEM_PROMPT,
        human_message: str = "{query}",
        input_variables: Optional[List[str]] = None
    ) -> ChatPromptTemplate:
        """
        Register a chat prompt template.
        
        Args:
            template_id: Identifier for the template
            system_message: System message template
            human_message: Human message template
            input_variables: List of input variables
            
        Returns:
            The registered chat template
        """
        if not input_variables:
            # extract input variables from both templates
            import re
            system_vars = re.findall(r'\{([^{}]*)\}', system_message)
            human_vars = re.findall(r'\{([^{}]*)\}', human_message)
            input_variables = list(set(system_vars + human_vars))
        
        template = ChatPromptTemplate.from_messages([
            ("system", system_message),
            ("human", human_message)
        ])
        
        self.templates[template_id] = template
        logger.info(f"Registered chat prompt template: {template_id}")
        return template
    
    def register_parser(
        self,
        parser_id: str,
        parser: BaseOutputParser
    ) -> BaseOutputParser:
        """
        Register an output parser.
        
        Args:
            parser_id: Identifier for the parser
            parser: The output parser
            
        Returns:
            The registered parser
        """
        self.parsers[parser_id] = parser
        logger.info(f"Registered output parser: {parser_id}")
        return parser
    
    def get_template(
        self,
        template_id: str
    ) -> Union[PromptTemplate, ChatPromptTemplate]:
        """
        Get a registered template by ID.
        
        Args:
            template_id: Identifier for the template
            
        Returns:
            The prompt template
        """
        if template_id not in self.templates:
            raise ValueError(f"Template {template_id} not found")
        return self.templates[template_id]
    
    def get_parser(self, parser_id: str) -> BaseOutputParser:
        """
        Get a registered parser by ID.
        
        Args:
            parser_id: Identifier for the parser
            
        Returns:
            The output parser
        """
        if parser_id not in self.parsers:
            raise ValueError(f"Parser {parser_id} not found")
        return self.parsers[parser_id]
    
    def format_prompt(
        self,
        template_id: str,
        **kwargs
    ) -> str:
        """
        Format a prompt template with the provided variables.
        
        Args:
            template_id: Identifier for the template
            **kwargs: Variables to format the template with
            
        Returns:
            The formatted prompt string
        """
        template = self.get_template(template_id)
        return template.format(**kwargs)

# Common prompt templates
QA_TEMPLATE = """
Context information is below:
---------------------
{context}
---------------------

Given the context information and not prior knowledge, answer the following question:
{question}
"""

SUMMARIZATION_TEMPLATE = """
Please summarize the following text:
---------------------
{text}
---------------------

Summary:
"""

CHAT_WITH_CONTEXT_TEMPLATE = """
You are a helpful AI assistant. Use the following context to answer the user's question.
If you don't know the answer based on the provided context, say so - don't make up information.

Context:
---------------------
{context}
---------------------
"""

# Global prompt manager instance
prompt_manager = PromptManager()

# Register default templates
prompt_manager.register_template(
    "qa",
    QA_TEMPLATE,
    input_variables=["context", "question"]
)

prompt_manager.register_template(
    "summarize",
    SUMMARIZATION_TEMPLATE,
    input_variables=["text"]
)

prompt_manager.register_chat_template(
    "chat_with_context",
    system_message=CHAT_WITH_CONTEXT_TEMPLATE,
    human_message="{query}",
    input_variables=["context", "query"]
)
