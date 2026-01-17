"""
Prompt templates for RAG question answering.
Structured prompts for optimal LLM performance.
"""

from typing import Optional


class RAGPromptTemplates:
    """Collection of prompt templates for RAG operations."""
    
    @staticmethod
    def get_system_prompt() -> str:
        """
        Get system prompt for the RAG assistant.
        
        Returns:
            System prompt string
        """
        return """You are a helpful AI assistant specializing in answering questions based on provided documents. 

Your responsibilities:
1. Answer questions accurately using ONLY the information from the provided context
2. If the context doesn't contain enough information, clearly state this
3. Cite specific document references when possible
4. Be concise but comprehensive
5. Never make up information not present in the context

Response format:
- Provide a clear, direct answer
- Support your answer with relevant quotes or references from the documents
- If uncertain, acknowledge the limitations of the available information"""
    
    @staticmethod
    def get_qa_prompt(question: str, context: str) -> str:
        """
        Generate question-answering prompt with context.
        
        Args:
            question: User's question
            context: Retrieved document context
            
        Returns:
            Formatted prompt string
        """
        return f"""Based on the following context documents, please answer the question.

Context Documents:
{context}

Question: {question}

Instructions:
- Answer the question using ONLY information from the context above
- If the context doesn't contain relevant information, say so clearly
- Include brief citations (e.g., "According to Document 1...")
- Be factual and precise

Answer:"""
    
    @staticmethod
    def get_summarization_prompt(text: str, max_words: int = 150) -> str:
        """
        Generate document summarization prompt.
        
        Args:
            text: Text to summarize
            max_words: Maximum words in summary
            
        Returns:
            Formatted prompt string
        """
        return f"""Please provide a concise summary of the following text in approximately {max_words} words.

Text:
{text}

Summary:"""
    
    @staticmethod
    def get_followup_prompt(
        question: str,
        previous_answer: str,
        context: str
    ) -> str:
        """
        Generate prompt for follow-up questions.
        
        Args:
            question: Follow-up question
            previous_answer: Previous answer in conversation
            context: Retrieved context
            
        Returns:
            Formatted prompt string
        """
        return f"""You previously answered a question. Here is that exchange:

Previous Answer:
{previous_answer}

Now, based on the following additional context, please answer this follow-up question:

Context:
{context}

Follow-up Question: {question}

Answer:"""
    
    @staticmethod
    def get_multihop_prompt(
        question: str,
        contexts: list,
        reasoning_required: bool = True
    ) -> str:
        """
        Generate prompt for multi-hop reasoning questions.
        
        Args:
            question: Complex question requiring multiple documents
            contexts: List of context strings from different sources
            reasoning_required: Whether to show reasoning steps
            
        Returns:
            Formatted prompt string
        """
        context_text = "\n\n".join([
            f"Source {i+1}:\n{ctx}" 
            for i, ctx in enumerate(contexts)
        ])
        
        reasoning_instruction = ""
        if reasoning_required:
            reasoning_instruction = """
Before providing the final answer:
1. Identify which sources are relevant
2. Explain how you're combining information
3. Show your reasoning process
"""
        
        return f"""Answer the following question that may require synthesizing information from multiple sources.

{context_text}

Question: {question}
{reasoning_instruction}
Answer:"""
    
    @staticmethod
    def get_extraction_prompt(
        text: str,
        fields: list,
        output_format: str = "json"
    ) -> str:
        """
        Generate prompt for structured information extraction.
        
        Args:
            text: Source text
            fields: List of fields to extract
            output_format: Output format (json, table, etc.)
            
        Returns:
            Formatted prompt string
        """
        fields_str = ", ".join(fields)
        
        return f"""Extract the following information from the text below: {fields_str}

Text:
{text}

Please provide the extracted information in {output_format} format.

Extracted Information:"""
    
    @staticmethod
    def get_verification_prompt(
        claim: str,
        context: str
    ) -> str:
        """
        Generate prompt for fact verification.
        
        Args:
            claim: Claim to verify
            context: Supporting context
            
        Returns:
            Formatted prompt string
        """
        return f"""Verify whether the following claim is supported by the provided context.

Claim: {claim}

Context:
{context}

Provide your verification in this format:
- Verdict: [SUPPORTED / NOT SUPPORTED / INSUFFICIENT INFORMATION]
- Explanation: [Brief explanation of your reasoning]
- Relevant Evidence: [Quote relevant parts if supported]

Verification:"""
