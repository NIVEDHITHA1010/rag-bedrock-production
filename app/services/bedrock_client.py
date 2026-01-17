
"""
AWS Bedrock client for LLM and embedding operations.
Provides high-level interface for Bedrock API calls.
"""

import json
import boto3
from typing import List, Dict, Any, Optional
from botocore.exceptions import ClientError

from app.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


class BedrockClient:
    """Wrapper for AWS Bedrock API operations."""
    
    def __init__(
        self,
        region: Optional[str] = None,
        model_id: Optional[str] = None,
        embedding_model_id: Optional[str] = None
    ):
        """
        Initialize Bedrock client.
        
        Args:
            region: AWS region name
            model_id: Bedrock LLM model identifier
            embedding_model_id: Bedrock embedding model identifier
        """
        self.region = region or settings.AWS_REGION
        self.model_id = model_id or settings.BEDROCK_MODEL_ID
        self.embedding_model_id = embedding_model_id or settings.EMBEDDING_MODEL_ID
        
        # Initialize Bedrock runtime client
        self.client = boto3.client(
            service_name="bedrock-runtime",
            region_name=self.region
        )
        
        logger.info(f"Initialized Bedrock client in {self.region}")
        logger.info(f"LLM Model: {self.model_id}")
        logger.info(f"Embedding Model: {self.embedding_model_id}")
    
    def generate_text(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = None,
        max_tokens: int = None,
        top_p: float = None
    ) -> Dict[str, Any]:
        """
        Generate text using Claude via Bedrock.
        
        Args:
            prompt: User prompt
            system_prompt: System context/instructions
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
            
        Returns:
            Dict containing generated text and metadata
        """
        temperature = temperature or settings.LLM_TEMPERATURE
        max_tokens = max_tokens or settings.LLM_MAX_TOKENS
        top_p = top_p or settings.LLM_TOP_P
        
        # Construct Claude API request
        messages = [{"role": "user", "content": prompt}]
        
        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p
        }
        
        if system_prompt:
            request_body["system"] = system_prompt
        
        try:
            response = self.client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(request_body)
            )
            
            response_body = json.loads(response["body"].read())
            
            return {
                "text": response_body["content"][0]["text"],
                "stop_reason": response_body.get("stop_reason"),
                "usage": response_body.get("usage", {}),
                "model_id": self.model_id
            }
            
        except ClientError as e:
            logger.error(f"Bedrock API error: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during text generation: {e}")
            raise
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings using Bedrock Titan.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        embeddings = []
        
        for text in texts:
            try:
                request_body = {"inputText": text}
                
                response = self.client.invoke_model(
                    modelId=self.embedding_model_id,
                    body=json.dumps(request_body)
                )
                
                response_body = json.loads(response["body"].read())
                embedding = response_body.get("embedding", [])
                embeddings.append(embedding)
                
            except ClientError as e:
                logger.error(f"Error generating embedding for text: {e}")
                # Return zero vector on error
                embeddings.append([0.0] * settings.EMBEDDING_DIMENSION)
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                embeddings.append([0.0] * settings.EMBEDDING_DIMENSION)
        
        logger.info(f"Generated {len(embeddings)} embeddings")
        return embeddings
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate single embedding vector.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        return self.generate_embeddings([text])[0]
    
    async def generate_text_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = None,
        max_tokens: int = None
    ):
        """
        Generate text with streaming response (for future use).
        
        Args:
            prompt: User prompt
            system_prompt: System context
            temperature: Sampling temperature
            max_tokens: Max tokens to generate
            
        Yields:
            Text chunks as they're generated
        """
        # Note: Implement streaming when needed for real-time responses
        response = self.generate_text(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens
        )
        yield response["text"]
