import os
import json
import logging
import requests
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('APIHandler')

class OpenRouterAPIHandler:
    """
    Handles all interactions with the OpenRouter API
    """
    
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OpenRouter API key not found. Please add it to the .env file.")
        
        self.base_url = "https://openrouter.ai/api/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def get_models(self):
        """
        Fetch available models from OpenRouter API
        
        Returns:
            list: List of model objects with details
        """
        url = f"{self.base_url}/models"
        logger.info(f"Fetching models from {url}")
        
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            
            data = response.json()
            if "data" in data:
                models = data["data"]
                logger.info(f"Received {len(models)} models from OpenRouter")
                
                # Log the first model as an example to understand the structure
                if models:
                    logger.info(f"Example model structure: {json.dumps(models[0], indent=2)}")
                
                # Filter out specific models that we don't want to use
                filtered_models = []
                excluded_model_ids = [
                    "claude-3-7", 
                    "google/gemini-2.0-flash-lite-001",
                    "anthropic/claude-3.7-sonnet:beta",
                    "anthropic/claude-3.7-sonnet"
                ]
                
                for model in models:
                    model_id = model.get("id", "").lower()
                    # Check if the model id contains any of the excluded model ids
                    if not any(excluded_id in model_id for excluded_id in excluded_model_ids):
                        # Enrich model data with pricing information
                        if "pricing" not in model:
                            model["pricing"] = {
                                "prompt": 0,
                                "completion": 0,
                                "is_free": True
                            }
                        else:
                            model["pricing"]["is_free"] = (
                                model["pricing"].get("prompt", 0) == 0 and 
                                model["pricing"].get("completion", 0) == 0
                            )
                        filtered_models.append(model)
                    else:
                        logger.info(f"Excluding model: {model.get('id')} ({model.get('name', '')})")
                
                logger.info(f"Filtered to {len(filtered_models)} models after excluding specified models")
                return filtered_models
            else:
                raise ValueError("Unexpected response format from OpenRouter API")
                
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error fetching models from OpenRouter: {str(e)}")
    
    def generate_text(self, model, prompt=None, max_tokens=1000, messages=None):
        """
        Generate text from a specific model
        
        Args:
            model (str or dict): Model ID to use or model object
            prompt (str, optional): Prompt text to send to the model (if messages not provided)
            max_tokens (int): Maximum tokens to generate
            messages (list, optional): List of message objects (overrides prompt if provided)
            
        Returns:
            dict: Dictionary containing response text and usage statistics
        """
        url = f"{self.base_url}/chat/completions"
        
        # Extract model ID if a model object was passed
        if isinstance(model, dict) and "id" in model:
            model_id = model["id"]
            logger.info(f"Using model ID {model_id} from model object")
        else:
            model_id = model
        
        # Use either provided messages or create from prompt
        if messages:
            payload_messages = messages
            logger.info(f"Using provided messages format with {len(messages)} messages")
        elif prompt:
            payload_messages = [{"role": "user", "content": prompt}]
            logger.info(f"Using single prompt format")
        else:
            raise ValueError("Either prompt or messages must be provided")
        
        payload = {
            "model": model_id,
            "messages": payload_messages,
            "max_tokens": max_tokens,
            "temperature": 0.7  # Add sensible temperature value
        }
        
        logger.info(f"Generating text with model: {model_id}, max_tokens: {max_tokens}")
        
        # Log prompt or messages length
        if prompt:
            logger.info(f"Prompt length: {len(prompt)} characters")
        elif messages:
            total_chars = sum(len(m.get("content", "")) for m in messages if isinstance(m, dict))
            logger.info(f"Total messages content length: {total_chars} characters")
        
        try:
            # Log request attempt - create a unique request ID
            if prompt:
                request_content = prompt
            else:
                request_content = str(messages)
            request_id = str(hash(request_content))[:8]  # Create a simple request ID for tracking
            logger.info(f"Request {request_id}: Sending request to {model_id}")
            
            response = requests.post(url, headers=self.headers, json=payload)
            
            # Log raw response status
            logger.info(f"Request {request_id}: Response received from {model}, status code: {response.status_code}")
            
            # Handle HTTP errors
            try:
                response.raise_for_status()
            except requests.exceptions.HTTPError as http_err:
                logger.error(f"Request {request_id}: HTTP error: {http_err}")
                logger.error(f"Response content: {response.text[:500]}...")
                raise ValueError(f"HTTP error when calling OpenRouter API: {http_err}")
            
            # Convert to JSON with error handling
            try:
                data = response.json()
                # Log a small preview of the data for debugging
                data_preview = str(data)[:200] + "..." if len(str(data)) > 200 else str(data)
                logger.info(f"Request {request_id}: Received data: {data_preview}")
            except json.JSONDecodeError as json_err:
                logger.error(f"Request {request_id}: JSON decode error: {json_err}")
                logger.error(f"Raw response: {response.text[:500]}...")
                raise ValueError(f"Invalid JSON response from OpenRouter API: {json_err}")
            
            # Detailed validation and error handling for API response
            if not data:
                logger.error(f"Request {request_id}: Empty response from API")
                raise ValueError("Empty response from OpenRouter API")
                
            if "error" in data:
                error_message = data["error"].get("message", "Unknown error")
                logger.error(f"Request {request_id}: API error: {error_message}")
                raise ValueError(f"OpenRouter API error: {error_message}")
                
            if "choices" not in data:
                logger.error(f"Request {request_id}: Missing 'choices' in API response: {data}")
                
                # Try to salvage any useful content from the response
                if isinstance(data, dict):
                    for key, value in data.items():
                        if isinstance(value, str) and len(value) > 10:
                            logger.warning(f"Request {request_id}: Using alternative field '{key}' as response")
                            return {
                                "response": value,
                                "usage": data.get("usage", {}),
                                "model": model,
                                "note": f"Using alternative field '{key}' due to missing 'choices'"
                            }
                
                raise ValueError("Malformed response from OpenRouter API: Missing 'choices'")
                
            if not data["choices"] or not isinstance(data["choices"], list):
                logger.error(f"Request {request_id}: Empty or invalid choices in API response: {data['choices']}")
                raise ValueError("No valid choices returned from OpenRouter API")
                
            # Extract the response content, handling different API formats
            first_choice = data["choices"][0]
            
            # Log the complete first_choice data for debugging
            logger.info(f"Request {request_id}: First choice structure: {json.dumps(first_choice, indent=2)}")
            
            # Try different response formats that various APIs might return
            content = None
            
            # Standard format
            if "message" in first_choice and isinstance(first_choice["message"], dict):
                message = first_choice["message"]
                logger.info(f"Request {request_id}: Message object: {json.dumps(message, indent=2)}")
                if "content" in message:
                    content = message["content"]
                    logger.info(f"Request {request_id}: Found content in standard format")
            
            # Alternative format 1
            if content is None and "text" in first_choice:
                content = first_choice["text"]
                logger.info(f"Request {request_id}: Found content in alternative format 'text'")
            
            # Alternative format 2
            if content is None and "content" in first_choice:
                content = first_choice["content"]
                logger.info(f"Request {request_id}: Found content in alternative format 'content'")
                
            # Alternative format 3 - check for 'value'
            if content is None and "value" in first_choice:
                content = first_choice["value"]
                logger.info(f"Request {request_id}: Found content in alternative format 'value'")
                
            # Last resort - try to extract any string field
            if content is None:
                logger.warning(f"Request {request_id}: Content not found in expected fields, trying to extract any string")
                for key, value in first_choice.items():
                    if isinstance(value, str) and len(value) > 10:
                        content = value
                        logger.warning(f"Request {request_id}: Using field '{key}' as content")
                        break
            
            # If still no content, log the entire response for debugging and raise a specific error
            if content is None:
                # Log the entire response data structure for debugging 
                logger.error(f"Request {request_id}: Could not extract any content from response. Full data structure: {json.dumps(data, indent=2)}")
                raise ValueError("Could not extract content from API response - no recognized content format")
                
            # Ensure content is a string
            if not isinstance(content, str):
                logger.warning(f"Request {request_id}: Content is not a string, converting from {type(content)}")
                content = str(content)
                
            if content is None or content.strip() == "":
                logger.error(f"Request {request_id}: Received empty content from API")
                raise ValueError("Empty content received from API")
            
            # Log successful response info
            usage = data.get("usage", {})
            logger.info(f"Request {request_id}: Response generated successfully. Content length: {len(content)}")
            if usage:
                logger.info(f"Request {request_id}: Usage - Prompt tokens: {usage.get('prompt_tokens', 'N/A')}, " +
                           f"Completion tokens: {usage.get('completion_tokens', 'N/A')}, " +
                           f"Total tokens: {usage.get('total_tokens', 'N/A')}")
            
            return {
                "response": content,
                "usage": usage,
                "model": model,
                "request_id": request_id
            }
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error with {model}: {str(e)}", exc_info=True)
            raise Exception(f"Error generating text from {model}: {str(e)}")
        except ValueError as e:
            # Re-raise already logged validation errors
            raise
        except Exception as e:
            logger.error(f"Unexpected error with {model}: {str(e)}", exc_info=True)
            raise Exception(f"Unexpected error with {model}: {str(e)}")
    
    def calculate_cost(self, model, prompt_tokens, completion_tokens):
        """
        Calculate the cost of an API call
        
        Args:
            model (dict): Model object with pricing information
            prompt_tokens (int): Number of tokens in the prompt
            completion_tokens (int): Number of tokens in the completion
            
        Returns:
            float: Estimated cost in USD
        """
        if "pricing" not in model:
            return 0
        
        pricing = model["pricing"]
        prompt_cost = pricing.get("prompt", 0) * prompt_tokens / 1000
        completion_cost = pricing.get("completion", 0) * completion_tokens / 1000
        
        return prompt_cost + completion_cost
