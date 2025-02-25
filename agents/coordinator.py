import os
import json
import logging
from dotenv import load_dotenv
from utils.api_handler import OpenRouterAPIHandler

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('Coordinator')

class Coordinator:
    """
    The Coordinator agent manages the flow of queries, analyzes query types,
    assigns tasks to appropriate agents, and evaluates the final response.
    """
    
    def __init__(self, model_labels_data, coordinator_model="anthropic/claude-3-opus-20240229"):
        """
        Initialize the coordinator agent
        
        Args:
            model_labels_data (list): List containing model-label mappings
            coordinator_model (str): Model ID to use for coordination tasks (default: Claude-3-Opus)
        """
        load_dotenv()
        self.api_handler = OpenRouterAPIHandler()
        self.model_labels_data = model_labels_data
        
        # Extract all unique labels for analysis
        self.available_labels = set()
        for model_info in self.model_labels_data:
            if "labels" in model_info:
                self.available_labels.update(model_info["labels"])
        
        logger.info(f"Coordinator initialized with {len(self.available_labels)} unique labels: {self.available_labels}")
        
        # Use the model specified by the user for coordination tasks
        self.coordinator_model = coordinator_model
        logger.info(f"Using {self.coordinator_model} as coordinator model")

    def analyze_query(self, query):
        """
        Analyze the query and identify relevant tags/labels
        
        Args:
            query (str): User query text
            
        Returns:
            list: List of labels that are relevant to the query
        """
        logger.info(f"Analyzing query: '{query}'")
        logger.info(f"Using coordinator model: {self.coordinator_model}")
        
        # Create a prompt for the coordinator to analyze the query
        analysis_prompt = f"""
        You are a query analyzer for a multi-agent AI system. Your task is to analyze the following query 
        and determine the most relevant categories or tags from the given list.

        Query: "{query}"

        Available labels (you MUST ONLY choose from these EXACT labels):
        {list(self.available_labels)}

        ⚠️ CRITICAL RULES ⚠️
        1. You MUST ONLY select labels from the list above. DO NOT invent new labels or variations.
        2. The system will COMPLETELY FAIL if you return any label not in the list above.
        3. If a label is not on the list, DO NOT USE IT even if it seems appropriate.
        4. Double check each label against the available labels before including it.
        
        Label Selection Guidelines:
        - For logic puzzles, deductive tasks, sequence tasks and pattern recognition, ALWAYS include BOTH 'reasoning_expert' AND 'math_expert' labels
        - For any puzzle involving arrangement, order, colors, people positioning, or logical constraints, include BOTH 'reasoning_expert' AND 'math_expert'
        - For math problems and calculations, ALWAYS include 'math_expert'
        - For programming and software development questions, include 'code_expert'
        - Ensure that logic puzzles (e.g. Zebra puzzles, arrangement problems, "who sits where", "who wears what color") get BOTH 'reasoning_expert' AND 'math_expert'
        - Use 'general_assistant' as a fallback for general knowledge questions
        - Use multiple labels when appropriate (e.g., a math problem that requires reasoning)
        
        Return ONLY a JSON array of relevant labels from the available list, ranking them in order of relevance.
        For example: ["reasoning_expert", "math_expert"]
        
        Your response should:
        1. Contain ONLY labels from the provided list
        2. Include 2-3 most relevant labels for this query
        3. Be formatted as a valid JSON array
        4. NOT include any explanations, only the JSON array
        
        Triple-check that EVERY label you include EXISTS EXACTLY as written in the available labels list.
        """
        
        logger.info(f"Sending request to OpenRouter API with model {self.coordinator_model}")
        
        try:
            # Get analysis from the model
            response = self.api_handler.generate_text(
                self.coordinator_model, 
                analysis_prompt,
                max_tokens=500
            )
            
            # Extract labels from the response
            content = response["response"]
            logger.info(f"Received analysis response from {self.coordinator_model}: {content}")
            
            # Try to extract JSON array from the response
            import re
            json_match = re.search(r'\[.*?\]', content, re.DOTALL)
            
            if json_match:
                try:
                    labels = json.loads(json_match.group(0))
                    # Validate labels against available labels
                    valid_labels = [label for label in labels if label in self.available_labels]
                    
                    if not valid_labels:
                        logger.warning(f"No valid labels found in model response. Response had: {labels}")
                        logger.warning("Using fallback mechanism for label selection")
                        # Use the same fallback code that's defined below for JSON extraction failures
                        # Create a list of available labels for easier checking
                        query_lower = query.lower()
                        fallback_labels = []
                        
                        available_labels_list = list(self.available_labels)
                        logger.info(f"Available labels for fallback: {available_labels_list}")
                        
                        # Map common keywords to specific available labels (same mapping as below)
                        keyword_mappings = {
                            # Math related
                            "math": ["math_expert", "reasoning_expert"],
                            "calculate": ["math_expert", "reasoning_expert"],
                            "equation": ["math_expert", "reasoning_expert"],
                            "solve": ["math_expert", "reasoning_expert", "code_expert"],
                            
                            # Logic/reasoning related 
                            "logic": ["reasoning_expert", "math_expert"],  # Mantık için math ekledik
                            "puzzle": ["reasoning_expert", "math_expert"],  # Puzzles için math ekledik
                            "problem": ["reasoning_expert", "math_expert"],
                            "analyze": ["reasoning_expert", "math_expert"],  # Analiz için math ekledik
                            "deduction": ["reasoning_expert", "math_expert"],  # Dedüktif düşünme için math ekledik
                            "inference": ["reasoning_expert", "math_expert"],  # Çıkarım için math ekledik
                            "reasoning": ["reasoning_expert", "math_expert"],  # Akıl yürütme için math ekledik
                            "deduce": ["reasoning_expert", "math_expert"],
                            "infer": ["reasoning_expert", "math_expert"],
                            "order": ["reasoning_expert", "math_expert"],
                            "arrangement": ["reasoning_expert", "math_expert"],
                            "sequence": ["reasoning_expert", "math_expert"],
                            "pattern": ["reasoning_expert", "math_expert"],
                            "if": ["reasoning_expert", "math_expert"],
                            "then": ["reasoning_expert", "math_expert"],
                            "wear": ["reasoning_expert", "math_expert"],  # Kıyafet ilgili puzzles için math ekledik
                            "color": ["reasoning_expert", "math_expert"],  # Renk ilgili puzzles için math ekledik 
                            "dress": ["reasoning_expert", "math_expert"],  # Elbise ilgili puzzles için math ekledik
                            
                            # Code related
                            "code": ["code_expert"],
                            "program": ["code_expert"],
                            "function": ["code_expert"],
                            "algorithm": ["code_expert"],
                            "programming": ["code_expert"],
                            "developer": ["code_expert"],
                            
                            # Writing related
                            "writing": ["creative_writer"],
                            "write": ["creative_writer"],
                            "essay": ["creative_writer"],
                            "story": ["creative_writer"],
                            "summarize": ["creative_writer", "general_assistant"],
                            "article": ["creative_writer", "general_assistant"],
                            
                            # Vision related
                            "image": ["vision_expert"],
                            "picture": ["vision_expert"],
                            "photo": ["vision_expert"],
                            
                            # Multilingual
                            "translate": ["multilingual"],
                            "language": ["multilingual"]
                        }
                        
                        # Check for keywords and add corresponding labels only if they exist in available_labels
                        for keyword, possible_labels in keyword_mappings.items():
                            if keyword in query_lower:
                                for label in possible_labels:
                                    if label in available_labels_list and label not in fallback_labels:
                                        fallback_labels.append(label)
                                        logger.info(f"Added '{label}' to fallback labels due to keyword '{keyword}'")
                        
                        # Always include general_assistant as a fallback if available
                        if "general_assistant" in available_labels_list and "general_assistant" not in fallback_labels:
                            fallback_labels.append("general_assistant")
                            logger.info("Added 'general_assistant' as default fallback label")
                        
                        # If still no labels matched, pick the first available label
                        if not fallback_labels and available_labels_list:
                            fallback_labels.append(available_labels_list[0])
                            logger.warning(f"No matching labels found, using first available label: {available_labels_list[0]}")
                        
                        logger.info(f"Final fallback labels (valid_labels empty case): {fallback_labels}")
                        return fallback_labels[:3]  # Limit to 3 fallback labels
                    
                    if len(valid_labels) != len(labels):
                        invalid_labels = [label for label in labels if label not in self.available_labels]
                        logger.warning(f"Some labels were invalid and filtered out: {invalid_labels}")
                    
                    logger.info(f"Extracted valid labels: {valid_labels}")
                    return valid_labels
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON from model response: {str(e)}")
                    logger.warning("Using fallback mechanism for label selection")
                    return self._fallback_label_selection(query)
            else:
                logger.warning("Failed to extract JSON array from response, using fallback")
                # Fallback if JSON parsing fails - using ONLY labels from available_labels
                logger.warning("JSON parsing failed, using keyword-based fallback mechanism with ONLY available labels")
                query_lower = query.lower()
                fallback_labels = []
                
                # Create a list of available labels for easier checking
                available_labels_list = list(self.available_labels)
                logger.info(f"Available labels for fallback: {available_labels_list}")
                
                # Map common keywords to specific available labels
                keyword_mappings = {
                    # Math related
                    "math": ["math_expert", "reasoning_expert"],
                    "calculate": ["math_expert", "reasoning_expert"],
                    "equation": ["math_expert", "reasoning_expert"],
                    "solve": ["math_expert", "reasoning_expert", "code_expert"],
                    
                    # Logic/reasoning related 
                    "logic": ["reasoning_expert", "math_expert"],  # Mantık için math ekledik
                    "puzzle": ["reasoning_expert", "math_expert"],  # Puzzles için math ekledik
                    "problem": ["reasoning_expert", "math_expert"],
                    "analyze": ["reasoning_expert", "math_expert"],  # Analiz için math ekledik
                    "deduction": ["reasoning_expert", "math_expert"],  # Dedüktif düşünme için math ekledik
                    "inference": ["reasoning_expert", "math_expert"],  # Çıkarım için math ekledik
                    "reasoning": ["reasoning_expert", "math_expert"],  # Akıl yürütme için math ekledik
                    "deduce": ["reasoning_expert", "math_expert"],
                    "infer": ["reasoning_expert", "math_expert"],
                    "order": ["reasoning_expert", "math_expert"],
                    "arrangement": ["reasoning_expert", "math_expert"],
                    "sequence": ["reasoning_expert", "math_expert"],
                    "pattern": ["reasoning_expert", "math_expert"],
                    "if": ["reasoning_expert", "math_expert"],
                    "then": ["reasoning_expert", "math_expert"],
                    "wear": ["reasoning_expert", "math_expert"],  # Kıyafet ilgili puzzles için math ekledik
                    "color": ["reasoning_expert", "math_expert"],  # Renk ilgili puzzles için math ekledik 
                    "dress": ["reasoning_expert", "math_expert"],  # Elbise ilgili puzzles için math ekledik
                    
                    # Code related
                    "code": ["code_expert"],
                    "program": ["code_expert"],
                    "function": ["code_expert"],
                    "algorithm": ["code_expert"],
                    "programming": ["code_expert"],
                    "developer": ["code_expert"],
                    
                    # Writing related
                    "writing": ["creative_writer"],
                    "write": ["creative_writer"],
                    "essay": ["creative_writer"],
                    "story": ["creative_writer"],
                    "summarize": ["creative_writer", "general_assistant"],
                    "article": ["creative_writer", "general_assistant"],
                    
                    # Vision related
                    "image": ["vision_expert"],
                    "picture": ["vision_expert"],
                    "photo": ["vision_expert"],
                    
                    # Multilingual
                    "translate": ["multilingual"],
                    "language": ["multilingual"]
                }
                
                # Check for keywords and add corresponding labels only if they exist in available_labels
                for keyword, possible_labels in keyword_mappings.items():
                    if keyword in query_lower:
                        for label in possible_labels:
                            if label in available_labels_list and label not in fallback_labels:
                                fallback_labels.append(label)
                                logger.info(f"Added '{label}' to fallback labels due to keyword '{keyword}'")
                
                # Always include general_assistant as a fallback if available
                if "general_assistant" in available_labels_list and "general_assistant" not in fallback_labels:
                    fallback_labels.append("general_assistant")
                    logger.info("Added 'general_assistant' as default fallback label")
                
                # If still no labels matched, pick the first available label
                if not fallback_labels and available_labels_list:
                    fallback_labels.append(available_labels_list[0])
                    logger.warning(f"No matching labels found, using first available label: {available_labels_list[0]}")
                
                logger.info(f"Final fallback labels: {fallback_labels}")
                return fallback_labels[:3]  # Limit to 3 fallback labels
                
        except Exception as e:
            # If analysis fails, return a default set of general labels
            logger.error(f"Error analyzing query: {str(e)}")
            return ["general_assistant"] if "general_assistant" in self.available_labels else []
    
    def determine_complexity(self, query, labels):
        """
        Determine the complexity of the query to decide the number of agents
        
        Args:
            query (str): User query
            labels (list): Identified query labels
            
        Returns:
            str: Complexity level (low, medium, high)
        """
        # Create a prompt to determine complexity
        complexity_prompt = f"""
        You are tasked with determining the complexity of a query.
        
        Query: "{query}"
        Identified labels: {labels}
        
        Rate the complexity of this query as one of: "low", "medium", or "high".
        Consider factors like:
        - Technical depth required
        - Number of distinct concepts involved
        - Whether it requires specialized knowledge
        - Whether it requires step-by-step reasoning
        
        Return only the complexity level as a single word.
        """
        
        try:
            # Get complexity assessment from the model
            response = self.api_handler.generate_text(
                self.coordinator_model, 
                complexity_prompt,
                max_tokens=100
            )
            
            # Extract complexity from the response
            content = response["response"].strip().lower()
            
            if "high" in content:
                return "high"
            elif "medium" in content:
                return "medium"
            else:
                return "low"
                
        except Exception as e:
            # Default to medium complexity if assessment fails
            logger.error(f"Error determining complexity: {str(e)}")
            return "medium"
    
    def evaluate_responses(self, agent_responses, agents):
        """
        Evaluate the responses from multiple agents and return the final result
        
        Args:
            agent_responses (list): List of agent responses
            agents (list): List of agents that generated the responses
        
        Returns:
            tuple: (final_response, agent_details)
        """
        logger.info(f"Evaluating responses from {len(agent_responses)} agents")
        
        # Handle the case of no valid responses
        if not agent_responses:
            logger.error("No valid responses to evaluate")
            raise ValueError("No valid responses to evaluate")
        
        # Validation: ensure agent_responses is a list
        if not isinstance(agent_responses, list):
            logger.error(f"Expected agent_responses to be a list, got {type(agent_responses)}")
            if isinstance(agent_responses, dict):
                # Convert dict to list if needed
                logger.warning("Converting dictionary response to list format")
                agent_responses = [agent_responses]
            else:
                # Try to convert to list or raise error
                try:
                    agent_responses = list(agent_responses)
                except:
                    raise ValueError(f"Cannot convert agent_responses of type {type(agent_responses)} to list")
        
        # If only one agent response, return it directly
        if len(agent_responses) == 1:
            agent_data = agent_responses[0]
            
            # Ensure agent_data is a dictionary
            if not isinstance(agent_data, dict):
                logger.error(f"Expected agent_data to be a dict, got {type(agent_data)}")
                if isinstance(agent_data, str):
                    # Simple string response, construct a proper dict
                    agent = agents[0] if agents and len(agents) > 0 else "Unknown Agent"
                    agent_data = {"agent": agent, "response": agent_data, "labels": []}
                else:
                    raise ValueError(f"Cannot process agent_data of type {type(agent_data)}")
            
            agent = agent_data.get("agent", "Unknown Agent")
            response = agent_data.get("response", "")
            labels = agent_data.get("labels", [])
            
            # Validate the response content
            if not response or not isinstance(response, str) or not response.strip():
                logger.error(f"Invalid or empty response from agent {agent}")
                raise ValueError(f"Invalid or empty response from agent {agent}")
            
            logger.info(f"Only one response available from {agent}, using it directly")
            
            # Prepare the agent details
            agent_details = [
                {
                    "agent": agent,
                    "labels": labels,
                    "rating": "N/A (Single Agent)"
                }
            ]
            
            return response, agent_details
        
        # For multiple responses, prepare for evaluation
        try:
            # Extract responses and prepare for rating
            responses_for_rating = []
            response_by_agent = {}
            
            for agent_data in agent_responses:
                # Skip if agent_data is not a dictionary
                if not isinstance(agent_data, dict):
                    logger.warning(f"Skipping non-dictionary agent data: {type(agent_data)}")
                    continue
                
                agent = agent_data.get("agent", "Unknown Agent")
                response = agent_data.get("response", "")
                
                # Skip empty or invalid responses
                if not response or not isinstance(response, str) or not response.strip():
                    logger.warning(f"Skipping empty or invalid response from {agent}")
                    continue
                
                # Ensure agent is hashable - convert to string if it's not
                if not isinstance(agent, (str, int, float, bool, tuple)):
                    logger.warning(f"Agent is not hashable type: {type(agent)}. Converting to string.")
                    if isinstance(agent, dict):
                        # For dict agents, use the id or name as key if available
                        if "id" in agent:
                            agent = agent["id"]
                        elif "name" in agent:
                            agent = agent["name"]
                        else:
                            # Create a stable representation of the dictionary
                            agent = str(sorted(agent.items()) if hasattr(agent, "items") else agent)
                    else:
                        agent = str(agent)
                
                responses_for_rating.append({
                    "agent": agent,
                    "response": response
                })
                response_by_agent[agent] = response
            
            # If we've filtered out all responses, raise an error
            if not responses_for_rating:
                logger.error("All responses were invalid or empty after validation")
                raise ValueError("All responses were invalid or empty after validation")
            
            # If we only have one valid response after filtering, use it directly
            if len(responses_for_rating) == 1:
                agent = responses_for_rating[0]["agent"]
                response = responses_for_rating[0]["response"]
                logger.info(f"Only one valid response after filtering from {agent}, using it directly")
                
                agent_details = [
                    {
                        "agent": agent,
                        "labels": next((a.get("labels", []) for a in agent_responses if a.get("agent") == agent), []),
                        "rating": "N/A (Single Valid Agent)"
                    }
                ]
                
                return response, agent_details
            
            # For multiple valid responses, perform rating and selection
            agent_ratings = self.rate_agent_responses(responses_for_rating)
            
            # Sort by rating
            sorted_ratings = sorted(agent_ratings, key=lambda x: x["rating"], reverse=True)
            
            # Check if any agents reported errors
            agents_with_errors = [a for a in agent_responses if a.get("error_status")]
            if agents_with_errors:
                error_messages = [f"{a.get('agent')}: {a.get('error_status')}" for a in agents_with_errors]
                logger.warning(f"Some agents reported errors: {', '.join(error_messages)}")
                
                # If all agents had errors, we'll still use the best one but add a warning
                if len(agents_with_errors) == len(agent_responses):
                    logger.warning("All agents reported errors, using highest rated despite errors")
            
            # Use the highest-rated response
            best_agent = sorted_ratings[0]["agent"]
            
            # Ensure the best agent exists in response_by_agent
            if best_agent not in response_by_agent:
                logger.error(f"Agent {best_agent} not found in response_by_agent dictionary")
                # Fallback to the first agent if best agent is not found
                best_agent = list(response_by_agent.keys())[0]
                logger.info(f"Falling back to agent {best_agent}")
            
            best_response = response_by_agent[best_agent]
            
            # Check if the best agent response contains an error status
            best_agent_error = next((a.get("error_status") for a in agent_responses if a.get("agent") == best_agent), None)
            if best_agent_error:
                logger.warning(f"Best agent {best_agent} had error: {best_agent_error}")
                # Add an error note to the response
                best_response = f"**Note:** The system encountered an issue with this model: {best_agent_error}\n\n{best_response}"
            
            logger.info(f"Selected best response from agent {best_agent}")
            
            # Prepare agent details for display
            agent_details = []
            for rating_data in sorted_ratings:
                agent = rating_data["agent"]
                agent_response = next((a for a in agent_responses if a.get("agent") == agent), {})
                agent_details.append({
                    "agent": agent,
                    "labels": agent_response.get("labels", []),
                    "rating": f"{rating_data['rating']:.2f}",
                    "error_status": agent_response.get("error_status"),
                    "model_info": agent_response.get("model_info", {})
                })
            
            return best_response, agent_details
            
        except Exception as e:
            logger.error(f"Error during response evaluation: {str(e)}", exc_info=True)
            
            # Fallback: Use the first non-empty response if evaluation fails
            for agent_data in agent_responses:
                # Handle non-dictionary response
                if not isinstance(agent_data, dict):
                    if isinstance(agent_data, str) and agent_data.strip():
                        logger.info("Using raw string response as fallback")
                        agent = "Unknown Agent"
                        agent_details = [{
                            "agent": agent,
                            "labels": [],
                            "rating": "N/A (Fallback - Raw String)"
                        }]
                        return agent_data, agent_details
                    continue
                
                response = agent_data.get("response", "")
                agent = agent_data.get("agent", "Unknown Agent")
                
                if response and isinstance(response, str) and response.strip():
                    logger.info(f"Using response from {agent} as fallback due to evaluation error")
                    
                    agent_details = [{
                        "agent": agent,
                        "labels": agent_data.get("labels", []),
                        "rating": "N/A (Fallback)"
                    }]
                    
                    return response, agent_details
            
            # If all fallbacks fail, attempt one last desperate approach
            try:
                # Try to concatenate all responses as a last resort
                logger.info("Attempting to concatenate all responses as final fallback")
                all_responses = []
                
                for agent_data in agent_responses:
                    if isinstance(agent_data, dict) and "response" in agent_data:
                        resp = agent_data["response"]
                        if isinstance(resp, str) and resp.strip():
                            all_responses.append(resp)
                    elif isinstance(agent_data, str) and agent_data.strip():
                        all_responses.append(agent_data)
                
                if all_responses:
                    combined_response = "\n\n---\n\n".join(all_responses)
                    agent_details = [{
                        "agent": "Combined Responses",
                        "labels": [],
                        "rating": "N/A (Emergency Fallback)"
                    }]
                    return combined_response, agent_details
            except Exception as concat_error:
                logger.error(f"Failed to concatenate responses: {str(concat_error)}")
                
            # If no valid fallback found, raise the original error
            raise ValueError(f"Failed to evaluate responses: {str(e)}")
    
    def rate_agent_responses(self, responses):
        """
        Rate the responses from multiple agents based on quality criteria
        
        Args:
            responses (list): List of agent responses
        
        Returns:
            list: List of agent ratings
        """
        logger.info("Rating responses from multiple agents")
        
        # First check if we actually have responses to rate
        if not responses:
            logger.warning("No responses to rate")
            return []
            
        # Create prompt for the coordinator to rate the responses
        rating_prompt = f"""
        You are evaluating responses from multiple AI assistants to choose the best one.

        {'-' * 50}
        QUERY:
        {'-' * 50}

        Now review these responses:

        """
        
        # Add each response to the prompt with a separator
        for i, response_data in enumerate(responses):
            agent = response_data["agent"]
            response = response_data["response"]
            rating_prompt += f"{'-' * 50}\nRESPONSE {i+1} from {agent}:\n{'-' * 50}\n{response[:1000]}\n\n"
        
        # Add scoring instructions
        rating_prompt += f"""
        {'-' * 50}
        SCORING INSTRUCTIONS:
        {'-' * 50}
        For each response, assign a score from 0.0 to 10.0 based on these criteria:
        
        1. Correctness (4 points) - factual accuracy and logical coherence
        2. Completeness (3 points) - addresses all aspects of the query
        3. Clarity (2 points) - clear, concise, well-structured
        4. Helpfulness (1 point) - practical, actionable, useful
        
        Return ONLY a JSON object with scores, with no explanation:
        {{
            "response_1": 8.5,
            "response_2": 7.2,
            ...
        }}
        """
        
        try:
            # Get ratings from the coordinator model
            rating_response = self.api_handler.generate_text(
                self.coordinator_model,
                rating_prompt,
                max_tokens=500
            )
            
            content = rating_response["response"]
            logger.info(f"Rating response received: {content}")
            
            # Parse the ratings from the response
            import re
            import json
            
            # Find a JSON object in the response
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                try:
                    ratings_dict = json.loads(json_match.group(0))
                    
                    # Transform the dictionary into the expected format
                    ratings = []
                    for i, response_data in enumerate(responses):
                        response_key = f"response_{i+1}"
                        rating_value = ratings_dict.get(response_key, 5.0)  # Default to 5.0 if not found
                        ratings.append({
                            "agent": response_data["agent"],
                            "rating": rating_value
                        })
                    
                    logger.info(f"Parsed ratings: {ratings}")
                    return ratings
                    
                except json.JSONDecodeError:
                    logger.error("Failed to parse ratings as JSON")
                    pass
        except Exception as e:
            logger.error(f"Error getting ratings: {str(e)}")
        
        # Fallback rating logic if the rating process fails
        logger.warning("Using fallback rating logic")
        
        # Simple heuristic rating based on response length and keywords
        ratings = []
        for response_data in responses:
            agent = response_data["agent"]
            response = response_data["response"]
            
            # Start with a base rating
            base_rating = 5.0
            
            # Adjust for length - prefer medium-length responses (neither too short nor too long)
            length = len(response)
            if length < 50:
                length_score = 0.7  # Too short
            elif 50 <= length < 200:
                length_score = 0.85  # A bit short
            elif 200 <= length < 1000:
                length_score = 1.0  # Good length
            elif 1000 <= length < 3000:
                length_score = 0.95  # Somewhat long
            else:
                length_score = 0.85  # Too long
            
            # Check for positive indicators (code blocks, numbered lists, etc.)
            positive_indicators = [
                "```", "1.", "2.", "3.", "example", "for instance", 
                "in summary", "to conclude", "because", "therefore"
            ]
            
            positive_score = sum(0.2 for ind in positive_indicators if ind in response) / len(positive_indicators)
            positive_score = min(positive_score, 1.5)  # Cap at 1.5
            
            # Check for negative indicators (uncertainty, etc.)
            negative_indicators = [
                "I'm not sure", "I don't know", "I'm uncertain", "might be", 
                "could be", "possibly", "maybe", "I think"
            ]
            
            negative_score = sum(0.15 for ind in negative_indicators if ind in response) / len(negative_indicators)
            negative_score = min(negative_score, 1.0)  # Cap at 1.0
            
            # Calculate final rating
            final_rating = base_rating * length_score * (1 + positive_score) * (1 - negative_score)
            final_rating = max(1.0, min(final_rating, 10.0))  # Ensure rating is between 1 and 10
            
            ratings.append({
                "agent": agent,
                "rating": final_rating
            })
        
        return ratings
    
    def _get_model_labels(self, model_id):
        """
        Get labels for a specific model ID
        
        Args:
            model_id (str): Model identifier
            
        Returns:
            list: List of labels for the model
        """
        for model_info in self.model_labels_data:
            if model_info.get("model") == model_id:
                return model_info.get("labels", [])
        
        return []
