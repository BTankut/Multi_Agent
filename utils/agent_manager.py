import os
import json
import logging
import concurrent.futures
import requests
from dotenv import load_dotenv
from utils.api_handler import OpenRouterAPIHandler

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('AgentManager')

class AgentManager:
    """
    Manages the selection and coordination of multiple AI agents
    based on query tags and user preferences.
    """
    
    def __init__(self, models_data, model_labels_data, model_roles_data=None):
        """
        Initialize the agent manager
        
        Args:
            models_data (list): List of available models with their details
            model_labels_data (list): List containing model-label mappings
            model_roles_data (dict, optional): Dictionary containing role descriptions for each label
        """
        load_dotenv()
        self.api_handler = OpenRouterAPIHandler()
        self.models_data = models_data
        self.model_labels_data = model_labels_data
        self.model_roles_data = model_roles_data
        
        # Print debugging information
        logger.info(f"models_data type: {type(models_data)}, length: {len(models_data) if models_data else 0}")
        logger.info(f"model_labels_data type: {type(model_labels_data)}, length: {len(model_labels_data) if model_labels_data else 0}")
        if model_roles_data:
            logger.info(f"model_roles_data type: {type(model_roles_data)}, keys: {model_roles_data.keys() if hasattr(model_roles_data, 'keys') else 'No keys'}")
        
        # Önce model_labels'daki "free" etiketine sahip modelleri belirleyelim
        self.labeled_free_models = set()
        for model_info in model_labels_data:
            if isinstance(model_info, dict) and "model" in model_info and "labels" in model_info:
                if "free" in model_info["labels"]:
                    self.labeled_free_models.add(model_info["model"].lower())
                    logger.info(f"Found model with 'free' label in model_labels.json: {model_info['model']}")
        
        logger.info(f"Found {len(self.labeled_free_models)} models with 'free' label in model_labels.json")
        
        # Separate models into free and paid lists and verify pricing information
        self.free_models = []
        self.paid_models = []
        
        # Log model details for debugging and validate pricing information
        logger.info("Classifying models as free or paid:")
        for model in models_data:
            model_id = model.get("id", "unknown")
            model_name = model.get("name", "unknown")
            
            # Ensure pricing information exists
            if "pricing" not in model:
                logger.warning(f"Model {model_id} ({model_name}) has no pricing info. Adding default.")
                model["pricing"] = {"is_free": False, "prompt": 0.0, "completion": 0.0}
            
            # Ücretsiz model tespiti için üç yöntem uygulayacağız:
            # 1. Model ID'si veya isminde ":free" varsa (OpenRouter API'deki model adlandırması)
            # 2. model_labels.json dosyasında "free" etiketi varsa
            # 3. Fiyatlandırma bilgisine bakılır
            
            # 1. Model ID'si veya isminde "free" kontrolü
            model_lower_name = model_name.lower()
            model_lower_id = model_id.lower()
            
            is_name_free = False
            if ":free" in model_lower_id or model_lower_name.endswith("(free)"):
                is_name_free = True
                logger.info(f"Model {model_id} ({model_name}) identified as free from model name")
            
            # 2. model_labels.json'daki "free" etiketli modellere eşleşme kontrolü
            is_labeled_free = False
            for free_model_name in self.labeled_free_models:
                # Eşleşme için farklı kombinasyonları dene
                if (free_model_name in model_lower_name or 
                    model_lower_name in free_model_name or 
                    free_model_name in model_lower_id or
                    model_lower_id in free_model_name):
                    is_labeled_free = True
                    logger.info(f"Model {model_id} ({model_name}) matched with free label in model_labels.json")
                    break
            
            # 3. Fiyatlandırma bilgisine göre kontrol (en düşük öncelikli)
            is_pricing_free = False
            if "is_free" in model["pricing"]:
                is_pricing_free = model["pricing"]["is_free"]
            else:
                # Determine if model is free based on pricing values
                prompt_cost = model["pricing"].get("prompt", 0)
                completion_cost = model["pricing"].get("completion", 0)
                is_pricing_free = (prompt_cost == 0 and completion_cost == 0)
                model["pricing"]["is_free"] = is_pricing_free
                logger.info(f"Added is_free={is_pricing_free} flag to model {model_id} based on pricing")
            
            # Eğer herhangi bir şekilde ücretsiz olarak belirlendiyse, free_models listesine ekle
            # Öncelik: isim > model_labels > pricing
            if is_name_free or is_labeled_free or is_pricing_free:
                # Modeli kesinlikle ücretsiz olarak işaretle
                model["pricing"]["is_free"] = True
                self.free_models.append(model)
                
                # Hangi yöntemle tespit edildiğini logla
                if is_name_free:
                    logger.info(f"Classified as FREE (from model name): {model_id} ({model_name})")
                elif is_labeled_free:
                    logger.info(f"Classified as FREE (from model_labels.json): {model_id} ({model_name})")
                else:
                    logger.info(f"Classified as FREE (from pricing): {model_id} ({model_name})")
            else:
                self.paid_models.append(model)
                logger.info(f"Classified as PAID: {model_id} ({model_name})")
        
        logger.info(f"Classification complete. Free models: {len(self.free_models)}, Paid models: {len(self.paid_models)}")
        
        # Create a mapping from model IDs to their labels
        self.model_to_labels = {}
        
        # Add a default label for all models
        for model in models_data:
            self.model_to_labels[model["id"]] = ["general_assistant"]
            logger.info(f"Set default label 'general_assistant' for model {model['id']}")
        
        # Create indexes for quick model lookup
        self.model_index = {model["id"]: model for model in models_data}
        
        # Add labels from model_labels_data
        if isinstance(model_labels_data, list):
            logger.info(f"Processing {len(model_labels_data)} model label entries")
            for model_info in self.model_labels_data:
                if isinstance(model_info, dict) and "model" in model_info and "labels" in model_info:
                    model_name = model_info["model"].lower()  # Convert to lowercase for easier matching
                    logger.info(f"Processing model labels for '{model_name}' with labels {model_info['labels']}")
                    
                    # Try to find a matching model from the API response
                    matched = False
                    for model in self.models_data:
                        # Try different ways to match the model name
                        api_model_name = model.get("name", "").lower()
                        api_model_id = model.get("id", "").lower()
                        
                        # Check if our model_name is contained in the API model name or id
                        if (model_name in api_model_name or
                            api_model_name in model_name or
                            model_name in api_model_id or
                            any(part in api_model_name for part in model_name.split())):
                            
                            self.model_to_labels[model["id"]] = model_info["labels"]
                            logger.info(f" Matched model '{model_name}' to API model '{api_model_name}' (id: {model['id']}) with labels {model_info['labels']}")
                            matched = True
                            break
                    
                    if not matched:
                        logger.warning(f" Could not match model '{model_name}' to any API model")
        
        logger.info(f"Created model_to_labels with {len(self.model_to_labels)} entries")
        
        # Log a few example mappings
        example_mappings = list(self.model_to_labels.items())[:5]
        logger.info("Example label mappings:")
        for model_id, labels in example_mappings:
            logger.info(f"Model {model_id} has labels: {labels}")
        
        if len(self.model_to_labels) == 0:
            logger.error("No model-to-labels mappings were created! This will cause 'No suitable agents found'")
    
    def select_agents(self, query_labels, option_selected, num_agents=1):
        """
        Select appropriate agent models based on query labels and user preferences
        
        Args:
            query_labels (list): Labels identified for the query
            option_selected (str): User's preference option
            num_agents (int): Desired number of agents to select (default: 1)
            
        Returns:
            list: List of selected agent models
        """
        logger.info(f"Selecting {num_agents} agents for labels: {query_labels}, option: {option_selected}")
        
        if not query_labels:
            logger.error("No query labels provided, cannot select agents")
            return []
        
        # Map option string to numeric option
        option_map = {
            "Free models only": 1,
            "Paid models only": 2,
            "Optimized mix of free and paid models": 3
        }
        option = option_map.get(option_selected, 3)
        logger.info(f"Option selected: {option_selected} (mapped to {option})")
        
        # Önce seçeneklere göre modellerimizi filtreleyelim - bu iyileştirme
        available_models = []
        if option == 1:  # Free models only
            available_models = self.free_models.copy()
            logger.info(f"Filtering to Free models only - {len(available_models)} models available")
        elif option == 2:  # Paid models only
            available_models = self.paid_models.copy()
            logger.info(f"Filtering to Paid models only - {len(available_models)} models available") 
        else:  # Optimized mix - tüm modeller
            available_models = self.models_data.copy()
            logger.info(f"Using all models for Optimized mix - {len(available_models)} models available")
        
        # Eğer seçilen opsiyon için hiç model yoksa, hata logla ve boş dön
        if not available_models:
            logger.error(f"No models available for option {option_selected}")
            return []
        
        # İlgili modeller içinden etikete göre modelleri bul
        candidate_models = []
        for label in query_labels:
            logger.info(f"Looking for models with label: {label}")
            # Sadece kullanılabilir modeller içinden filtreleme yap
            model_ids = [model['id'] for model in available_models]
            
            for model_id, labels in self.model_to_labels.items():
                if label in labels and model_id in model_ids:
                    model = self.model_index[model_id]
                    candidate_models.append({
                        "model": model,
                        "label": label
                    })
                    logger.info(f"Found matching model {model_id} for label {label}")
            
        logger.info(f"Total candidate models found: {len(candidate_models)}")
        
        # If no models match the labels, use default models
        if not candidate_models:
            logger.warning("No candidate models found, falling back to defaults")
            if option == 1:  # Free models only
                if not self.free_models:
                    logger.error("No free models available for fallback")
                    return []
                default_model = next((m for m in self.free_models if "claude" in m["id"].lower()), self.free_models[0])
                logger.info(f"Using default free model: {default_model.get('id', 'unknown')}")
                return [default_model]
            elif option == 2:  # Paid models only
                if not self.paid_models:
                    logger.error("No paid models available for fallback")
                    return []
                default_model = next((m for m in self.paid_models if "claude" in m["id"].lower()), self.paid_models[0])
                logger.info(f"Using default paid model: {default_model.get('id', 'unknown')}")
                return [default_model]
            else:  # Optimized
                default_model = next((m for m in self.models_data if "claude" in m["id"].lower()), self.models_data[0])
                logger.info(f"Using default optimized model: {default_model.get('id', 'unknown')}")
                return [default_model]
        
        # Filter and select models based on user option
        selected_agents = []
        
        # Candidate modellerden grup oluştur ve etiketlere göre ayır
        label_models = {}
        for candidate in candidate_models:
            label = candidate["label"]
            if label not in label_models:
                label_models[label] = []
            label_models[label].append(candidate)
        
        # Her etiket için en iyi modeli seç
        for label, candidates in label_models.items():
            if not candidates:
                continue
            
            if option == 1:  # Free models only
                # Ücretsiz modeller içinden en iyisini seç
                free_models = [c for c in candidates if c["model"].get("pricing", {}).get("is_free", True)]
                
                if free_models:
                    # Tüm free modelleri puana göre sırala
                    best_free_model = None
                    for model in free_models:
                        if best_free_model is None or self._is_better_model(model["model"], best_free_model["model"]):
                            best_free_model = model
                            
                    if best_free_model:
                        selected_agents.append(best_free_model["model"])
                        logger.info(f"Selected free model {best_free_model['model'].get('id')} for label {label}")
                else:
                    logger.warning(f"No free models found for label {label}")
                
            elif option == 2:  # Paid models only  
                # Ücretli modeller içinden en iyisini seç
                paid_models = [c for c in candidates if not c["model"].get("pricing", {}).get("is_free", False)]
                
                if paid_models:
                    # Tüm paid modelleri puana göre sırala
                    best_paid_model = None
                    for model in paid_models:
                        if best_paid_model is None or self._is_better_model(model["model"], best_paid_model["model"]):
                            best_paid_model = model
                            
                    if best_paid_model:
                        selected_agents.append(best_paid_model["model"])  
                        logger.info(f"Selected paid model {best_paid_model['model'].get('id')} for label {label}")
                else:
                    logger.warning(f"No paid models found for label {label}")
            
            else:  # Optimized mix
                # Modelleri maliyet etkinliğine göre sırala
                scored_candidates = [(c, self._calculate_cost_effectiveness(c["model"])) for c in candidates]
                scored_candidates.sort(key=lambda x: x[1], reverse=True)
                
                # En iyi modeli seç
                if scored_candidates:
                    selected_agents.append(scored_candidates[0][0]["model"])
                    logger.info(f"Selected optimized model {scored_candidates[0][0]['model'].get('id')} for label {label}")
                    
                    # Eğer daha fazla ajana ihtiyaç varsa, bu etiket için birden fazla model seç
                    max_agents_per_label = max(1, num_agents // 2)
                    if len(scored_candidates) > 1 and len(selected_agents) < num_agents:
                        for i in range(1, min(max_agents_per_label, len(scored_candidates))):
                            if len(selected_agents) >= num_agents:
                                break
                            selected_agents.append(scored_candidates[i][0]["model"])
                            logger.info(f"Added additional optimized agent for label {label}")
                else:
                    logger.warning(f"No scored candidates found for label {label}")
        
        # Ensure we have enough agents
        if not selected_agents or len(selected_agents) < num_agents:
            required_count = num_agents - len(selected_agents)
            logger.warning(f"Selection logic selected only {len(selected_agents)} agents, need {required_count} more")
            logger.warning("Attempting to select additional models to meet the requested agent count")
            
            # Group candidates by label
            label_models = {}
            for candidate in candidate_models:
                label = candidate["label"]
                if label not in label_models:
                    label_models[label] = []
                label_models[label].append(candidate)
            
            # For each label, select the first appropriate model based on option
            for label, candidates in label_models.items():
                if not candidates:
                    continue
                
                if option == 1:  # Free models only
                    # Find first free model for this label
                    for candidate in candidates:
                        if candidate["model"].get("pricing", {}).get("is_free", False):
                            selected_agents.append(candidate["model"])
                            logger.info(f"Fallback: Selected free model {candidate['model'].get('id')} for label {label}")
                            break
                
                elif option == 2:  # Paid models only
                    # Find first paid model for this label
                    for candidate in candidates:
                        if not candidate["model"].get("pricing", {}).get("is_free", False):
                            selected_agents.append(candidate["model"])
                            logger.info(f"Fallback: Selected paid model {candidate['model'].get('id')} for label {label}")
                            break
                
                else:  # Optimized mix
                    # Just take the first model for this label
                    selected_agents.append(candidates[0]["model"])
                    logger.info(f"Fallback: Selected model {candidates[0]['model'].get('id')} for label {label}")
            
            # If still no agents selected, try to find a model that matches the selected option
            if not selected_agents and candidate_models:
                logger.warning("Extreme fallback: Trying to select appropriate model based on option")
                
                if option == 1:  # Free models only
                    # Find the first free model from candidates
                    free_candidates = [c for c in candidate_models if c["model"].get("pricing", {}).get("is_free", False)]
                    if free_candidates:
                        selected_agents.append(free_candidates[0]["model"])
                        logger.info(f"Extreme fallback: Selected free model {free_candidates[0]['model'].get('id')}")
                    else:
                        logger.error("Could not find any free models among candidates despite 'Free models only' option")
                        # Check if there are any free models available at all
                        if self.free_models:
                            selected_agents.append(self.free_models[0])
                            logger.info(f"Emergency fallback: Selected free model {self.free_models[0].get('id')}")
                        else:
                            logger.error("No free models available")
                
                elif option == 2:  # Paid models only
                    # Find the first paid model from candidates
                    paid_candidates = [c for c in candidate_models if not c["model"].get("pricing", {}).get("is_free", False)]
                    if paid_candidates:
                        selected_agents.append(paid_candidates[0]["model"])
                        logger.info(f"Extreme fallback: Selected paid model {paid_candidates[0]['model'].get('id')}")
                    else:
                        logger.error("Could not find any paid models among candidates despite 'Paid models only' option")
                        # Check if there are any paid models available at all
                        if self.paid_models:
                            selected_agents.append(self.paid_models[0])
                            logger.info(f"Emergency fallback: Selected paid model {self.paid_models[0].get('id')}")
                        else:
                            logger.error("No paid models available")
                
                else:  # Optimized mix
                    # For optimized, we can just take the first candidate
                    selected_agents.append(candidate_models[0]["model"])
                    logger.info(f"Extreme fallback: Selected model {candidate_models[0]['model'].get('id')} for optimized mix")
        
        # Final check: we need to ensure we have the minimum requested number of agents
        if len(selected_agents) < num_agents:
            logger.warning(f"Could not find enough agents to meet the requested count of {num_agents}")
            
            # Try to fill in with agents we haven't used yet
            if option == 1:  # Free models only
                # Sadece ücretsiz modeller içinden seç
                remaining_agents = []
                for model in self.free_models:
                    if model not in selected_agents:
                        is_free = model.get("pricing", {}).get("is_free", False)
                        if is_free:
                            remaining_agents.append(model)
                
                # Uygun etiketlere göre modelleri sırala
                sorted_agents = []
                for label in query_labels:
                    for model in remaining_agents[:]:  # Kopyasını al ki içinde değişiklik yapabilelim
                        model_labels = self.get_model_labels(model)
                        if label in model_labels:
                            sorted_agents.append(model)
                            if model in remaining_agents:  # Çift kontrol
                                remaining_agents.remove(model)
                
                # Önce etiket eşleşenlerden ekle
                for model in sorted_agents:
                    if len(selected_agents) >= num_agents:
                        break
                    selected_agents.append(model)
                    logger.info(f"Added supplemental free agent (matching label) to meet count: {model.get('id', 'unknown')}")
                
                # Hala yeteri kadar agent yoksa, kalan free modelleri ekle
                while len(selected_agents) < num_agents and remaining_agents:
                    selected_agents.append(remaining_agents.pop(0))
                    logger.info(f"Added supplemental free agent to meet count: {selected_agents[-1].get('id', 'unknown')}")
            
            elif option == 2:  # Paid models only
                # Sadece ücretli modelleri kullan
                remaining_agents = []
                for model in self.paid_models:
                    if model not in selected_agents:
                        is_free = model.get("pricing", {}).get("is_free", False)
                        if not is_free:  # Ücretli modeller için kontrol
                            remaining_agents.append(model)
                
                # Uygun etiketlere göre modelleri sırala
                sorted_agents = []
                for label in query_labels:
                    for model in remaining_agents[:]:  # Kopyasını al ki içinde değişiklik yapabilelim
                        model_labels = self.get_model_labels(model)
                        if label in model_labels:
                            sorted_agents.append(model)
                            if model in remaining_agents:  # Çift kontrol
                                remaining_agents.remove(model)
                
                # Önce etiket eşleşenlerden ekle
                for model in sorted_agents:
                    if len(selected_agents) >= num_agents:
                        break
                    selected_agents.append(model)
                    logger.info(f"Added supplemental paid agent (matching label) to meet count: {model.get('id', 'unknown')}")
                
                # Hala yeteri kadar agent yoksa, kalan paid modelleri ekle
                while len(selected_agents) < num_agents and remaining_agents:
                    selected_agents.append(remaining_agents.pop(0))
                    logger.info(f"Added supplemental paid agent to meet count: {selected_agents[-1].get('id', 'unknown')}")
            
            else:  # Optimized mix - use any model
                # İlk olarak etiketlere uyumlu modelleri bul
                remaining_agents = [m for m in self.models_data if m not in selected_agents]
                
                # Uygun etiketlere göre modelleri sırala
                sorted_agents = []
                for label in query_labels:
                    for model in remaining_agents[:]:  # Kopyasını al ki içinde değişiklik yapabilelim
                        model_labels = self.get_model_labels(model)
                        if label in model_labels:
                            sorted_agents.append(model)
                            if model in remaining_agents:  # Çift kontrol
                                remaining_agents.remove(model)
                
                # Önce etiket eşleşenlerden ekle
                for model in sorted_agents:
                    if len(selected_agents) >= num_agents:
                        break
                    selected_agents.append(model)
                    logger.info(f"Added supplemental agent (matching label) to meet count: {model.get('id', 'unknown')}")
                
                # Hala yeteri kadar agent yoksa, kalan modelleri ekle
                while len(selected_agents) < num_agents and remaining_agents:
                    selected_agents.append(remaining_agents.pop(0))
                    logger.info(f"Added supplemental agent to meet count: {selected_agents[-1].get('id', 'unknown')}")
                    
            # Eğer hala yeterli ajan yoksa, uyarı logu
            if len(selected_agents) < num_agents:
                logger.warning(f"Could not find {num_agents} agents, only found {len(selected_agents)} agents")
        
        # EMERGENCY FALLBACK: If we still have no agents, add ANY agents regardless of free/paid preference
        if not selected_agents:
            logger.error("CRITICAL: No agents selected after all attempts. Using emergency fallback.")
            
            # Tüm modelleri sortelemek için genel listeler oluştur
            if option == 1:  # Free models only
                # Free models only seçeneği için, kesinlikle ücretsiz modelleri seç
                available_models = [m for m in self.free_models if m.get("pricing", {}).get("is_free", False)]
                logger.info(f"Emergency: Using only free models ({len(available_models)} models)")
            elif option == 2:  # Paid models only
                # Paid models only seçeneği için, kesinlikle ücretli modelleri seç
                available_models = [m for m in self.paid_models if not m.get("pricing", {}).get("is_free", False)]
                logger.info(f"Emergency: Using only paid models ({len(available_models)} models)")
            else:  # Optimized mix
                # Tüm modelleri kullan, ancak önce en iyi modelleri seç
                available_models = self.models_data
                logger.info(f"Emergency: Using all models ({len(available_models)} models)")
            
            # First try: use "general_assistant" labeled models from available_models
            for model in available_models:
                model_labels = self.get_model_labels(model)
                if "general_assistant" in model_labels:
                    selected_agents.append(model)
                    logger.info(f"Emergency fallback: Added general assistant model: {model.get('id', 'unknown')}")
                    if len(selected_agents) >= num_agents:
                        break
            
            # Second try: use any models from available_models if still no agents
            if not selected_agents and available_models:
                logger.error("CRITICAL: Still no agents after general_assistant fallback. Using ANY available models.")
                
                # Sort available_models by some priority (e.g., context length)
                sorted_models = sorted(
                    available_models, 
                    key=lambda m: m.get("context_length", 0), 
                    reverse=True
                )
                
                # Add models until we reach the desired count
                for i in range(min(num_agents, len(sorted_models))):
                    selected_agents.append(sorted_models[i])
                    logger.info(f"Last resort fallback: Added model: {sorted_models[i].get('id', 'unknown')}")
                
            # If still no models available after all attempts
            if not selected_agents:
                logger.error("CRITICAL: No models available for fallback. Cannot proceed.")
        
        # Ensure we don't return more than the requested number of agents
        if len(selected_agents) > num_agents:
            logger.info(f"Selected {len(selected_agents)} agents, limiting to requested {num_agents}")
            selected_agents = selected_agents[:num_agents]
        
        logger.info(f"Final selected agents ({len(selected_agents)}): {[agent.get('id', 'unknown') for agent in selected_agents]}")
        return selected_agents
    
    def process_query(self, agents, query):
        """
        Process the user query by sending it to the selected agents in parallel
        and returning their responses.
        
        Args:
            agents (list): List of agents to process the query
            query (str): The user query to be processed
            
        Returns:
            list: List of agent responses with their metadata
        """
        logger.info(f"Processing query with {len(agents)} agents in parallel")
        
        if not agents:
            logger.error("No agents provided to process query")
            return []
            
        # Log the specific agents being used
        for i, agent in enumerate(agents):
            if isinstance(agent, dict):
                agent_id = agent.get("id", "unknown")
                agent_name = agent.get("name", "unknown")
                agent_context = agent.get("context_length", "unknown")
                agent_pricing = "Free" if agent.get("pricing", {}).get("is_free", False) else "Paid"
                logger.info(f"Agent {i+1}: ID={agent_id}, Name={agent_name}, Context={agent_context}, Pricing={agent_pricing}")
            else:
                logger.info(f"Agent {i+1}: {agent}")
        
        # Create a sequential backup in case parallel execution fails
        sequential_backup_enabled = True  # Enable/disable sequential backup
        
        # Try the normal parallel execution first
        try:
            futures = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(agents)) as executor:
                for agent in agents:
                    logger.info(f"Preparing query for agent: {agent} with labels: {self.get_model_labels(agent)}")
                    
                    # Define the agent's system prompt
                    system_prompt = self.create_system_prompt(agent)
                    
                    # Submit the task to the executor
                    future = executor.submit(
                        self.query_single_agent,
                        agent=agent,
                        query=query,
                        system_prompt=system_prompt
                    )
                    futures.append(future)
            
            # Collect valid responses
            responses = []
            valid_response_count = 0
            error_messages = []
            
            for i, future in enumerate(futures):
                agent = agents[i]
                try:
                    response = future.result()
                    logger.info(f"Got future result for agent {i+1}")
                    
                    # Handle empty responses
                    if response is None:
                        logger.warning(f"Agent {agent} returned None response")
                        continue
                    
                    # Log the raw response for debugging
                    if isinstance(response, dict):
                        logger.info(f"Raw response from agent {i+1} is a dictionary with keys: {list(response.keys())}")
                    else:
                        logger.info(f"Raw response from agent {i+1} is type: {type(response)}")
                        
                    # Handle the case where response might be a dictionary with response inside
                    if isinstance(response, dict):
                        if "response" in response:
                            logger.info(f"Response is a dictionary with 'response' key, extracting content")
                            response_content = response["response"]
                            
                            # Verify the extracted content
                            if isinstance(response_content, str) and response_content.strip():
                                response = response_content
                            else:
                                logger.warning(f"Extracted 'response' content is invalid: {type(response_content)}")
                        else:
                            # Try to find any string field with substantial content
                            found_content = False
                            for key, value in response.items():
                                if isinstance(value, str) and len(value.strip()) > 10:
                                    logger.info(f"Using dictionary field '{key}' as response content")
                                    response = value
                                    found_content = True
                                    break
                            
                            if not found_content:
                                logger.warning(f"Could not extract usable content from response dictionary")
                                # Log the full dictionary content for debugging
                                try:
                                    logger.debug(f"Full response dictionary: {json.dumps(response, indent=2)}")
                                except:
                                    logger.debug(f"Could not serialize response dictionary")
                                continue
                    
                    # Final check to ensure we have a valid string response
                    if not isinstance(response, str):
                        logger.warning(f"Agent {agent} returned non-string response type: {type(response)}")
                        try:
                            response = str(response)
                            logger.info(f"Converted response to string: {response[:100]}...")
                        except:
                            logger.error(f"Failed to convert response to string")
                            continue
                    
                    if not response.strip():
                        logger.warning(f"Agent {agent} returned empty string response")
                        continue
                    
                    # Ensure agent is a hashable type
                    agent_key = agent
                    if not isinstance(agent_key, (str, int, float, bool, tuple)):
                        logger.warning(f"Agent type {type(agent_key)} is not hashable, converting to string")
                        if isinstance(agent_key, dict):
                            # For dict agents, use the id or name as key if available
                            if "id" in agent_key:
                                agent_key = agent_key["id"]
                            elif "name" in agent_key:
                                agent_key = agent_key["name"]
                            else:
                                # Create a stable representation of the dictionary
                                agent_key = str(sorted(agent_key.items()) if hasattr(agent_key, "items") else agent_key)
                        else:
                            agent_key = str(agent_key)
                    
                    # Log response details
                    logger.info(f"Received valid response from agent {agent_key}")
                    logger.info(f"Response content (first 100 chars): {response[:100]}...")
                    
                    # Check for error indicators in the response
                    error_keywords = ["api rate limit", "context limit", "token limit", "maximum context", 
                                     "quota exceeded", "too many tokens", "server error", "service unavailable",
                                     "failed to process", "exceeded maximum", "request timed out"]
                    
                    has_error = any(keyword in response.lower() for keyword in error_keywords)
                    if has_error:
                        error_type = next((keyword for keyword in error_keywords if keyword in response.lower()), "unknown error")
                        logger.warning(f"Agent {agent_key} response contains error indicator: {error_type}")
                        # We still include the response but mark it as containing an error
                        error_status = f"Possible error detected: {error_type}"
                    else:
                        error_status = None
                    
                    # Normalize labels
                    try:
                        labels = self.get_model_labels(agent)
                        if not isinstance(labels, list):
                            logger.warning(f"Labels for agent {agent_key} is not a list: {type(labels)}")
                            if labels is None:
                                labels = []
                            else:
                                try:
                                    labels = list(labels)  # Try to convert to list
                                except:
                                    labels = [str(labels)]  # Fall back to single-item list
                    except Exception as label_error:
                        logger.error(f"Error getting labels for agent {agent_key}: {str(label_error)}")
                        labels = []
                        
                    responses.append({
                        "agent": agent_key,
                        "response": response,
                        "labels": labels,
                        "error_status": error_status,  # Include error status if any
                        "model_info": {
                            "id": agent.get("id", str(agent)) if isinstance(agent, dict) else str(agent),
                            "name": agent.get("name", "Unknown") if isinstance(agent, dict) else "Unknown",
                            "context_length": agent.get("context_length", "Unknown") if isinstance(agent, dict) else "Unknown",
                            "is_free": agent.get("pricing", {}).get("is_free", False) if isinstance(agent, dict) else False
                        }
                    })
                    valid_response_count += 1
                    
                except Exception as e:
                    error_msg = str(e)
                    logger.error(f"Agent {agent} failed: {error_msg}")
                    logger.exception("Exception details:")
                    error_messages.append(f"{agent}: {error_msg}")
                    
                    # Try fallback - log error but continue with other agents
                    try:
                        # Attempt a retry with simpler prompt if possible
                        logger.info(f"Attempting fallback for agent {agent}")
                        fallback_response = self.fallback_query(agent, query)
                        
                        if fallback_response:
                            logger.info(f"Got fallback response of type: {type(fallback_response)}")
                            
                            # Extract string content from fallback response
                            if isinstance(fallback_response, dict) and "response" in fallback_response:
                                fallback_response = fallback_response["response"]
                            
                            if isinstance(fallback_response, str) and fallback_response.strip():
                                logger.info(f"Fallback successful for agent {agent}")
                                
                                # Ensure agent is a hashable type
                                if isinstance(agent, (str, int, float, bool, tuple)):
                                    agent_key = f"{agent} (fallback)"
                                else:
                                    if isinstance(agent, dict):
                                        # For dict agents, use the id or name as key if available
                                        if "id" in agent:
                                            agent_key = f"{agent['id']} (fallback)"
                                        elif "name" in agent:
                                            agent_key = f"{agent['name']} (fallback)"
                                        else:
                                            # Create a stable representation of the dictionary
                                            agent_key = f"{str(sorted(agent.items()) if hasattr(agent, 'items') else agent)} (fallback)"
                                    else:
                                        agent_key = f"{str(agent)} (fallback)"
                                        
                                logger.warning(f"Using fallback agent key: {agent_key}")
                                    
                                # Normalize labels with fallback
                                try:
                                    labels = self.get_model_labels(agent)
                                    if not isinstance(labels, list):
                                        if labels is None:
                                            labels = []
                                        else:
                                            try:
                                                labels = list(labels)
                                            except:
                                                labels = [str(labels)]
                                except:
                                    labels = []
                                
                                responses.append({
                                    "agent": agent_key,
                                    "response": fallback_response,
                                    "labels": labels
                                })
                                valid_response_count += 1
                    except Exception as fallback_e:
                        logger.error(f"Fallback for agent {agent} also failed: {str(fallback_e)}")
            
            logger.info(f"Query processing complete. Received {valid_response_count} valid responses out of {len(agents)} agents")
            
            if not responses and sequential_backup_enabled:
                logger.warning("No valid responses received from parallel execution. Trying sequential execution...")
                # The sequential backup will be executed below if we have no responses
            else:
                # Validate final response structure
                for i, resp in enumerate(responses):
                    if not isinstance(resp, dict):
                        logger.error(f"Response {i} is not a dictionary: {type(resp)}")
                        if isinstance(resp, str):
                            # Try to convert string to proper response format
                            responses[i] = {
                                "agent": f"Unknown Agent {i}",
                                "response": resp,
                                "labels": []
                            }
                            logger.info(f"Converted string response to dictionary format for response {i}")
                        else:
                            # Remove invalid response
                            logger.warning(f"Removing invalid response at index {i}")
                            responses[i] = None
                
                # Remove None values from responses
                responses = [r for r in responses if r is not None]
                
                if responses:
                    return responses
                    
        except Exception as parallel_error:
            logger.error(f"Parallel execution failed: {str(parallel_error)}")
            logger.exception("Exception details for parallel execution failure:")
        
        # If we got here, either parallel execution failed completely or returned no valid responses
        # Try sequential execution as a backup if enabled
        if sequential_backup_enabled:
            logger.warning("Using sequential execution backup")
            backup_responses = []
            
            for i, agent in enumerate(agents):
                try:
                    logger.info(f"Sequential backup: Processing agent {i+1}")
                    # Try with a simplified approach
                    if isinstance(agent, dict) and "id" in agent:
                        agent_id = agent["id"]
                    else:
                        agent_id = str(agent)
                        
                    # Extremely simplified prompt
                    simple_messages = [
                        {"role": "system", "content": "You are a helpful assistant. Provide a clear and concise response."},
                        {"role": "user", "content": query}
                    ]
                    
                    # Direct API call with minimal parameters
                    try:
                        url = f"{self.api_handler.base_url}/chat/completions"
                        payload = {
                            "model": agent_id,
                            "messages": simple_messages,
                            "max_tokens": 500,
                            "temperature": 0.7
                        }
                        
                        logger.info(f"Sequential backup: Sending direct API request to {agent_id}")
                        response = requests.post(
                            url, 
                            headers=self.api_handler.headers,
                            json=payload,
                            timeout=30
                        )
                        
                        if response.status_code == 200:
                            data = response.json()
                            if "choices" in data and len(data["choices"]) > 0:
                                choice = data["choices"][0]
                                if "message" in choice and "content" in choice["message"]:
                                    content = choice["message"]["content"]
                                    if content and isinstance(content, str) and content.strip():
                                        logger.info(f"Sequential backup: Got valid content from {agent_id}")
                                        backup_responses.append({
                                            "agent": agent_id,
                                            "response": content,
                                            "labels": self.get_model_labels(agent)
                                        })
                                        continue
                            
                            logger.warning(f"Sequential backup: Could not extract valid content from API response")
                    except Exception as api_error:
                        logger.error(f"Sequential backup: Direct API call failed: {str(api_error)}")
                    
                    # If we got here, try the standard methods as a last resort
                    system_prompt = "Please answer the following question concisely and accurately."
                    response = self.query_single_agent(agent, query, system_prompt)
                    
                    if isinstance(response, dict) and "response" in response:
                        response_content = response["response"]
                    elif isinstance(response, str):
                        response_content = response
                    else:
                        logger.warning(f"Sequential backup: Invalid response type: {type(response)}")
                        continue
                    
                    if response_content and isinstance(response_content, str) and response_content.strip():
                        backup_responses.append({
                            "agent": agent_id,
                            "response": response_content,
                            "labels": self.get_model_labels(agent)
                        })
                        
                except Exception as seq_error:
                    logger.error(f"Sequential backup: Failed for agent {i+1}: {str(seq_error)}")
            
            if backup_responses:
                logger.info(f"Sequential backup: Retrieved {len(backup_responses)} valid responses")
                return backup_responses
            else:
                logger.error("Sequential backup also failed to get any valid responses")
                
                # Last resort: Create a dummy response so the system doesn't crash
                logger.warning("Creating emergency fallback response")
                
                # Check if OpenRouter API key is valid
                api_key = self.api_handler.api_key
                if not api_key or len(api_key) < 20:  # Simple validation
                    error_message = "OpenRouter API key appears to be missing or invalid. Please check your .env file and ensure a valid API key is set."
                else:
                    error_message = "The system encountered issues retrieving responses from AI models. This could be due to API rate limits, service availability, or issues with the selected models. Please try again with a different query or select different models."
                    
                emergency_response = [{
                    "agent": "Emergency Fallback",
                    "response": error_message,
                    "labels": ["general_assistant"]
                }]
                return emergency_response
        
        # If we get here and have no responses but sequential backup is disabled, return empty list
        logger.error("No valid responses could be retrieved from any method")
        return []

    def fallback_query(self, agent, query):
        """
        Attempt a simpler query as fallback when the main query fails
        
        Args:
            agent (str): The agent model to query
            query (str): The user query
            
        Returns:
            str: The agent's response or None if failed
        """
        try:
            # Use a simplified system prompt and message format
            simple_system_prompt = "You are a helpful AI assistant. Provide a clear and concise response."
            
            # Make a simpler request with fewer tokens and simpler formatting
            response = self.api_handler.generate_text(
                model=agent,
                messages=[
                    {"role": "system", "content": simple_system_prompt},
                    {"role": "user", "content": f"Please answer this question briefly: {query}"}
                ],
                max_tokens=500  # Use fewer tokens for fallback
            )
            
            return response
        except Exception as e:
            logger.error(f"Fallback query failed: {str(e)}")
            return None

    def query_single_agent(self, agent, query, system_prompt):
        """
        Query a single agent with the given query and system prompt.
        
        Args:
            agent (str or dict): The agent model to query
            query (str): The user query
            system_prompt (str): The system prompt for the agent
            
        Returns:
            str: The agent's response
        """
        try:
            # Log the agent type and identification details
            if isinstance(agent, dict):
                agent_id = agent.get("id", "Unknown")
                agent_name = agent.get("name", "Unknown")
                logger.info(f"Querying agent dict: id={agent_id}, name={agent_name}")
            else:
                logger.info(f"Querying agent string: {agent}")
            
            # Prepare the query request
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ]
            
            logger.info(f"Sending message to agent with system prompt ({len(system_prompt)} chars)")
            
            # Make the query request with messages parameter
            response = self.api_handler.generate_text(
                model=agent,
                messages=messages,
                max_tokens=1000
            )
            
            # Log response structure
            if response:
                if isinstance(response, dict):
                    logger.info(f"Received dictionary response with keys: {response.keys()}")
                else:
                    logger.info(f"Received response of type: {type(response)}")
            
            return response
        except Exception as e:
            logger.error(f"Query failed for agent {agent}: {str(e)}")
            logger.exception("Exception details:")
            return None
    
    def _get_models_with_label(self, label):
        """
        Get models that have a specific label
        
        Args:
            label (str): Label to match
            
        Returns:
            list: List of models with the specified label
        """
        logger.info(f"Getting models with label: {label}")
        logger.info(f"model_to_labels has {len(self.model_to_labels)} items")
        
        matching_models = []
        for model_id, labels in self.model_to_labels.items():
            logger.info(f"Checking model: {model_id} with labels: {labels}")
            if label in labels:
                logger.info(f"Found match: {model_id} has label {label}")
                # Return the actual model object
                if model_id in self.model_index:
                    model = self.model_index[model_id]
                    matching_models.append(model)
                    logger.info(f"Added model {model_id} to matching models")
                else:
                    logger.warning(f"Model {model_id} has the label but not found in model_index")
        
        logger.info(f"Returning {len(matching_models)} models for label {label}")
        return matching_models
    
    def _get_labels_for_model(self, model_id):
        """
        Get the labels for a specific model
        
        Args:
            model_id (str): Model identifier
            
        Returns:
            list: List of labels for the model
        """
        return self.model_to_labels.get(model_id, [])
    
    def _get_role_description(self, label):
        """
        Get the role description for a specific label
        
        Args:
            label (str): Label to get role description for
            
        Returns:
            str: Role description or None if not found
        """
        if self.model_roles_data and "labels" in self.model_roles_data:
            for label_info in self.model_roles_data.get("labels", []):
                if label_info.get("label") == label:
                    return label_info.get("role_description")
        return None
    
    def _is_better_model(self, model_a, model_b):
        """
        Compare two models to determine which is better
        
        This is a simplified comparison based on context length and model reputation
        
        Args:
            model_a, model_b: Model objects to compare
            
        Returns:
            bool: True if model_a is better than model_b
        """
        # Preferred model families (in order of preference)
        preferred_families = ["anthropic/claude", "openai/gpt-4", "meta-llama/llama-3", "google/gemini", "mistral"]
        
        # Check context length
        context_a = model_a.get("context_length", 0)
        context_b = model_b.get("context_length", 0)
        
        if context_a > context_b * 1.5:  # Significantly more context
            return True
        elif context_b > context_a * 1.5:  # Significantly less context
            return False
        
        # Check for preferred model families
        for family in preferred_families:
            a_matches = family in model_a["id"].lower()
            b_matches = family in model_b["id"].lower()
            
            if a_matches and not b_matches:
                return True
            elif b_matches and not a_matches:
                return False
        
        # If equal on other factors, prefer newer models (assuming higher version numbers)
        model_a_name = model_a["id"].lower()
        model_b_name = model_b["id"].lower()
        
        # Extract version numbers if present
        import re
        a_version = re.search(r'(\d+\.\d+)', model_a_name)
        b_version = re.search(r'(\d+\.\d+)', model_b_name)
        
        if a_version and b_version:
            return float(a_version.group(1)) > float(b_version.group(1))
        
        # Default to model A if we can't determine
        return True
    
    def _calculate_cost_effectiveness(self, model):
        """
        Calculate a cost-effectiveness score for a model
        
        Args:
            model: Model object with pricing information
            
        Returns:
            float: Cost-effectiveness score (higher is better)
        """
        # For free models, give a high score but not infinite
        if model.get("pricing", {}).get("is_free", False):
            return 1000
        
        # Calculate average cost per token
        pricing = model.get("pricing", {})
        prompt_cost = pricing.get("prompt", 0)
        completion_cost = pricing.get("completion", 0)
        avg_cost = (prompt_cost + completion_cost) / 2
        
        if avg_cost == 0:  # Avoid division by zero
            return 1000
        
        # Factors in the calculation:
        # 1. Context length (bigger is better)
        # 2. Average cost per token (lower is better)
        # 3. Model quality factor (based on model family)
        
        context_length = model.get("context_length", 4000)
        
        # Determine quality factor based on model family
        quality_factor = 1.0
        model_id = model["id"].lower()
        
        if "claude" in model_id and "opus" in model_id:
            quality_factor = 3.0
        elif "claude" in model_id or "gpt-4" in model_id:
            quality_factor = 2.5
        elif "llama-3" in model_id or "gemini" in model_id:
            quality_factor = 2.0
        elif "mistral" in model_id:
            quality_factor = 1.5
        
        # Calculate cost-effectiveness score
        # Higher context, lower cost, and higher quality increase the score
        score = (context_length / 10000) * quality_factor / avg_cost
        
        return score

    def get_model_labels(self, model_id):
        """
        Get the labels for a specific model
        
        Args:
            model_id (dict or str): Model identifier or model object
            
        Returns:
            list: List of labels for the model
        """
        # Handle the case where model_id is a dictionary (model object)
        if isinstance(model_id, dict):
            # Try to get the id from the dictionary
            if "id" in model_id:
                return self.model_to_labels.get(model_id["id"], [])
            else:
                logger.warning(f"Could not extract id from model dictionary: {model_id}")
                return []
        else:
            # Handle the case where model_id is already a string or other hashable type
            try:
                return self.model_to_labels.get(model_id, [])
            except Exception as e:
                logger.error(f"Error getting labels for model {model_id}: {str(e)}")
                return []

    def create_system_prompt(self, model_id):
        """
        Create a system prompt for a specific model
        
        Args:
            model_id (str): Model identifier
            
        Returns:
            str: System prompt for the model
        """
        labels = self.get_model_labels(model_id)
        
        # Get role descriptions for the model's labels
        role_descriptions = []
        for label in labels:
            description = self._get_role_description(label)
            if description:
                role_descriptions.append(description)
        
        # Create a prompt with role context if available
        if role_descriptions:
            prompt = f"""
            {' '.join(role_descriptions)}
            
            Please answer the following query:
            
            Provide a detailed, accurate, and helpful response.
            """
        else:
            # Generic prompt if no role descriptions available
            prompt = f"""
            As an AI assistant with expertise in various domains, please answer the following query:
            
            Provide a detailed, accurate, and helpful response.
            """
        
        return prompt
