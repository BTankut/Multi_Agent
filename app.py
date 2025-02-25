import os
import json
import requests
import time
import logging
import streamlit as st
import pandas as pd
import altair as alt
from dotenv import load_dotenv
from agents.coordinator import Coordinator
from utils.api_handler import OpenRouterAPIHandler
from utils.agent_manager import AgentManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('App')

# Load environment variables
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="Multi-Agent System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a clean, modern UI with pastel colors
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        background-color: #a8c0ff;
        color: #333;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #8da6e8;
    }
    .stTextInput>div>div>input {
        border-radius: 5px;
    }
    .stSelectbox>div>div {
        border-radius: 5px;
    }
    .stProgress>div>div {
        background-color: #a8c0ff;
    }
</style>
""", unsafe_allow_html=True)

# Reset fonksiyonu tanımla
def reset_app():
    # Widget ile bağlantılı olmayan state değişkenlerini sıfırla
    st.session_state.response = None
    st.session_state.agent_details = None
    st.session_state.error = None
    st.session_state.progress = 0
    st.session_state.status = ""
    st.session_state.query_submitted = False
    
    # Widget ile bağlantılı anahtarı temizle    
    if 'query' in st.session_state:
        st.session_state.query = ""
    
    # Diğer anahtarları temizle
    keys_to_delete = ['query_labels', 'complexity']
    for key in keys_to_delete:
        if key in st.session_state:
            del st.session_state[key]

# Initialize session state
if 'models_data' not in st.session_state:
    st.session_state.models_data = None
if 'selected_option' not in st.session_state:
    st.session_state.selected_option = None
if 'coordinator_model' not in st.session_state:
    st.session_state.coordinator_model = "anthropic/claude-3-opus-20240229"
if 'query_submitted' not in st.session_state:
    st.session_state.query_submitted = False
if 'progress' not in st.session_state:
    st.session_state.progress = 0
if 'status' not in st.session_state:
    st.session_state.status = ""
if 'response' not in st.session_state:
    st.session_state.response = None
if 'agent_details' not in st.session_state:
    st.session_state.agent_details = None
if 'error' not in st.session_state:
    st.session_state.error = None

# Load model labels and roles from JSON files
def load_model_data():
    try:
        logger.info("Loading model data from JSON files")
        with open("data/model_labels.json", "r") as f:
            model_labels = json.load(f)
            logger.info(f"Loaded model_labels.json with {len(model_labels)} model entries")
        
        with open("data/model_roles.json", "r") as f:
            model_roles = json.load(f)
            logger.info(f"Loaded model_roles.json with {len(model_roles.get('labels', []))} label entries")
        
        return model_labels, model_roles
    except Exception as e:
        logger.error(f"Error loading model data: {str(e)}", exc_info=True)
        st.error(f"Error loading model data: {str(e)}")
        return None, None

# Functions
def fetch_openrouter_models():
    """Fetch available models from OpenRouter API and store in session state"""
    st.session_state.status = "Fetching models from OpenRouter..."
    st.session_state.progress = 10
    api_handler = OpenRouterAPIHandler()
    
    try:
        logger.info("Fetching models from OpenRouter API")
        models_data = api_handler.get_models()
        logger.info(f"Successfully fetched {len(models_data)} models from OpenRouter")
        
        # Filter out specific models
        filtered_models = []
        filtered_out = []
        models_to_filter = ["claude-3.7", "google/gemini-2.0-flash-lite-001", 
                          "anthropic/claude-3.7-sonnet:beta", "anthropic/claude-3.7-sonnet",
                          "google/gemini-2.0-flash-001"]
        
        for model in models_data:
            model_id = model.get("id", "").lower()
            if any(filter_id.lower() in model_id for filter_id in models_to_filter):
                filtered_out.append(model.get("id", "unknown"))
                logger.info(f"Filtered out model: {model.get('id', 'unknown')}")
            else:
                filtered_models.append(model)
        
        logger.info(f"Filtered out {len(filtered_out)} models: {filtered_out}")
        logger.info(f"Remaining models after filtering: {len(filtered_models)}")
        
        st.session_state.models_data = filtered_models
        st.session_state.progress = 30
        st.session_state.status = f"Successfully fetched {len(filtered_models)} models from OpenRouter"
        
        # Ensure coordinator_model is initialized if it doesn't exist
        if "coordinator_model" not in st.session_state or not st.session_state.coordinator_model:
            # Try several good coordinator models in order of preference
            default_models = [
                "anthropic/claude-3-opus-20240229",
                "anthropic/claude-3-sonnet-20240229",
                "anthropic/claude-3-5-sonnet-20240620",
                "anthropic/claude-3-haiku-20240307",
                "openai/gpt-4-turbo",
                "openai/gpt-4o",
                "mistral/mistral-large"
            ]
            
            # Try to find the first available model from our preferred list
            coordinator_set = False
            for default_model in default_models:
                if any(model.get("id") == default_model for model in filtered_models):
                    st.session_state.coordinator_model = default_model
                    logger.info(f"Set default coordinator model to {default_model}")
                    coordinator_set = True
                    break
            
            # If none of our preferred models were found, use the first available model
            if not coordinator_set:
                if filtered_models:
                    st.session_state.coordinator_model = filtered_models[0].get("id")
                    logger.info(f"No preferred models found, set coordinator model to {st.session_state.coordinator_model}")
                else:
                    logger.warning("No models available to set as coordinator")
        
        return True
    except Exception as e:
        logger.error(f"Error fetching models: {str(e)}", exc_info=True)
        st.session_state.error = f"Error fetching models: {str(e)}"
        st.session_state.progress = 0
        return False

def process_query():
    try:
        # Reset previous response
        st.session_state.response = None
        st.session_state.agent_details = None
        st.session_state.error = None
        
        query = st.session_state.query
        option = st.session_state.selected_option
        coordinator_model = st.session_state.coordinator_model
        
        logger.info(f"Processing query: '{query}' with option: {option}")
        logger.info(f"Selected coordinator model: {coordinator_model}")
        
        if not query:
            st.session_state.error = "Please enter a query"
            return
        
        if not option:
            st.session_state.error = "Please select an option"
            return
        
        if not st.session_state.models_data:
            st.session_state.error = "Please fetch models from OpenRouter first"
            return
        
        st.session_state.status = "Loading model data..."
        st.session_state.progress = 35
        
        # Load labels and roles data
        model_labels, model_roles = load_model_data()
        if not model_labels or not model_roles:
            st.session_state.error = "Failed to load model data"
            st.session_state.progress = 0
            return
        
        logger.info(f"Model data loaded. model_labels: {type(model_labels)}, model_roles: {type(model_roles)}")
        
        st.session_state.status = "Initializing coordinator..."
        st.session_state.progress = 40
        
        # Initialize coordinator and agent manager
        logger.info(f"Initializing coordinator with model: {coordinator_model}")
        coordinator = Coordinator(model_labels, coordinator_model)
        agent_manager = AgentManager(st.session_state.models_data, model_labels, model_roles)
        
        # Analyze query and assign labels
        st.session_state.status = "Analyzing query..."
        st.session_state.progress = 50
        query_labels = coordinator.analyze_query(query)
        logger.info(f"Query analysis results: {query_labels}")
        
        # Store query labels in session state for display
        st.session_state.query_labels = query_labels
        
        # Determine query complexity to decide how many agents to use
        st.session_state.status = "Determining query complexity..."
        complexity = coordinator.determine_complexity(query, query_labels)
        logger.info(f"Query complexity determined as: {complexity}")
        
        # Store complexity in session state for display
        st.session_state.complexity = complexity
        
        # Check if the query is related to math or coding - if so, ensure we use at least 2 agents
        is_math_or_coding_query = any(label in ["math_expert", "code_expert"] for label in query_labels)
        min_agents = 2 if is_math_or_coding_query else 1
        if is_math_or_coding_query:
            logger.info(f"Query involves math or coding, setting minimum agents to {min_agents}")
            st.session_state.status = f"Math/Coding query detected, using at least {min_agents} agents..."
        
        # Map complexity to desired number of agents
        if complexity == "high":
            desired_agents = max(3, min_agents)  # At least 3 for high complexity
        elif complexity == "medium":
            desired_agents = max(2, min_agents)  # At least 2 for medium complexity
        else:  # low complexity
            desired_agents = min_agents  # 1 or 2 (if math/coding)
        
        logger.info(f"Based on complexity '{complexity}' and query type, using {desired_agents} agents")
        
        # Select agents based on labels and option
        st.session_state.status = f"Selecting {desired_agents} appropriate agents..."
        st.session_state.progress = 60
        agents = agent_manager.select_agents(query_labels, option, desired_agents)
        
        if not agents:
            logger.error("No suitable agents found for this query")
            st.session_state.error = "No suitable agents found for this query"
            st.session_state.progress = 0
            return
        
        # Process query with selected agents
        st.session_state.status = "Processing query with selected agents..."
        st.session_state.progress = 70
        
        try:
            agent_responses = agent_manager.process_query(agents, query)
            
            if not agent_responses:
                logger.error("No valid responses received from agents")
                st.session_state.error = "No valid responses received from agents. API returned empty content."
                st.session_state.progress = 0
                st.session_state.status = "Query processing failed"
                return
            
            # Coordinator evaluates responses
            st.session_state.status = "Evaluating agent responses..."
            st.session_state.progress = 90
            final_response, agent_details = coordinator.evaluate_responses(agent_responses, agents)
            
            # Check if this was a math or coding query and if we need to show agent details
            if is_math_or_coding_query:
                response_text = f"**Final Response:**\n\n{final_response}\n\n"
                response_text += "**Note:** Multiple agents were used to ensure accuracy for this math/coding query.\n"
                final_response = response_text
            
            # Update session state with results
            st.session_state.response = final_response
            st.session_state.agent_details = agent_details
            st.session_state.status = f"Query processed successfully with {len(agent_responses)} agents"
            st.session_state.progress = 100
            st.session_state.query_submitted = True
            
            # Log the response for debugging
            logger.info(f"Final response generated: {final_response[:100]}...")  # Log first 100 chars
            logger.info(f"Agent details: {json.dumps(agent_details, indent=2)}")
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}", exc_info=True)
            st.session_state.error = f"Error processing query: {str(e)}"
            st.session_state.progress = 0
            st.session_state.status = "Query processing failed"
            
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        st.session_state.error = f"Error processing query: {str(e)}"
        st.session_state.progress = 0

# UI Layout
st.title("Multi-Agent System")

col1, col2 = st.columns([3, 1])

with col2:
    st.subheader("System Controls")
    
    if st.button("Fetch OpenRouter Models"):
        fetch_openrouter_models()
    
    if st.session_state.models_data:
        st.success(f"✅ Loaded {len(st.session_state.models_data)} models")
    
    st.selectbox(
        "Select Agent Configuration",
        ["Free models only", "Paid models only", "Optimized mix of free and paid models"],
        key="selected_option"
    )
    
    # Koordinatör model seçimi için OpenRouter'dan alınan tüm modelleri kullan
    if st.session_state.models_data:
        # Tüm model ID ve isimlerini hazırla
        coordinator_model_options = []
        coordinator_model_ids = []
        
        for model in st.session_state.models_data:
            if "id" in model and "name" in model:
                model_display = f"{model['name']} ({model['id']})"
                coordinator_model_options.append(model_display)
                coordinator_model_ids.append(model['id'])
        
        # Eğer mevcut seçili model listede yoksa, ilk modeli seç
        if st.session_state.coordinator_model not in coordinator_model_ids and coordinator_model_ids:
            st.session_state.coordinator_model = coordinator_model_ids[0]
        
        # Seçim kutusu oluştur
        selected_index = 0
        if st.session_state.coordinator_model in coordinator_model_ids:
            selected_index = coordinator_model_ids.index(st.session_state.coordinator_model)
        
        selected_option = st.selectbox(
            "Select Coordinator Model",
            coordinator_model_options,
            index=selected_index
        )
        
        # Seçilen modelin ID'sini al ve session_state'e kaydet
        if selected_option:
            selected_index = coordinator_model_options.index(selected_option)
            st.session_state.coordinator_model = coordinator_model_ids[selected_index]
    else:
        # Model verileri yoksa, varsayılan seçenekleri göster
        default_models = [
            "anthropic/claude-3-opus-20240229",
            "anthropic/claude-3-5-sonnet-20240620",
            "anthropic/claude-3-haiku-20240307",
            "openai/gpt-4-turbo",
            "google/gemini-1.5-pro-latest"
        ]
        
        st.selectbox(
            "Select Coordinator Model (Fetch models first for full list)",
            default_models,
            key="coordinator_model"
        )

with col1:
    st.subheader("Query Input")
    st.text_area("Enter your query here", height=150, key="query")
    
    # İki buton yan yana
    button_col1, button_col2 = st.columns([3, 1])
    
    with button_col1:
        submit_button = st.button("Submit Query", key="submit")
        if submit_button:
            try:
                st.session_state.status = "Processing query..."
                st.session_state.progress = 10
                process_query()
            except Exception as e:
                import traceback
                st.error(f"Error submitting query: {str(e)}")
                logger.error(f"Error in submit query button handler: {str(e)}")
                logger.error(traceback.format_exc())
    
    with button_col2:
        # Reset butonu
        reset_button = st.button("Reset", key="reset", on_click=reset_app)

# Progress bar and status
if st.session_state.progress > 0:
    st.progress(st.session_state.progress / 100)
    st.info(st.session_state.status)

# Display error if any
if st.session_state.error:
    st.error(st.session_state.error)

# Display response and query information if available
if st.session_state.response:
    # Create tabs for Result, Analysis, and Debug Information
    result_tab, analysis_tab = st.tabs(["Result", "Query Analysis"])
    
    with result_tab:
        st.subheader("Response")
        
        # Ensure response is properly displayed
        try:
            # Log detailed information about the response
            logger.info(f"Displaying response of type: {type(st.session_state.response)}")
            try:
                preview = str(st.session_state.response)[:100]
                logger.info(f"Response content preview: {preview}...")
            except Exception as preview_error:
                logger.error(f"Could not generate response preview: {str(preview_error)}")
            
            # Check response type and format accordingly
            if isinstance(st.session_state.response, str):
                # Try markdown first with error handling
                try:
                    st.markdown(st.session_state.response)
                    logger.info("Response displayed successfully using markdown")
                except Exception as markdown_error:
                    logger.error(f"Markdown display failed: {str(markdown_error)}, trying text display")
                    try:
                        st.text(st.session_state.response)
                        logger.info("Response displayed successfully using text")
                    except Exception as text_error:
                        logger.error(f"Both markdown and text display failed: {str(text_error)}")
                        st.error("Could not display response properly. See logs for details.")
            elif isinstance(st.session_state.response, dict):
                # For dictionary responses
                logger.info(f"Response is a dictionary with keys: {st.session_state.response.keys()}")
                
                # Check if the dictionary has a 'response' key
                if 'response' in st.session_state.response:
                    try:
                        content = st.session_state.response['response']
                        if isinstance(content, str):
                            st.markdown(content)
                        else:
                            st.write(content)
                        logger.info("Dictionary response displayed successfully")
                    except Exception as dict_error:
                        logger.error(f"Error displaying dictionary response: {str(dict_error)}")
                        st.write(st.session_state.response)
                else:
                    # Display the whole dictionary
                    st.write(st.session_state.response)
            else:
                # For non-string types
                logger.info(f"Response is not a string, using st.write()")
                try:
                    st.write(st.session_state.response)
                    logger.info("Response displayed successfully using write()")
                except Exception as write_error:
                    logger.error(f"Write display failed: {str(write_error)}")
                    st.error("Could not display complex response. See logs for details.")
            
            logger.info("Response display process completed")
        except Exception as e:
            st.error(f"Error displaying response: {str(e)}")
            logger.error(f"Error displaying response: {str(e)}", exc_info=True)
            
            # Enhanced fallback options in sequence
            logger.info("Starting fallback display sequence")
            
            # First fallback: Try code block
            try:
                logger.info("Trying code block display as fallback")
                st.code(str(st.session_state.response))
                logger.info("Code block display successful")
            except Exception as code_error:
                logger.error(f"Code block display failed: {str(code_error)}")
                
                # Second fallback: Try JSON
                try:
                    logger.info("Trying JSON display as fallback")
                    if isinstance(st.session_state.response, dict):
                        st.json(st.session_state.response)
                    else:
                        st.json({"response": str(st.session_state.response)})
                    logger.info("JSON display successful")
                except Exception as json_error:
                    logger.error(f"JSON display failed: {str(json_error)}")
                    
                    # Third fallback: Try raw text in expander
                    try:
                        logger.info("Trying raw text in expander as final fallback")
                        with st.expander("Raw Response"):
                            st.text(str(st.session_state.response))
                        logger.info("Raw text display successful")
                    except Exception as raw_error:
                        logger.error(f"All display methods failed: {str(raw_error)}")
                        st.error("Could not display response in any format. See logs for details.")
                        
                        # Provide helpful debug information
                        with st.expander("Debug Information"):
                            st.write("Response type:", type(st.session_state.response))
                            try:
                                st.write("Response length:", len(str(st.session_state.response)))
                            except:
                                st.write("Could not determine response length")
        
        # Display agent details directly in the result tab
        if st.session_state.agent_details:
            with st.expander("View Agent Details"):
                # Add a header for agent details
                st.subheader("Agents Used For This Query")
                
                # Create a better formatted display for agent details
                for i, agent in enumerate(st.session_state.agent_details):
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.markdown(f"**Agent {i+1}:** {agent.get('agent', 'Unknown')}")
                        st.markdown(f"**Rating:** {agent.get('rating', 'N/A')}")
                    
                    with col2:
                        # Show labels in a more readable format
                        labels = agent.get('labels', [])
                        if labels:
                            st.markdown("**Expertise Labels:**")
                            label_text = ", ".join([f"`{label}`" for label in labels])
                            st.markdown(label_text)
                        
                        # Show model information if available
                        model_info = agent.get('model_info', {})
                        if model_info:
                            name = model_info.get('name', 'Unknown')
                            context_length = model_info.get('context_length', 'Unknown')
                            is_free = "Free" if model_info.get('is_free', False) else "Paid"
                            
                            st.markdown(f"**Model Name:** {name}")
                            st.markdown(f"**Context Length:** {context_length}")
                            st.markdown(f"**Pricing Type:** {is_free}")
                    
                    # Add error status if any
                    error_status = agent.get('error_status')
                    if error_status:
                        st.warning(f"**Error Status:** {error_status}")
                    
                    # Add a separator between agents unless it's the last one
                    if i < len(st.session_state.agent_details) - 1:
                        st.markdown("---")
    
    # Create an analysis tab that shows more information about the query processing
    with analysis_tab:
        # Create a better visualization of the query analysis process
        st.subheader("Query Processing Information")
        
        # Create sections for different parts of the analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Query Analysis")
            # Display the selected coordinator model
            st.markdown(f"**Coordinator Model:** `{st.session_state.coordinator_model}`")
            
            # Display the agent selection mode
            if st.session_state.selected_option:
                st.markdown(f"**Agent Selection Mode:** `{st.session_state.selected_option}`")
            
            # Display the complexity if available in session state
            # We need to add complexity to session state in process_query()
            if 'complexity' in st.session_state:
                st.markdown(f"**Query Complexity:** `{st.session_state.complexity}`")
            
            # Display the selected labels if available
            if 'query_labels' in st.session_state:
                st.markdown("**Detected Query Labels:**")
                for label in st.session_state.query_labels:
                    st.markdown(f"- `{label}`")
            else:
                st.info("Query label information not available")
            
        with col2:
            st.markdown("### Agent Selection")
            
            # Display the number of agents used
            if st.session_state.agent_details:
                agent_count = len(st.session_state.agent_details)
                st.markdown(f"**Number of Agents:** {agent_count}")
                
                # Count how many free vs paid models were used
                if all(agent.get('model_info', {}).get('is_free', False) for agent in st.session_state.agent_details):
                    pricing_mix = "All Free Models"
                elif all(not agent.get('model_info', {}).get('is_free', False) for agent in st.session_state.agent_details):
                    pricing_mix = "All Paid Models"
                else:
                    free_count = sum(1 for agent in st.session_state.agent_details if agent.get('model_info', {}).get('is_free', False))
                    paid_count = agent_count - free_count
                    pricing_mix = f"Mixed ({free_count} Free, {paid_count} Paid)"
                
                st.markdown(f"**Pricing Mix:** {pricing_mix}")
                
                # Show specialized agents used
                specialized_labels = set()
                for agent in st.session_state.agent_details:
                    for label in agent.get('labels', []):
                        if label != 'general_assistant':
                            specialized_labels.add(label)
                
                if specialized_labels:
                    st.markdown("**Specialized Agent Types:**")
                    for label in sorted(specialized_labels):
                        st.markdown(f"- `{label}`")
                else:
                    st.info("No specialized agents were used for this query")
            else:
                st.info("Agent selection information not available")
        
        # Add data visualization of agent ratings
        if st.session_state.agent_details:
            st.markdown("### Agent Rating Visualization")
            
            # Prepare data for visualization
            agent_data = []
            for agent in st.session_state.agent_details:
                agent_name = agent.get('agent', 'Unknown')
                # Strip to a reasonable length if too long
                if len(agent_name) > 30:
                    agent_name = agent_name[:27] + "..."
                
                # Convert rating to float
                try:
                    rating = float(agent.get('rating', 0))
                except (ValueError, TypeError):
                    rating = 0
                
                # Check if this was the best agent (highest rating)
                is_best = False
                try:
                    best_rating = max(float(a.get('rating', 0)) for a in st.session_state.agent_details)
                    is_best = (rating == best_rating)
                except (ValueError, TypeError):
                    pass
                
                agent_data.append({
                    'Agent': agent_name,
                    'Rating': rating,
                    'Is Best': 'Selected' if is_best else 'Not Selected',
                    'Model Type': 'Free' if agent.get('model_info', {}).get('is_free', False) else 'Paid'
                })
            
            # Create DataFrame for visualization
            if agent_data:
                df = pd.DataFrame(agent_data)
                
                # Create a bar chart
                highlight_color = "#a8c0ff"  # Match the UI theme color
                base_color = "#d3d3d3"
                
                chart = alt.Chart(df).mark_bar().encode(
                    x=alt.X('Rating:Q', scale=alt.Scale(domain=[0, 10]), title='Rating (0-10)'),
                    y=alt.Y('Agent:N', sort='-x', title=None),
                    color=alt.Color('Is Best:N', 
                                   scale=alt.Scale(domain=['Selected', 'Not Selected'],
                                                  range=[highlight_color, base_color]),
                                   legend=None),
                    tooltip=['Agent', 'Rating', 'Model Type']
                ).properties(
                    title='Agent Performance Comparison',
                    height=min(250, 50 * len(agent_data))  # Adjust height based on number of agents
                )
                
                st.altair_chart(chart, use_container_width=True)
                
                # Also create a pie chart for model types
                model_type_counts = df['Model Type'].value_counts().reset_index()
                model_type_counts.columns = ['Model Type', 'Count']
                
                if len(model_type_counts) > 1:  # Only show pie chart if there are different model types
                    pie_chart = alt.Chart(model_type_counts).mark_arc().encode(
                        theta=alt.Theta(field="Count", type="quantitative"),
                        color=alt.Color(field="Model Type", type="nominal",
                                      scale=alt.Scale(domain=['Free', 'Paid'],
                                                     range=["#a8dadc", "#e63946"])),
                        tooltip=['Model Type', 'Count']
                    ).properties(
                        title='Agent Model Types',
                        width=200,
                        height=200
                    )
                    
                    st.altair_chart(pie_chart)
    

# Debug info
with st.expander("Debug Information"):
    st.write("Session State:", st.session_state)
