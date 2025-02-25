# Multi-Agent System

A Streamlit-based application that orchestrates multiple AI agents to efficiently process user queries using the OpenRouter API.

## Features

- Fetches and displays current OpenRouter model list
- Allows users to select between free models, paid models, or an optimized mix
- Analyzes queries to determine appropriate agent roles
- Assigns specialized agents based on query type and user preferences
- Coordinates multiple agents to process complex queries
- Synthesizes responses from multiple agents into a coherent final answer
- Real-time progress tracking with status updates
- Error handling for API failures and context limitations
- Comprehensive analysis tab with query insights and visualizations
- Agent performance rating with detailed information display
- Visualizations of agent selection and performance metrics
- Easy reset functionality for quick testing of different queries
- Robust free/paid model selection respecting user preferences

## Setup

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Add your OpenRouter API key to the `.env` file:
   ```
   OPENROUTER_API_KEY=your_api_key_here
   ```
4. Ensure you have the required JSON files in the `data` directory:
   - `model_labels.json`: Contains label information for all models
   - `model_roles.json`: Contains role definitions mapped to labels

## Usage

Run the application:

```
streamlit run app.py
```

## How It Works

1. The application fetches the current OpenRouter model list
2. Users select their preferred agent configuration:
   - Free models only - Strictly uses models with free pricing
   - Paid models only - Uses premium paid models for higher quality  
   - Optimized mix - Balances cost-effectiveness with performance
3. Users enter their query
4. The Coordinator analyzes the query to identify relevant tags:
   - Intelligently maps query keywords to specialized agent labels
   - Determines query complexity (low, medium, high)
   - Ensures proper labeling for specialized queries (logic, math, code)
5. Based on tags, complexity, and user preference, appropriate agents are selected:
   - Ensures minimum agent count based on query complexity
   - Uses at least 2 agents for math and coding queries
   - Prioritizes agents with most relevant expertise
6. Selected agents process the query in parallel
7. The Coordinator evaluates the responses based on quality criteria:
   - Correctness (factual accuracy)
   - Completeness (addresses all aspects)
   - Clarity (well-structured)
   - Helpfulness (actionable information)
8. The result is presented to the user with:
   - Detailed agent performance information
   - Query analysis with complexity assessment
   - Agent selection visualization
   - Performance comparison charts

## System Components

- **Coordinator**: Analyzes queries, assigns labels, determines complexity, and synthesizes responses
- **AgentManager**: Selects and manages appropriate agents based on query labels and user preferences
- **OpenRouterAPIHandler**: Handles all API interactions with OpenRouter
- **UI Components**: 
  - Query input interface with Reset functionality
  - Response visualization with tabbed interface
  - Agent performance charts and visualizations
  - Detailed analysis dashboard

## Requirements

- Python 3.8+
- Streamlit
- Requests
- python-dotenv
- pandas
- altair

See requirements.txt for specific versions.
