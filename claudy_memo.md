# Multi-Agent System: Development Overview

## Project Overview
This project implements a sophisticated multi-agent AI system that coordinates and orchestrates multiple language models through the OpenRouter API to efficiently process user queries. The system analyzes the query, determines its complexity and type, selects appropriate models based on user preferences, and coordinates multiple AI agents to deliver high-quality responses.

## Core Architecture

### Critical Components
1. **Coordinator Agent**: The central orchestrator that analyzes queries, assigns appropriate labels, determines complexity, and synthesizes responses from multiple agents.
2. **Agent Manager**: Responsible for selecting and managing multiple AI agents based on query labels and user preferences.
3. **API Handler**: Manages all interactions with the OpenRouter API, including model fetching and query processing.
4. **Streamlit Interface**: User-friendly interface for query input, agent selection, and response display.

### Key Data Resources
The system relies on two critical JSON files that serve as its knowledge base and configuration:

1. **`model_labels.json`**: Contains mappings between model names and their capability labels/tags
   - Each model is associated with specific labels (e.g., reasoning_expert, code_expert, math_expert)
   - This file defines which models are suitable for which types of tasks
   - This is the primary reference for agent selection

2. **`model_roles.json`**: Contains definitions of different agent roles
   - Maps labels to detailed role descriptions
   - Defines how each agent should behave based on its assigned label
   - Used to generate appropriate system prompts for each agent

These JSON files are **essential** for the system's operation. The Coordinator must only use labels defined in these files, as they form the foundation of the agent selection mechanism.

## Query Processing Flow

1. **Query Analysis**:
   - Coordinator analyzes the query using its model
   - Assigns labels from the available set in model_labels.json
   - Determines query complexity (low, medium, high)
   - For math and coding questions, ensures minimum 2 agents are used

2. **Agent Selection**:
   - Based on user's preference (Free, Paid, or Optimized), selects appropriate models
   - Uses complexity to determine the number of agents (1-3 based on complexity)
   - Ensures proper agent count based on query type
   - Special handling for math/coding queries (minimum 2 agents)

3. **Model Option Modes**:
   - **Free models only**: Strictly selects only models with free pricing
   - **Paid models only**: Selects only non-free models for higher quality
   - **Optimized mix**: Balances cost-effectiveness, using a mix of free and paid models based on query needs

4. **Parallel Query Processing**:
   - Selected agents process the query simultaneously
   - Each agent receives appropriate system prompts based on its labels
   - Responses are collected and validated
   - Error detection system identifies API rate limits, context limits, etc.

5. **Response Evaluation**:
   - Coordinator rates all responses based on correctness, completeness, clarity, and helpfulness
   - Selects the highest-rated response as the final answer
   - Adds error notices if any problems were detected
   - For math/coding, informs user that multiple agents were used

6. **Error Handling**:
   - Detects API rate limits, context limits, token limits, etc.
   - Implemented fallback mechanisms for API failures
   - Sequential execution backup when parallel execution fails
   - Emergency response when all else fails
   - Clear user notifications for all error states

## Recent Updates and Improvements

### Label Handling Updates
- Enhanced coordinator prompting to ensure it only uses labels from model_labels.json
- Improved label selection for logic puzzles to include both reasoning_expert and math_expert
- Updated coordinator prompt to better handle pattern recognition, arrangement tasks, and logic problems
- Added comprehensive keyword mapping for specific domain detection (math, logic, coding, etc.)
- Improved fallback mechanisms for when model returns invalid labels
- Comprehensive keyword-based labeling system when parsing fails

### Agent Selection Improvements
- Fixed option selection to ensure "Free models only" strictly uses free models through a robust 3-tier approach:
  1. Models with ":free" in ID/name (OpenRouter API convention)
  2. Models with "free" label in model_labels.json
  3. Models with zero pricing from API data (fallback)
- Enhanced logic puzzles handling by ensuring both "reasoning_expert" AND "math_expert" labels
- Implemented more intelligent agent selection based on query complexity
- Added special handling for math and coding queries (minimum 2 agents)
- Improved agent diversity to get a range of perspectives
- Comprehensive fallback mechanisms that respect user's model selection preference

### Error Handling Updates
- Added comprehensive error detection in responses
- Implemented automatic fallback to different models when errors occur
- Added clear user notifications for API issues, context limits, etc.
- Created sequential backup execution when parallel execution fails

### Response Evaluation Enhancements
- Implemented sophisticated rating system using the coordinator model
- Added fallback rating mechanism when evaluation fails
- Enhanced response aggregation and synthesis
- Added user notifications about multiple agent usage

### UI Improvements
- Better progress tracking and status updates
- Enhanced error messages and notifications
- Completely redesigned agent details display with comprehensive information
- Added new "Query Analysis" tab with detailed information about:
  - Detected query labels
  - Query complexity
  - Agent selection process
  - Agent ratings visualization
- Added visual charts showing agent performance and selection
- Improved results display with better formatting and organization
- Math/coding query special notifications

## Technical Implementation Notes

### Query Complexity Determination
The system determines query complexity through:
- Analysis of query content and structure
- Label types assigned to the query
- Keyword detection and pattern matching
- Coordinator model assessment

### Agent Count Selection
- Low complexity: 1 agent (or 2 for math/coding)
- Medium complexity: 2 agents
- High complexity: 3 agents

### Agent Selection Logic
- Applies tiered filtering approach based on user preference:
  - "Free models only": Selects only models verified as free through multiple checks
  - "Paid models only": Selects only premium paid models
  - "Optimized mix": Considers all models with best cost-effectiveness
- Identifies models matching query labels with robust fallback mechanisms
- Groups models by their expertise labels for optimal selection
- Prioritizes models with most relevant expertise for each query type
- Ensures minimum agent count based on complexity and query type
- For "optimized" mode, scores models based on cost-effectiveness
- Includes intelligent fallback mechanisms that respect user preferences
- Maintains strict adherence to the user's free/paid preference

### Error Detection
The system monitors responses for:
- API rate limit messages
- Context length limitations
- Token limit exceedances
- Server errors or timeouts
- Response quality issues

## Usage Instructions

1. Start the application with `streamlit run app.py`
2. Fetch current OpenRouter models using the button
3. Select agent configuration mode (Free, Paid, Optimized)
4. Enter your query
5. View results, including detailed agent information

## Recent Fix Summary (February 26, 2025)

### Major Improvements
1. **Fixed Free Models Option**
   - Implemented multi-tier free model detection: model ID pattern, labels, and pricing data
   - "Free models only" option now strictly adheres to free model selection
   - Added detailed logging for model classification process
   - Prioritized models with ":free" in ID (OpenRouter format) or "free" label in model_labels.json

2. **Enhanced Logic Puzzle Handling**
   - Updated coordinator to assign both "reasoning_expert" AND "math_expert" to logic puzzles
   - Added specific keyword patterns for logic puzzle detection (colors, arrangements, sequences, etc.)
   - Improved prompt engineering for better label assignment
   - Enhanced keyword mapping for specialized query types

3. **UI Enhancements**
   - Added new "Query Analysis" tab with detailed information
   - Implemented data visualizations for agent performance (bar charts, pie charts)
   - Redesigned agent details view with comprehensive information
   - Added "Reset" button for clearing query and results using Streamlit's callback pattern
   - Implemented state management best practices for widget interactions
   - Ensured smooth user experience when resetting between queries

4. **User Experience Improvements**
   - Single-click reset functionality to clear all results and state
   - Better organization of query input and model selection
   - More informative agent performance visualization
   - Clearer presentation of model selection process

### Technical Improvements
- Redesigned agent selection algorithm with better prioritization
- Enhanced fallback mechanisms with preference-specific behavior
- Improved robustness in handling edge cases
- Added more detailed logging throughout the process
- Streamlined model matching and label assignment

## Future Enhancements
- Improved agent specialization based on historical performance
- Enhanced error recovery mechanisms
- More sophisticated response synthesis for complex queries
- Expanded model compatibility and matching
- Better handling of vision and multimodal queries
- Personalized agent selection based on user history