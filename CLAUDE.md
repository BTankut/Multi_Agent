# Multi-Agent System Development Guide

## Commands
- **Run Application**: `streamlit run app.py`
- **Install Dependencies**: `pip install -r requirements.txt`
- **Environment Setup**: Create `.env` file with `OPENROUTER_API_KEY=your_key_here`
- **Test**: `cd tests && python test_multi_agent.py` (manual testing framework)
- **Test UI Reset**: Use Reset button to clear state between tests

## Code Style Guidelines
- **Imports**: Group standard library, third-party, and local imports with a blank line between groups
- **Formatting**: Use 4 spaces for indentation, max line length 100 characters
- **Naming**:
  - Classes: `PascalCase`
  - Functions/Methods: `snake_case`
  - Variables: `snake_case`
  - Constants: `UPPER_SNAKE_CASE`
- **Error Handling**: Use try/except blocks with specific exceptions and meaningful error messages
- **Logging**: Use standard logging module with appropriate levels (info, warning, error)
- **Documentation**: All classes and functions should have docstrings with parameter descriptions
- **JSON Data**: Store configuration in `data/model_labels.json` and `data/model_roles.json`
- **State Management**: Use Streamlit session state with callbacks for widget interactions

## Architecture
- **Coordinator**: Analyzes queries, assigns labels, determines complexity, rates responses
- **AgentManager**: Handles model selection with free/paid preference enforcement
- **OpenRouterAPIHandler**: Manages API communication with robust error handling
- **UI Components**: Tabbed interface with visualizations using pandas/altair