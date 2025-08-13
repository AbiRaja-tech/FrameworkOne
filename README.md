# FrameworkOne

A comprehensive travel planning framework that combines commonsense reasoning, tool integration, and constraint programming for intelligent city itinerary planning.

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/AbiRaja-tech/FrameworkOne.git
cd FrameworkOne
```

### 2. Set Up Virtual Environment
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows (PowerShell):
.\.venv\Scripts\Activate.ps1

# On Windows (Command Prompt):
.\.venv\Scripts\activate.bat

# On macOS/Linux:
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

**Note:** If `requirements.txt` doesn't exist, install these core packages:
```bash
pip install langgraph langchain-core google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client
```

### 4. Set Up Google Cloud Credentials
1. Place your Google Cloud service account JSON file in the project root
2. **Important:** Never commit credentials to Git (they're already in `.gitignore`)
3. Set environment variable (optional):
   ```bash
   set GOOGLE_APPLICATION_CREDENTIALS=path/to/your/credentials.json
   ```

## 🏃‍♂️ Running the Application

### Main Orchestrator (Recommended)
The orchestrator runs the complete pipeline from user query to final itinerary:

```bash
# Quick run with default settings
python orchestrator.py

# Custom query with stamina-first policy
python orchestrator.py --query "5-day London trip for museums and parks" --policy policy_stamina_first.json

# Verbose constraint programming logs
python orchestrator.py --cp-verbose
```

### Individual Components

#### Tool Graph
Test the tool integration (Google Sheets + distance calculations):
```bash
python tool_graph.py
```

#### Commonsense Agent
Test the commonsense reasoning pipeline:
```bash
python commonsense_agent.py
```

#### Intracity Planner
Test the constraint programming solver:
```bash
python intracity_planner.py
```

## 📁 Project Structure

```
FrameworkOne/
├── orchestrator.py          # Main pipeline orchestrator
├── commonsense_agent.py     # Commonsense reasoning agent
├── commonsense_graph.py     # LangGraph for commonsense pipeline
├── tool_agent.py           # Google Sheets integration
├── tool_graph.py           # Tool execution pipeline
├── intracity_planner.py    # Constraint programming solver
├── adapter.py              # Data format conversion
├── policy_*.json           # Planning policies
├── data_city_large.json    # City data
└── .venv/                  # Virtual environment
```

## 🔧 Configuration

### Policy Files
- `policy_budget_first.json` - Optimizes for budget constraints
- `policy_stamina_first.json` - Optimizes for energy/stamina management

### Environment Variables
- `GOOGLE_APPLICATION_CREDENTIALS` - Path to Google Cloud credentials
- `LOG_LEVEL` - Logging level (DEBUG, INFO, WARNING, ERROR)

## 🐛 Troubleshooting

### Common Issues

1. **Virtual Environment Not Activated**
   - Ensure you see `(.venv)` in your terminal prompt
   - Re-run the activation command if needed

2. **Missing Dependencies**
   - Activate virtual environment first
   - Run `pip install -r requirements.txt`

3. **Google Sheets API Errors**
   - Check credentials file exists and is valid
   - Verify Google Sheets API is enabled in your project

4. **Import Errors**
   - Ensure you're running scripts from the project root directory
   - Check that virtual environment is activated

### Getting Help
- Check the error messages in the terminal
- Verify all dependencies are installed
- Ensure credentials are properly configured

## 📚 Usage Examples

### Basic Travel Planning
```bash
python orchestrator.py --query "3-day Paris trip for art lovers"
```

### Budget-Conscious Planning
```bash
python orchestrator.py --query "Weekend getaway to Edinburgh" --policy policy_budget_first.json
```

### Detailed Logging
```bash
python orchestrator.py --query "London museums tour" --cp-verbose
```

## 🔒 Security Notes

- **Never commit credentials** to version control
- The `.gitignore` file already excludes credential files
- Use environment variables for sensitive configuration in production

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
