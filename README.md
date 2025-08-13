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

