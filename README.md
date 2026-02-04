# AI Data Analyst: A Multi-Agent Autonomous Framework

This project implements a sophisticated multi-agent system designed to automate the end-to-end data science lifecycle. Utilizing LangGraph, LangChain, and Llama 3.3 (via Groq), the system breaks down complex data tasks into specialized agents that profile, clean, merge, and analyze datasets autonomously.

---

## Project Architecture and Workflow

The system is designed as a series of specialized sub-agents. Each agent is responsible for a specific stage of the data pipeline, ensuring high accuracy and the ability to self-correct during code execution.

![Ingestion Architecture Placeholder](./screenshots/ingestion_architecture.png)

### 1. Data Ingestion Agent
The Ingestion Agent serves as the entry point for all datasets. Its primary responsibility is to load raw files into memory while performing initial optimizations.
*   **Memory Optimization:** The agent detects large datasets (e.g., >50,000 rows) and automatically converts object/string columns to categorical types, significantly reducing memory overhead.
*   **Schema Extraction:** It generates a high-level summary of the data, including data types, row counts, and sample records, which is then passed to downstream agents.

### 2. Merge Agent
Designed for relational data tasks, the Merge Agent handles multi-file integration.
*   **Relational Reasoning:** The agent analyzes the schemas of multiple uploaded files to identify potential join keys (e.g., matching 'user_id' in one file with 'ID' in another).
*   **Strategy Selection:** It suggests merge strategies to the user, allowing for inner, outer, or left joins based on the identified relationships.
*   **Code Generation:** It writes and executes the Pandas `merge()` logic to produce a unified dataframe for analysis.

![Merge Workflow Placeholder](./screenshots/merge_workflow.png)

### 3. Preprocessing Agent (Deep Scanner)
The Preprocessing Agent is a three-stage pipeline focused on data quality.
*   **The Profiler:** Performs a deep scan of every column. It identifies "pollution" (non-numeric characters in numeric columns), outlier counts via Interquartile Range (IQR), and inconsistent string casing.
*   **The Strategist:** An LLM node that receives the profile and generates a cleaning JSON plan (e.g., "impute age with median," "normalize city names to lowercase").
*   **The Engineer:** This node translates the strategy into linear, modern Python code. It avoids deprecated syntax and ensures all operations (like forward-filling or type-casting) are performed efficiently.

![Preprocessing Workflow Placeholder](./screenshots/preprocessing_workflow.png)

### 4. Graph Agent
The Graph Agent focuses on automated Exploratory Data Analysis (EDA) and visualization.
*   **Analysis Planning:** Based on the data profile, it identifies the most significant variables and plans univariate and bivariate analyses.
*   **Automated Visualization:** It writes Python code using Seaborn and Matplotlib to generate charts (histograms, scatter plots, box plots).
*   **Artifact Generation:** Each chart is saved as a physical PNG file in the system sandbox for user review.

![Graph Visualization Placeholder](./screenshots/graph_visualization.png)

### 5. Chat with Data Agent
The primary interface for the end-user, this agent allows for natural language querying of the dataset.
*   **Dynamic Intent Routing:** It classifies user queries into 'Fact-Finding' (SQL-based) or 'Visualization' (Python-based).
*   **SQL Fact-Finder:** For specific data lookups, it utilizes DuckDB to execute high-performance SQL queries directly against the CSV files.
*   **Self-Correction Loop:** If the agent generates code that results in a runtime error, it observes the traceback and automatically regenerates the code to fix the issue.

![Analysis Workflow Placeholder](./screenshots/analysis_workflow.png)

---

## Technical Implementation Details

### The Python REPL (Read-Eval-Print Loop)
The core of this system's "acting" capability is the Python REPL tool. When an agent needs to perform a task, it doesn't just describe the action; it writes a functional Python script. This script is sent to the REPL, which:
1.  Executes the code in an isolated environment.
2.  Captures `stdout` (printed results) and returns them to the agent.
3.  Captures `stderr` (runtime errors) so the agent can debug itself.
4.  Persists the dataframe state in memory using a global scope.

### State Management and Checkpointing
The framework utilizes LangGraph's `StateGraph` to maintain context.
*   **TypedDict State:** Each agent shares a structured state containing the file path, the current dataframe metadata, and the conversation history.
*   **Persistence:** A PostgreSQL-backed checkpointer is used to save the state of every thread. This allows users to stop and resume complex analyses without losing progress or context.

---

## Directory Structure

```text
AI-DATA-ANALYST/
├── chat_data/                # Conversational EDA Module
│   ├── app/
│   │   ├── main.py           # FastAPI endpoints and LangGraph orchestration
│   │   ├── state.py          # Definition of AgentState TypedDict
│   │   ├── tools.py          # Python REPL and Schema extraction tools
│   │   └── sandbox/          # Storage for generated plots and temp files
│   ├── Dockerfile            # Containerization for the Chat API
│   └── docker-compose.yml    # Orchestration for API and Postgres Database
├── ingestion-merge/          # Data Integration Module
│   ├── merge_agent.py        # Logic for multi-file join reasoning
│   └── sandbox_artifacts/    # Temporary storage for merged datasets
├── preprocessing-graph/      # Data Cleaning Module
│   ├── preprocessing.py      # Column profiling and "Deep Scan" logic
│   ├── graph_agent.py        # Automated EDA and chart generation
│   └── generated_cleaning_script.py # The output of the Engineer node
└── screenshots/              # Architectural diagrams and workflow visuals
```

---

## Installation and Usage

### Prerequisites
*   Python 3.10+
*   PostgreSQL (or Docker for containerized setup)
*   Groq API Key (Llama 3.3 70B recommended)

### Setup
1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-repo/ai-data-analyst.git
    cd ai-data-analyst
    ```
2.  **Environment Configuration:**
    Create a `.env` file in the root directory:
    ```env
    GROQ_API_KEY=your_key_here
    DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/dbname
    ```
3.  **Run with Docker:**
    ```bash
    docker-compose up --build
    ```

---

## Future Roadmap

Currently, the system operates as a suite of highly specialized sub-agents. The next phase of development involves:
*   **Unified Orchestration:** Integrating all five agents into a single "Master Agent" that can autonomously route a task from ingestion through cleaning to final visualization.
*   **Advanced Statistical Testing:** Adding nodes for hypothesis testing (T-tests, ANOVA) and basic predictive modeling (Regression, Classification).
*   **Multi-Modal Feedback:** Allowing users to provide feedback on generated charts to refine visualization parameters dynamically.