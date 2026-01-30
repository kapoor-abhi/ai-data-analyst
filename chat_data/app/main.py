import os
import uuid
import shutil
import logging
import psycopg
from contextlib import asynccontextmanager
from typing import Literal
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.responses import FileResponse
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from psycopg_pool import AsyncConnectionPool
from dotenv import load_dotenv

# Import local modules
from app.state import AgentState
from app.tools import python_repl, get_csv_schema

load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
DB_URI = os.getenv("DATABASE_URL")
SANDBOX_DIR = "/code/app/sandbox"

# Ensure sandbox exists on startup
if not os.path.exists(SANDBOX_DIR):
    os.makedirs(SANDBOX_DIR, exist_ok=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles DB migrations and connection pooling."""
    try:
        logger.info("Connecting to PostgreSQL for migrations...")
        # Use autocommit=True to allow LangGraph to create indexes/tables
        async with await psycopg.AsyncConnection.connect(DB_URI, autocommit=True) as conn:
            checkpointer = AsyncPostgresSaver(conn)
            await checkpointer.setup()
        logger.info("Database migrations complete.")
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        raise e

    # Initialize connection pool for the API
    async with AsyncConnectionPool(conninfo=DB_URI, max_size=20) as pool:
        app.state.pool = pool
        yield

app = FastAPI(lifespan=lifespan)

# --- AGENT SETUP ---
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    groq_api_key=os.getenv("GROQ_API_KEY")
)
tools = [python_repl, get_csv_schema]
llm_with_tools = llm.bind_tools(tools)

def call_model(state: AgentState):
    messages = state['messages']
    # Inject System Message if not present
    if not any(isinstance(m, SystemMessage) for m in messages):
        sys_msg = SystemMessage(content=(
            "You are a Senior Data Analyst AI. You have access to a Python REPL.\n"
            f"The current file path is: {state['file_path']}\n"
            "1. ALWAYS start by calling 'get_csv_schema' to understand the data.\n"
            "2. Use pandas for analysis. Always use print() to show results.\n"
            "3. Save all plots to 'app/sandbox/temp_plot.png'.\n"
            "4. Provide a clear summary of your findings."
        ))
        messages = [sys_msg] + messages
    
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
    last_message = state['messages'][-1]
    return "tools" if last_message.tool_calls else "__end__"

# Define the Graph
workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", ToolNode(tools))
workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", should_continue)
workflow.add_edge("tools", "agent")

# --- ENDPOINTS ---

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        file_id = str(uuid.uuid4())[:8]
        # Use absolute path for Docker consistency
        file_name = f"{file_id}_{file.filename}"
        full_path = os.path.join(SANDBOX_DIR, file_name)
        
        with open(full_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        return {"file_path": full_path}
    except Exception as e:
        logger.error(f"Upload error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/chat")
async def chat(message: str = Form(...), thread_id: str = Form(...), file_path: str = Form(...)):
    try:
        async with app.state.pool.connection() as conn:
            checkpointer = AsyncPostgresSaver(conn)
            graph = workflow.compile(checkpointer=checkpointer)
            
            config = {"configurable": {"thread_id": thread_id}}
            inputs = {
                "messages": [HumanMessage(content=message)], 
                "file_path": file_path
            }
            
            output = await graph.ainvoke(inputs, config)
            last_msg = output["messages"][-1].content
            
            # Post-process plots: move from temp to unique filename
            plot_url = None
            temp_plot = os.path.join(SANDBOX_DIR, "temp_plot.png")
            if os.path.exists(temp_plot):
                unique_name = f"plot_{thread_id}_{uuid.uuid4().hex[:6]}.png"
                shutil.move(temp_plot, os.path.join(SANDBOX_DIR, unique_name))
                plot_url = f"/plots/{unique_name}"

            return {"response": last_msg, "plot_url": plot_url}
    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/plots/{plot_name}")
async def get_plot(plot_name: str):
    full_path = os.path.join(SANDBOX_DIR, plot_name)
    if os.path.exists(full_path):
        return FileResponse(full_path)
    return JSONResponse(status_code=404, content={"error": "Plot not found"})