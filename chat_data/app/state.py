from typing import Annotated, Sequence, TypedDict, Optional
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    # This must be named 'messages' to work with the logic in main.py
    messages: Annotated[Sequence[BaseMessage], add_messages]
    
    # Metadata about the file we are working with
    file_path: str
    
    # Optional fields for advanced tracking
    last_code_generated: Optional[str]
    analysis_complete: Optional[bool]