from typing import Annotated, Sequence, TypedDict, Optional
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict):

    messages: Annotated[Sequence[BaseMessage], add_messages]
    

    file_path: str
    

    last_code_generated: Optional[str]
    analysis_complete: Optional[bool]