#  Main goal is to create a chatbot with memory
# create a form of memory for our agent

import os
from dotenv import load_dotenv

from typing import TypedDict, List, Union
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

class AgentState(TypedDict):
    # messages: List[HumanMessage]
    # messages_ai: List[AIMessage] # Naive Approach

    messages: List[Union[HumanMessage, AIMessage]] 

llm = ChatGoogleGenerativeAI(
    model = "gemini-2.5-flash"
)

def process(state: AgentState) -> AgentState:
    """This node will solve the request that you input"""