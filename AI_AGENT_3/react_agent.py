# it can reason and use tools
# tools, react graphs, different message types like ToolMessages, test robustness of graph

# main goal: create a robust react agent

from typing import TypedDict, Annotated, Sequence
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from langchain_core.tools import tool

load_dotenv()

# annottated: provides additional context without affecting the type itself
# email: Annotated[str, "user's email address"]

# print(email.__metadata__)  # Output: "user's email address"

# Sequence - to automatically handle the state updates for sequences such as by adding new messages to chat history

# ToolMessage - passes the data back to LLM after it calls a tool such as content and tool_call_id

# SystemMessage - to provide instruction to the LLM

# BaseMessage - foundational class for all message types in langgraph

# add_messages - it is a reducer function
# reducer function is a rule that controls how updates from nodes are combined with the existing states
# it tells us how to merge new data into current state
# without a reducer function, updates would have replace the existing contents entirely
# it allows us to append everything to the state

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# decorator - tells python that this function is special

@tool 
def add(a: int, b: int):
    """Adds two numbers together."""
    # docstring is needed as it tells the LLM what the tool is for
    return a+b

@tool
def subtract(a: int, b: int):
    """Subtracts two numbers."""
    return a-b

@tool
def multiply(a: int, b: int):
    """Multiplies two numbers."""
    return a*b

@tool
def divide(a: int, b: int):
    """Divides two numbers."""
    if b == 0:
        return "Error: Division by zero is not allowed."
    return a/b

tools = [add, subtract, multiply, divide]

model = ChatGoogleGenerativeAI(
    model = "gemini-2.5-flash"
).bind_tools(tools)

def model_call(state:AgentState) -> AgentState:
    system_prompt = SystemMessage(
        content="You are my AI assistant, please answer my query to the best of your ability."
    )
    response = model.invoke([system_prompt] + state["messages"])
    return {"messages":[response]}

def should_continue(state:AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"
    
graph = StateGraph(AgentState)
graph.add_node("our_agent", model_call)

tool_node = ToolNode(tools=tools)
graph.add_node("tools", tool_node)

graph.set_entry_point("our_agent")

graph.add_conditional_edges(
    "our_agent",
    should_continue,
    {
        "continue": "tools",
        "end": END
    }
)

graph.add_edge("tools", "our_agent")

app = graph.compile()

def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

user_input = input("Enter your Question: ")
inputs = {
    "messages": [("user", user_input)]
}

print_stream(app.stream(inputs, stream_mode="values"))