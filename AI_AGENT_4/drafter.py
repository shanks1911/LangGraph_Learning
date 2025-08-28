from typing import TypedDict, Annotated, Sequence
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from langchain_core.tools import tool
from pathlib import Path

load_dotenv()

# global variable for human ai collaboration, usually we use langraph injected state
document_content = ""

class AgentState(TypedDict):
    messages : Annotated[Sequence[BaseMessage], add_messages]

@tool
def update(content:str) -> str:
    """Updates document with content provided by the LLM"""
    global document_content
    document_content = content
    return f"Document has been updated successfully.\nCurrent Document content:\n{document_content}"

@tool
def save_tool(filename: str)-> str:
    """
    Saves the curent document to the text file and finishes the process.
    Args:
        filename: Name of the text file.
    """

    global document_content

    if not filename.endswith(".txt"):
        filename = f"{filename}.txt"

    # --- Start of new code ---
    # Get the directory where the script is located
    script_directory = Path(__file__).parent.resolve()
    # Create the full path for the new file
    full_path = script_directory / filename
    # --- End of new code ---
    
    # Allows you to save the contents in the global variable as a text file

    try:
        with open(full_path, "w") as file:
            file.write(document_content)
        print(f"The Document has been saved to {filename}")
        return f"Document has been saved successfully to '{filename}'."
    
    except Exception as e:
        return f"Error saving document: {str(e)}"
    
tools = [update, save_tool]

model = ChatGoogleGenerativeAI(
    model = 'gemini-2.5-flash'
).bind_tools(tools)

def agent(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(content=f"""
    You are Drafter, a helpful writing assistant. You are going to help the user update and modify documents.
    
    - If the user wants to update or modify content, use the 'update' tool with the complete updated content.
    - If the user wants to save and finish, you need to use the 'save_tool' tool.
    - Make sure to always show the current document state after modifications.
    
    The current document content is:{document_content}
    """)

    if not state["messages"]:
        user_input = "I am ready to help you update the document. What would you like to create ?\n"
        user_message = HumanMessage(content=user_input)
    
    else:
        user_input = input("\nWhat would you like to do with the document ?\n")
        print(f"\n👤 USER: {user_input}")
        user_message = HumanMessage(content=user_input)
    
    all_messages = [system_prompt] + list(state["messages"]) + [user_message]

    response = model.invoke(all_messages)

    print(f"\n🤖 AI: {response.content}")
    
    if hasattr(response, "tool_calls") and response.tool_calls:
        print(f"🔧 USING TOOLS: {[tc['name'] for tc in response.tool_calls]}")

    return {"messages": list(state["messages"]) + [user_message, response]}

def should_continue(state: AgentState) -> AgentState:
    """Determine whether to continue this conversation or to end it"""

    messages = state["messages"]

    if not messages:
        return "continue"
    
    # to look if the recent tool message is from save_tools tool

    for message in reversed(messages):
        if (isinstance(message, ToolMessage) and
        "saved" in message.content.lower() and
        "document" in message.content.lower()):
            
            return "end"
        
    return "continue"

def print_messages(messages):
    """Function I made to print the messages in a more readable format"""
    if not messages:
        return
    
    for message in messages[-3:]:
        if isinstance(message, ToolMessage):
            print(f"\n⚙️ TOOL RESULT: {message.content}")

graph = StateGraph(AgentState)
graph.add_node("agent", agent)
graph.add_node("tools", ToolNode(tools))

graph.set_entry_point("agent")

graph.add_edge("agent", "tools")
graph.add_conditional_edges(
    "tools",
    should_continue,
    # Pathmap
    {
        "continue": "agent",
        "end": END
    }
)

app = graph.compile()

def run_document_agent():
    print("\n ===== DRAFTER =====")
    
    state = {"messages": []}
    
    for step in app.stream(state, stream_mode="values"):
        if "messages" in step:
            print_messages(step["messages"])
    
    print("\n ===== DRAFTER FINISHED =====")

if __name__ == "__main__":
    run_document_agent()