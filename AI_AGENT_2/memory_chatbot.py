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

    response = llm.invoke(input=state["messages"])

    state["messages"].append(AIMessage(content=response.content))

    print(f"AI: {response.content}")

    return state

graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)

agent = graph.compile()

convo_history = []


# the entire conversation history is passed to the model
user_input = input("\nEnter your Question (bye to exit): ")
while user_input != "bye":
    convo_history.append(HumanMessage(content=user_input))
    result = agent.invoke({"messages": convo_history})
    convo_history = result["messages"]
    user_input = input("\nEnter your Question (bye to exit): ")

with open("./AI_AGENT_2/memory_chatbot_history.txt", "w") as f:
    for msg in convo_history:
        if isinstance(msg, HumanMessage):
            f.write(f"Human: {msg.content}\n")
        elif isinstance(msg, AIMessage):
            f.write(f"AI: {msg.content}")
    
print(f"Chatbot history saved to memory_chatbot_history.txt")
