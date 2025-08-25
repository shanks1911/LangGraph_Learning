import os
from dotenv import load_dotenv


from langchain_google_genai import ChatGoogleGenerativeAI
from typing import TypedDict, List
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END

load_dotenv()

class AgentState(TypedDict):
    messages: List[HumanMessage]
    
llm = ChatGoogleGenerativeAI(
    model = "gemini-2.5-flash"
)

def process(state: AgentState) -> AgentState:
    response = llm.invoke(input = state["messages"])
    print(f"\nAI: {response.content}")

    return state

graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process",END)

agent = graph.compile()

user_input = input("Enter your Question: ")
while user_input != "bye":
    agent.invoke({
        "messages": [HumanMessage(content=user_input)]
    })
    user_input = input("\nEnter your Question (bye to exit) :")







# os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
# llm = ChatGoogleGenerativeAI(
#     model="gemini-2.5-flash",
#     temperature=0,
#     max_tokens=None,
#     timeout=None,
#     max_retries=2,
# )

# messages = [
#     (
#         "system",
#         "You are a helpful assistant that translates English to French. Translate the user sentence.",
#     ),
#     ("human", "I love programming."),
# ]
# ai_msg = llm.invoke(messages)
# print(ai_msg.content)