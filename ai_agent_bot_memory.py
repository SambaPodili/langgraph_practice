
# Simple Bot

#Objectives:
# Define state structure with a list of HumanMessage objects.
# Initialize a GPT-4o model using LangChain's ChatOpenAI
# Sending and handling different types of messages
# Buidling and compiling the graph of the Agent

#Main Goal: How to integrate LLM's in out Graphs


from typing import TypedDict, List, Union
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END # Framework that helps to design and manage the flow o teh tasks in application using a StateGraph
from dotenv import load_dotenv


load_dotenv()

class AgentState(TypedDict):
    messages: List[Union[HumanMessage,AIMessage]]
    
llm =  ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=0)


def process(state: AgentState) -> AgentState:
    response = llm.invoke(state["messages"])
    state["messages"].append(AIMessage(content=response.content))
    print(f"\nAI: {response.content}")
    print("current_state:", state["messages"])
    return state
    
graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)
agent = graph.compile()

history = []

user_input  = input("Enter: ")

while user_input != "exit":
    history.append(HumanMessage(content=user_input))
    result = agent.invoke({"messages": history})
    history = result["messages"]
    user_input  = input("Enter: ")