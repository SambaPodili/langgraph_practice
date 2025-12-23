
#Objectives:
# How to create Tools in LangGraph
# How to createa ReAct Graph
# Work with different types of Messages such as ToolMessages
# Test robustness of graph

#Main Goal: Createa robust ReAct Agent


from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage
from langchain_core.messages import ToolMessage
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
#from langchain_google_genai import ChatGoogleGenerativeAI
#from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, END # Framework that helps to design and manage the flow o teh tasks in application using a StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv


load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

@tool
def add(a: int, b:int):
    """This is an addition function that adds two numbers"""
    return a + b

@tool
def subtract(a: int, b:int):
    """This is an subtract function that subtract two numbers"""
    return a - b

@tool
def multiply(a: int, b:int):
    """This is an multiply function that multiply two numbers"""
    return a + b

tools = [add, subtract, multiply]


#model =  ChatGoogleGenerativeAI(model="gemini-3-flash-preview").bind_tools(tools)
#model =  ChatOpenAI(model="gpt-4o").bind_tools(tools)
model =  ChatAnthropic(model="claude-opus-4-5-20251101").bind_tools(tools)

def model_call(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(content = "You are my AI assistant, please answer my query to the best of your ability.")
    response = model.invoke([system_prompt] + state["messages"])
    return {"messages": [response]}


def should_continue(state: AgentState):
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
agent = graph.compile()


def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()
            
#inputs = {"messages": [("user", "Add 34+21. Add 3 + 4. Ass 15+15")]}
inputs = {"messages": [("user", "Add 40 + 12 and then multiply the result by 6.Also tell me a joke please.")]}
print_stream(agent.stream(inputs, stream_mode="values"))