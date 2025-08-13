# tool_graph.py
from typing import Dict, Any
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda

from tool_agent import run_tool_agent

# Use simple dict instead of custom class for better LangGraph compatibility
State = Dict[str, Any]

def node_tools(state: State) -> State:
    try:
        result = run_tool_agent(state["policy"])
        if result is not None:
            state["bundle"] = result
        else:
            state["error"] = "ToolAgent returned None - check Google Sheets API access"
    except Exception as e:
        state["error"] = f"ToolAgent failed: {e}"
    return state

def build_graph():
    g = StateGraph(State)
    g.add_node("tools", RunnableLambda(node_tools))
    g.set_entry_point("tools")
    g.add_edge("tools", END)
    return g.compile()

if __name__ == "__main__":
    graph = build_graph()
    fake_policy = {"city_sequence": ["London","Manchester","Edinburgh"]}
    res: State = graph.invoke(dict(policy=fake_policy))
    
    if res is None:
        print("Error: Graph returned None")
    elif "error" in res:
        print("Error:", res["error"])
    else:
        b = res.get("bundle")
        if b is None:
            print("Error: No bundle in result")
        else:
            print("Cities:", b["cities"])
            print("POIs:", len(b["pois"]), "Hotels:", len(b["hotels"]), "Restaurants:", len(b["restaurants"]))
