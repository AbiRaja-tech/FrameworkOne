# tool_graph.py
from typing import Dict, Any
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda

from tool_agent import run_tool_agent

State = Dict[str, Any]

def node_tools(state: State) -> State:
    try:
        # accept either 'policy' or 'policy_raw' from earlier stage
        policy = state.get("policy") or state.get("policy_raw") or {}
        res = run_tool_agent(policy)
        state["bundle"] = res
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
    res: State = graph.invoke({"policy": {"city_sequence": ["London"]}})
    if "error" in res:
        print("Error:", res["error"])
    else:
        b = res["bundle"]
        print("Cities:", b["cities"])
        print("POIs:", len(b["pois"]), "Hotels:", len(b["hotels"]), "Restaurants:", len(b["restaurants"]))
