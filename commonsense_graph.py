# commonsense_graph.py
from typing import Dict, Any, List
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda

from fewshot import CATEGORY_KEYWORDS, EXAMPLES
from commonsense_agent import route_category, select_examples, build_messages, call_deepseek, extract_json

# --- Graph state schema ---
# Use simple dict instead of custom class for better LangGraph compatibility
State = Dict[str, Any]

# --- Nodes ---
def node_classify(state: State) -> State:
    # Ensure query exists
    if "query" not in state or not state["query"]:
        state["query"] = "7-day UK trip for a couple visiting London, Manchester, Edinburgh by train. Avoid overnight."
    
    print(f"DEBUG: Query = {state['query']}")
    cats = route_category(state["query"])
    print(f"DEBUG: Categories = {cats}")
    
    # Update state in place (LangGraph expects this)
    state["categories"] = cats
    print(f"DEBUG: State categories = {state.get('categories', 'NOT SET')}")
    return state

def node_pick_fewshots(state: State) -> State:
    print(f"DEBUG fewshots: State keys = {list(state.keys())}")
    print(f"DEBUG fewshots: Categories = {state.get('categories', 'NOT FOUND')}")
    print(f"DEBUG fewshots: Query = {state.get('query', 'NOT FOUND')}")
    
    # Update state in place
    state["fewshots"] = select_examples(state["categories"], k_per_cat=1)
    return state

def node_build_messages(state: State) -> State:
    state["messages"] = build_messages(state["query"], state["fewshots"])
    return state

def node_generate(state: State) -> State:
    state["raw"] = call_deepseek(state["messages"], max_new_tokens=2048)
    return state

def node_parse(state: State) -> State:
    try:
        raw = state["raw"]
        print(f"DEBUG: Raw output length: {len(raw)}")
        print(f"DEBUG: Raw output last 20 chars: {repr(raw[-20:])}")
        print(f"DEBUG: Raw output first 20 chars: {repr(raw[:20])}")
        
        # Check for hidden characters
        for i, char in enumerate(raw):
            if not char.isprintable() and char not in ['\n', '\t', ' ']:
                print(f"DEBUG: Non-printable char at position {i}: {repr(char)} (ord: {ord(char)})")
        
        state["policy"] = extract_json(raw)
    except Exception as e:
        state["error"] = f"JSON parse failed: {e}"
        print(f"DEBUG: Error details: {e}")
        print(f"DEBUG: Raw output: {repr(raw)}")
    return state

# --- Assemble graph ---
def build_graph():
    g = StateGraph(State)
    g.add_node("classify", RunnableLambda(node_classify))
    g.add_node("fewshots", RunnableLambda(node_pick_fewshots))
    g.add_node("messages", RunnableLambda(node_build_messages))
    g.add_node("generate", RunnableLambda(node_generate))
    g.add_node("parse", RunnableLambda(node_parse))

    g.set_entry_point("classify")
    g.add_edge("classify", "fewshots")
    g.add_edge("fewshots", "messages")
    g.add_edge("messages", "generate")
    g.add_edge("generate", "parse")
    g.add_edge("parse", END)
    return g.compile()

if __name__ == "__main__":
    graph = build_graph()
    init = {"query": "7-day UK trip for a couple visiting London, Manchester, Edinburgh by train. Avoid overnight."}
    result: State = graph.invoke(init)
    if "error" in result:
        print(result["error"])
        print("RAW:", result.get("raw",""))
    else:
        import json
        print(json.dumps(result["policy"], indent=2, ensure_ascii=False))
