# orchestrator.py
"""
End-to-end LangGraph pipeline:
User query -> commonsense policy -> tool bundle (POIs, hotels, dist) -> CP-SAT per-city -> results.

Usage examples:
  # quick run (uses default query + policy_budget_first.json)
  python orchestrator.py

  # custom query, stamina-first policy, verbose CP logs
  LOG_LEVEL=INFO python orchestrator.py \
      --query "5-day London trip for a solo traveler who loves museums and parks" \
      --policy policy_stamina_first.json --cp-verbose
"""

import os, json, argparse
from typing import Dict, Any, List, Tuple
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda

# your existing modules
from commonsense_graph import build_graph as build_commonsense_graph
from tool_graph import build_graph as build_tool_graph
from intracity_planner import load_policy as load_cp_policy, solve_intracity_city_day
from adapter import bundle_city_to_planner_data

# -----------------
# Shared graph state
# -----------------
# Use simple dict instead of custom class for better LangGraph compatibility
State = Dict[str, Any]

# -----------------
# Nodes
# -----------------
def node_commonsense(state: State) -> State:
    # Run your commonsense graph (classification -> fewshot -> messages -> generate -> parse)
    graph = build_commonsense_graph()
    out = graph.invoke({"query": state.get("query","")})
    if "error" in out:
        state["error"] = out["error"]
        state["raw"] = out.get("raw","")
        return state
    state["policy_raw"] = out["policy"]
    return state


def node_tools(state: State) -> State:
    # Run your tool agent graph (Google Sheets + Haversine matrix)
    g = build_tool_graph()
    out = g.invoke({"policy": state.get("policy_raw",{})})
    if "error" in out:
        state["error"] = out["error"]
        return state
    state["bundle"] = out["bundle"]
    return state


def node_choose_cp_policy(state: State, policy_path: str) -> State:
    # Load/merge a CP-SAT policy (budget-first or stamina-first etc.)
    # If user didn’t pass one, default to budget-first.
    if not policy_path:
        policy_path = "policy_budget_first.json"
    try:
        state["planner_policy"] = load_cp_policy(policy_path)
    except Exception as e:
        state["error"] = f"Failed to load CP policy '{policy_path}': {e}"
    return state


def node_planner(state: State, cp_verbose: bool = False) -> State:
    bundle = state.get("bundle") or {}
    cities: List[str] = bundle.get("cities", [])
    if not cities:
        state["error"] = "No cities from Tool Agent."
        return state

    cp_policy = state.get("planner_policy") or load_cp_policy(None)
    per_city: Dict[str, Any] = {}
    best: Tuple[str, Any] = ("", None)

    for city in cities:
        data_for_planner = bundle_city_to_planner_data(bundle, city)
        if not data_for_planner["pois"] or not data_for_planner["hotels"]:
            per_city[city] = {"status":"infeasible", "reason":"no POIs or hotels after adaptation"}
            continue

        res = solve_intracity_city_day(data_for_planner, cp_policy, cp_verbose=cp_verbose)
        per_city[city] = res
        if res.get("status") == "ok":
            if best[1] is None or res["best"]["objective"] > best[1]["best"]["objective"]:
                best = (city, res)

    state["per_city"] = per_city
    if best[1]:
        state["best"] = {"city": best[0], "result": best[1]["best"]}
    else:
        state["best"] = None
    return state

# ---------------
# Build big graph
# ---------------
def build_orchestrator(policy_path: str = "", cp_verbose: bool = False):
    g = StateGraph(State)

    # wrap simple nodes with RunnableLambda; for those with args, use lambda/closure
    g.add_node("commonsense", RunnableLambda(node_commonsense))
    g.add_node("tools", RunnableLambda(node_tools))
    g.add_node("choose_cp_policy", RunnableLambda(lambda s: node_choose_cp_policy(s, policy_path)))
    g.add_node("planner", RunnableLambda(lambda s: node_planner(s, cp_verbose)))

    g.set_entry_point("commonsense")
    g.add_edge("commonsense", "tools")
    g.add_edge("tools", "choose_cp_policy")
    g.add_edge("choose_cp_policy", "planner")
    g.add_edge("planner", END)
    return g.compile()

# ---------------
# CLI
# ---------------
def main():
    ap = argparse.ArgumentParser(description="End-to-end trip orchestration (commonsense → tools → CP-SAT)")
    ap.add_argument("--query", type=str, default="4-day London trip for a single 32-year-old male, loves history and museums, medium budget, prefers walking and Tube. Only intracity travel.",
                    help="User trip brief.")
    ap.add_argument("--policy", type=str, default="policy_budget_first.json",
                    help="CP-SAT policy JSON (e.g., policy_budget_first.json, policy_stamina_first.json)")
    ap.add_argument("--cp-verbose", action="store_true", help="Show OR-Tools internal logs")
    args = ap.parse_args()

    graph = build_orchestrator(policy_path=args.policy, cp_verbose=args.cp_verbose)
    result: State = graph.invoke(dict(query=args.query))

    if "error" in result and result["error"]:
        print("ERROR:", result["error"])
        if "raw" in result:
            print("RAW:", result["raw"])
        return

    # Pretty summary
    print("\n=== COMMONSENSE POLICY (raw) ===")
    print(json.dumps(result["policy_raw"], indent=2, ensure_ascii=False))

    print("\n=== TOOL BUNDLE SUMMARY ===")
    b = result["bundle"]
    print("Cities:", b.get("cities"))
    print("POIs:", len(b.get("pois", [])), "Hotels:", len(b.get("hotels", [])))

    if result.get("best"):
        print("\n=== BEST DAY (by objective) ===")
        print("City:", result["best"]["city"])
        print(json.dumps(result["best"]["result"], indent=2, ensure_ascii=False))
    else:
        print("\nNo feasible plans found.")

    print("\n=== PER-CITY RESULTS ===")
    print(json.dumps(result.get("per_city", {}), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
