# orchestrator.py
"""
End-to-end LangGraph pipeline:
User query -> commonsense policy -> tool bundle (POIs, hotels, dist) -> CP-SAT per-city -> results.

What's new in this version
--------------------------
1) Policy Synthesis middle layer:
   - We synthesize a planner-ready policy from the commonsense 'policy_raw' plus an optional
     template JSON (e.g., policy_budget_first.json / policy_stamina_first.json).
   - Safe lexicographic weights are generated to avoid CP-SAT overflow/flattening.

2) Data sanitization:
   - We sanitize the Tool Agent bundle (trim city keys, fix/derive names, drop invalid hotels)
     before adapting it for the CP-SAT solver.

Usage examples
--------------
# Default: uses commonsense -> tools -> synthesize using policy_budget_first.json template
python orchestrator.py

# Stamina-first: synthesize directly without a template
python orchestrator.py --policy synth:stamina_first

# Load a JSON file *directly* (no synthesis) as the CP policy (legacy mode)
python orchestrator.py --policy path/to/custom_cp_policy.json

# With verbose CP-SAT internal logs:
LOG_LEVEL=INFO python orchestrator.py --cp-verbose
"""

import os
import json
import argparse
import logging
from typing import Dict, Any, List, Tuple

from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda

# Middle-layer policy builder
from policy_synthesizer import synthesize_policy

# Your existing modules
from commonsense_graph import build_graph as build_commonsense_graph
from tool_graph import build_graph as build_tool_graph

# Planner (CP-SAT)
from intracity_planner import load_policy as load_cp_policy, solve_intracity_city_day

# Adapter helpers (now includes sanitization)
from adapter import bundle_city_to_planner_data, sanitize_bundle

# Setup logging
def setup_orchestrator_logging(log_file: str = "orchestrator.log"):
    """Setup logging to both console and file"""
    log = logging.getLogger("orchestrator")
    log.setLevel(logging.INFO)
    
    # Clear existing handlers
    log.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("[%(levelname)s] %(message)s")
    console_handler.setFormatter(console_formatter)
    log.addHandler(console_handler)
    
    # File handler
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    file_handler.setFormatter(file_formatter)
    log.addHandler(file_handler)
    
    return log

# Initialize logging
log = setup_orchestrator_logging()


# -----------------
# Shared graph state
# -----------------
State = Dict[str, Any]


# -----------------
# Helpers
# -----------------
def _file_exists(path: str) -> bool:
    return bool(path) and os.path.exists(path) and os.path.isfile(path)


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _should_direct_synthesize(spec: str) -> Tuple[bool, str]:
    """
    Recognize direct synth spec: 'synth:budget_first' or 'synth:stamina_first'.
    Returns (is_direct_synth, profile).
    """
    if not spec:
        return False, ""
    s = spec.strip().lower()
    if s in ("synth:budget_first", "synthesize:budget_first", "synth:budget", "synthesize:budget"):
        return True, "budget_first"
    if s in ("synth:stamina_first", "synthesize:stamina_first", "synth:stamina", "synthesize:stamina"):
        return True, "stamina_first"
    return False, ""


# -----------------
# Nodes
# -----------------
def node_commonsense(state: State) -> State:
    """Run the commonsense agent graph to produce a high-level 'policy_raw'."""
    graph = build_commonsense_graph()
    out = graph.invoke({"query": state.get("query", "")})
    if "error" in out:
        state["error"] = out["error"]
        state["raw"] = out.get("raw", "")
        return state
    state["policy_raw"] = out.get("policy", {}) or {}
    return state


def node_tools(state: State) -> State:
    """Run the tool agent graph to fetch POIs, hotels, distances, etc., then sanitize the bundle."""
    g = build_tool_graph()
    out = g.invoke({"policy": state.get("policy_raw", {})})
    if "error" in out:
        state["error"] = out["error"]
        return state
    raw_bundle = out.get("bundle", {}) or {}
    state["bundle"] = sanitize_bundle(raw_bundle)
    return state


def node_choose_cp_policy(state: State, policy_spec: str) -> State:
    """
    Build the planner policy. Three modes:

    A) Template-guided synth (recommended/default):
       - If 'policy_spec' is a JSON file path, we read it as a template and synthesize a policy
         from state['policy_raw'], inheriting template fields. If the template has 'priority_order',
         it will be respected.

    B) Direct synth (no template):
       - If 'policy_spec' looks like 'synth:budget_first' or 'synth:stamina_first', we synthesize
         purely from policy_raw and defaults.

    C) Legacy load:
       - If neither A nor B applies, we attempt to load the CP policy via load_cp_policy (no synthesis).
    """
    policy_raw = state.get("policy_raw", {}) or {}

    # B) Direct synth?
    is_direct_synth, profile = _should_direct_synthesize(policy_spec)
    if is_direct_synth:
        try:
            state["planner_policy"] = synthesize_policy(policy_raw, profile=profile, templates=None)
        except Exception as e:
            state["error"] = f"Direct policy synth failed ({profile}): {e}"
        return state

    # A) Template-guided synth if JSON exists
    if _file_exists(policy_spec):
        try:
            template = _read_json(policy_spec)
        except Exception as e:
            state["error"] = f"Failed to read policy template '{policy_spec}': {e}"
            return state

        # If template carries a 'priority_order', pass it through policy_raw so synthesizer respects it
        policy_raw_for_synth = dict(policy_raw)
        if isinstance(template, dict) and "priority_order" in template:
            policy_raw_for_synth["priority_order"] = template["priority_order"]

        try:
            state["planner_policy"] = synthesize_policy(policy_raw_for_synth, profile=None, templates=template)
        except Exception as e:
            state["error"] = f"Template-guided policy synth failed: {e}"
        return state

    # C) Legacy load (no synthesis) — keep for compatibility
    try:
        state["planner_policy"] = load_cp_policy(policy_spec if policy_spec else None)
    except Exception as e:
        state["error"] = f"Failed to load CP policy '{policy_spec}': {e}"
    return state


def node_planner(state: State, cp_verbose: bool = False) -> State:
    bundle = state.get("bundle") or {}
    cities: List[str] = bundle.get("cities", []) or []
    if not cities:
        state["error"] = "No cities from Tool Agent."
        return state

    cp_policy = state.get("planner_policy") or load_cp_policy(None)

    per_city: Dict[str, Any] = {}
    best_city: Tuple[str, Any] = ("", None)

    for city in cities:
        data_for_planner = bundle_city_to_planner_data(bundle, city)
        if not data_for_planner.get("pois") or not data_for_planner.get("hotels"):
            per_city[city] = {"status": "infeasible", "reason": "no POIs or hotels after adaptation"}
            continue

        res = solve_intracity_city_day(data_for_planner, cp_policy, cp_verbose=cp_verbose)
        per_city[city] = res

        if res.get("status") == "ok":
            # choose the best by objective value
            if best_city[1] is None or res["best"]["objective"] > best_city[1]["best"]["objective"]:
                best_city = (city, res)

    state["per_city"] = per_city
    state["best"] = {"city": best_city[0], "result": best_city[1]["best"]} if best_city[1] else None
    return state


# -----------------
# Output to JSON file
# -----------------
def save_results_to_json(result: State, output_file: str = "orchestrator_results.json"):
    """Save the complete orchestrator results to a JSON file."""
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=str)
        log.info(f"Results successfully saved to {output_file}")
        print(f"\n=== RESULTS SAVED TO {output_file} ===")
    except Exception as e:
        log.error(f"Failed to save results to {output_file}: {e}")
        print(f"Warning: Failed to save results to {output_file}: {e}")


# ---------------
# Build big graph
# ---------------
def build_orchestrator(policy_spec: str = "", cp_verbose: bool = False):
    g = StateGraph(State)

    # wrap simple nodes with RunnableLambda; for those with args, use lambda/closure
    g.add_node("commonsense", RunnableLambda(node_commonsense))
    g.add_node("tools", RunnableLambda(node_tools))
    g.add_node("choose_cp_policy", RunnableLambda(lambda s: node_choose_cp_policy(s, policy_spec)))
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
    ap.add_argument(
        "--query",
        type=str,
        default="4-day London trip for a single 32-year-old male, loves history and museums, medium budget, prefers walking and Tube. Only intracity travel.",
        help="User trip brief.",
    )
    ap.add_argument(
        "--policy",
        type=str,
        default="policy_budget_first.json",
        help=(
            "Policy selection:\n"
            "  1) Template-guided synth (recommended): path to a template JSON, e.g. policy_budget_first.json\n"
            "  2) Direct synth (no template): 'synth:budget_first' or 'synth:stamina_first'\n"
            "  3) Legacy load (no synthesis): path to a full CP policy JSON"
        ),
    )
    ap.add_argument("--cp-verbose", action="store_true", help="Show OR-Tools internal logs")
    ap.add_argument("--output", type=str, default="orchestrator_results.json", help="Output JSON file path")
    args = ap.parse_args()
    
    # Log startup information
    log.info(f"=== ORCHESTRATOR STARTUP ===")
    log.info(f"Query: {args.query}")
    log.info(f"Policy spec: {args.policy}")
    log.info(f"CP verbose: {args.cp_verbose}")
    log.info(f"Output file: {args.output}")

    graph = build_orchestrator(policy_spec=args.policy, cp_verbose=args.cp_verbose)
    result: State = graph.invoke(dict(query=args.query))

    if "error" in result and result["error"]:
        log.error(f"Orchestrator failed with error: {result['error']}")
        if "raw" in result:
            log.error(f"Raw error details: {result['raw']}")
        print("ERROR:", result["error"])
        if "raw" in result:
            print("RAW:", result["raw"])
        save_results_to_json(result, args.output)
        return

    log.info(f"=== ORCHESTRATOR RESULTS ===")
    log.info(f"Commonsense policy generated: {len(result.get('policy_raw', {}))} keys")
    log.info(f"Tool bundle created: {len(result.get('bundle', {}))} cities, {len(result.get('pois', []))} POIs, {len(result.get('hotels', []))} hotels")
    
    print("\n=== COMMONSENSE POLICY (raw) ===")
    print(json.dumps(result.get("policy_raw", {}), indent=2, ensure_ascii=False))

    print("\n=== TOOL BUNDLE SUMMARY (sanitized) ===")
    b = result.get("bundle", {})
    print("Cities:", b.get("cities"))
    print("POIs:", len(b.get("pois", [])), "Hotels:", len(b.get("hotels", [])))

    # print("\n=== SYNTHESIZED / LOADED CP POLICY (keys) ===")
    # cp_pol = result.get("planner_policy", {})
    # print(json.dumps(
    #     {k: (cp_pol[k] if k in ("priority_order", "max_pois_per_day", "min_pois_per_day") else "…")
    #      for k in cp_pol.keys()}, indent=2, ensure_ascii=False))

    print("\n=== SYNTHESIZED / LOADED CP POLICY (full) ===")
    print(json.dumps(result.get("planner_policy", {}), indent=2, ensure_ascii=False))

    if result.get("best"):
        log.info(f"Best plan found for city: {result['best']['city']}")
        log.info(f"Best plan objective: {result['best']['result'].get('objective', 'N/A')}")
        print("\n=== BEST DAY (by objective) ===")
        print("City:", result["best"]["city"])
        print(json.dumps(result["best"]["result"], indent=2, ensure_ascii=False))
    else:
        log.warning("No feasible plans found")
        print("\nNo feasible plans found.")

    print("\n=== PER-CITY RESULTS ===")
    print(json.dumps(result.get("per_city", {}), indent=2, ensure_ascii=False))

    save_results_to_json(result, args.output)
    log.info(f"Results saved to {args.output}")
    log.info(f"=== ORCHESTRATOR COMPLETED SUCCESSFULLY ===")


if __name__ == "__main__":
    main()