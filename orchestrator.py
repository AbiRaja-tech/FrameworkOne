# orchestrator.py
"""
End-to-end LangGraph pipeline with clear sequential flow:
1. User Query → 2. Commonsense Agent → 3. Tool Agent → 4. Policy Generator (DeepSeek) → 5. Planner → 6. Results

What's new in this version
--------------------------
1) Clear sequential flow with explicit stages
2) Enhanced logging at each stage showing input/output
3) DeepSeek-powered policy generation
4) Policy synthesis middle layer with template support
5) Data sanitization and CP-SAT planning

Usage examples
--------------
# Default: uses commonsense -> tools -> DeepSeek policy generation -> CP-SAT planning
python orchestrator.py

# With specific template for policy generation
python orchestrator.py --policy policy_budget_first.json

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

# Middle-layer policy builder with DeepSeek integration
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
# Stage 1: Commonsense Agent
# -----------------
def node_commonsense(state: State) -> State:
    """Stage 1: Run the commonsense agent to understand user intent and generate high-level policy."""
    print("\n" + "="*80)
    print("🎯 STAGE 1: COMMONSENSE AGENT")
    print("="*80)
    
    query = state.get("query", "")
    print(f"📝 Input Query: {query}")
    
    graph = build_commonsense_graph()
    out = graph.invoke({"query": query})
    
    if "error" in out:
        error_msg = f"Commonsense agent failed: {out['error']}"
        print(f"❌ {error_msg}")
        log.error(error_msg)
        state["error"] = out["error"]
        state["raw"] = out.get("raw", "")
        return state
    
    policy_raw = out.get("policy", {}) or {}
    state["policy_raw"] = policy_raw
    
    print(f"✅ Commonsense Agent Success!")
    print(f"📊 Generated Policy Raw: {len(policy_raw)} keys")
    print(f"🔑 Key sections: {list(policy_raw.keys())}")
    print(f"📋 Policy Raw Content:")
    print(json.dumps(policy_raw, indent=2, ensure_ascii=False))
    
    log.info(f"Stage 1 completed: Commonsense agent generated {len(policy_raw)} policy keys")
    return state


# -----------------
# Stage 2: Tool Agent
# -----------------
def node_tools(state: State) -> State:
    """Stage 2: Run the tool agent to fetch POIs, hotels, distances, etc."""
    print("\n" + "="*80)
    print("🛠️  STAGE 2: TOOL AGENT")
    print("="*80)
    
    policy_raw = state.get("policy_raw", {})
    print(f"📥 Input: Policy Raw from Commonsense Agent")
    print(f"📊 Policy Raw Keys: {list(policy_raw.keys())}")
    
    g = build_tool_graph()
    out = g.invoke({"policy": policy_raw})
    
    if "error" in out:
        error_msg = f"Tool agent failed: {out['error']}"
        print(f"❌ {error_msg}")
        log.error(error_msg)
        state["error"] = out["error"]
        return state
    
    raw_bundle = out.get("bundle", {}) or {}
    print(f"📦 Raw Tool Bundle Received:")
    print(f"   - Cities: {raw_bundle.get('cities', [])}")
    print(f"   - POIs: {len(raw_bundle.get('pois', []))}")
    print(f"   - Hotels: {len(raw_bundle.get('hotels', []))}")
    print(f"   - Restaurants: {len(raw_bundle.get('restaurants', []))}")
    
    # Sanitize the bundle
    sanitized_bundle = sanitize_bundle(raw_bundle)
    state["bundle"] = sanitized_bundle
    
    print(f"✅ Tool Agent Success!")
    print(f"🧹 Sanitized Bundle:")
    print(f"   - Cities: {sanitized_bundle.get('cities', [])}")
    print(f"   - POIs: {len(sanitized_bundle.get('pois', []))}")
    print(f"   - Hotels: {len(sanitized_bundle.get('hotels', []))}")
    print(f"   - Restaurants: {len(sanitized_bundle.get('restaurants', []))}")
    
    log.info(f"Stage 2 completed: Tool agent fetched data for {len(sanitized_bundle.get('cities', []))} cities")
    return state


# -----------------
# Stage 3: Policy Generator (DeepSeek)
# -----------------
def node_policy_generator(state: State, policy_spec: str) -> State:
    """Stage 3: Generate personalized policy using DeepSeek and policy synthesis."""
    print("\n" + "="*80)
    print("🤖 STAGE 3: POLICY GENERATOR (DEEPSEEK)")
    print("="*80)
    
    policy_raw = state.get("policy_raw", {}) or {}
    bundle = state.get("bundle", {})
    
    print(f"📥 Inputs:")
    print(f"   - Policy Raw: {len(policy_raw)} keys from Commonsense Agent")
    print(f"   - Tool Bundle: {len(bundle.get('cities', []))} cities, {len(bundle.get('pois', []))} POIs")
    print(f"   - Policy Spec: {policy_spec}")
    
    # Check for direct synthesis
    is_direct_synth, profile = _should_direct_synthesize(policy_spec)
    if is_direct_synth:
        print(f"🎯 Direct Policy Synthesis: {profile}")
        try:
            state["planner_policy"] = synthesize_policy(policy_raw, profile=profile, templates=None)
            print(f"✅ Direct Policy Synthesis Success!")
        except Exception as e:
            error_msg = f"Direct policy synthesis failed ({profile}): {e}"
            print(f"❌ {error_msg}")
            log.error(error_msg)
            state["error"] = error_msg
            return state
    else:
        # Template-guided synthesis
        print(f"🔍 Checking for policy template: {policy_spec}")
        if _file_exists(policy_spec):
            print(f"✅ Template found, reading...")
            try:
                template = _read_json(policy_spec)
                print(f"📖 Template loaded: {len(template)} keys")
                print(f"🔑 Template sections: {list(template.keys())}")
                
                # Pass priority_order if template has it
                policy_raw_for_synth = dict(policy_raw)
                if "priority_order" in template:
                    policy_raw_for_synth["priority_order"] = template["priority_order"]
                    print(f"🎯 Template priority order: {template['priority_order']}")
                
                print(f"🔧 Starting DeepSeek-powered policy synthesis...")
                state["planner_policy"] = synthesize_policy(policy_raw_for_synth, profile=None, templates=template)
                print(f"✅ Template-guided Policy Synthesis Success!")
                
            except Exception as e:
                error_msg = f"Template-guided policy synthesis failed: {e}"
                print(f"❌ {error_msg}")
                log.error(error_msg)
                import traceback
                traceback.print_exc()
                state["error"] = error_msg
                return state
        else:
            error_msg = f"Policy template not found: {policy_spec}"
            print(f"❌ {error_msg}")
            log.error(error_msg)
            state["error"] = error_msg
            return state
    
    # Log the generated policy
    planner_policy = state.get("planner_policy", {})
    print(f"🎯 Generated Policy Summary:")
    print(f"   - Policy ID: {planner_policy.get('unique_id', 'unknown')}")
    print(f"   - Budget: {planner_policy.get('budget', {}).get('max_budget', 'N/A')}")
    print(f"   - Stamina: {planner_policy.get('stamina', {}).get('max_stamina', 'N/A')}")
    print(f"   - Max POIs: {planner_policy.get('planning', {}).get('max_pois_per_day', 'N/A')}")
    print(f"   - Max Distance: {planner_policy.get('planning', {}).get('max_distance_per_day', 'N/A')}")
    
    print(f"📋 Full Generated Policy:")
    print(json.dumps(planner_policy, indent=2, ensure_ascii=False))
    
    log.info(f"Stage 3 completed: Policy generator created policy with ID {planner_policy.get('unique_id', 'unknown')}")
    return state


# -----------------
# Stage 4: Planner (CP-SAT)
# -----------------
def node_planner(state: State, cp_verbose: bool = False) -> State:
    """Stage 4: Run the CP-SAT planner to generate travel plans."""
    print("\n" + "="*80)
    print("📊 STAGE 4: PLANNER (CP-SAT)")
    print("="*80)
    
    bundle = state.get("bundle") or {}
    cities: List[str] = bundle.get("cities", []) or []
    planner_policy = state.get("planner_policy")
    
    print(f"📥 Inputs:")
    print(f"   - Cities: {cities}")
    print(f"   - Policy: {planner_policy.get('unique_id', 'unknown') if planner_policy else 'None'}")
    print(f"   - CP Verbose: {cp_verbose}")
    
    if not cities:
        error_msg = "No cities from Tool Agent."
        print(f"❌ {error_msg}")
        log.error(error_msg)
        state["error"] = error_msg
        return state

    if not planner_policy:
        error_msg = "No policy generated. Policy generation must complete before planning."
        print(f"❌ {error_msg}")
        log.error(error_msg)
        state["error"] = error_msg
        return state

    print(f"🚀 Starting CP-SAT planning for {len(cities)} cities...")
    
    per_city: Dict[str, Any] = {}
    best_city: Tuple[str, Any] = ("", None)

    for city in cities:
        print(f"\n🏙️  Planning for city: {city}")
        data_for_planner = bundle_city_to_planner_data(bundle, city)
        
        if not data_for_planner.get("pois") or not data_for_planner.get("hotels"):
            print(f"   ⚠️  City {city}: No POIs or hotels after adaptation")
            per_city[city] = {"status": "infeasible", "reason": "no POIs or hotels after adaptation"}
            continue

        print(f"   📍 POIs: {len(data_for_planner.get('pois', []))}")
        print(f"   🏨 Hotels: {len(data_for_planner.get('hotels', []))}")
        print(f"   🚗 Starting CP-SAT solver...")
        
        res = solve_intracity_city_day(data_for_planner, planner_policy, cp_verbose=cp_verbose)
        per_city[city] = res
        
        print(f"   ✅ Planning completed for {city}")
        print(f"      Status: {res.get('status', 'unknown')}")
        if res.get('status') == 'ok':
            print(f"      Objective: {res.get('best', {}).get('objective', 'N/A')}")
            print(f"      POIs: {len(res.get('best', {}).get('pois', []))}")
            print(f"      Distance: {res.get('best', {}).get('total_distance', 'N/A')} km")
            
            # DEBUG: Show what the intracity planner actually returned
            best_plan = res.get('best', {})
            print(f"      🔍 INTRACITY PLANNER RETURNED:")
            print(f"         Hotel: {best_plan.get('hotel', 'N/A')}")
            print(f"         POIs: {best_plan.get('selected_count', 'N/A')}")
            print(f"         Cost: £{best_plan.get('total_travel_cost_gbp', 'N/A')}")
            print(f"         Distance: {best_plan.get('total_distance_km', 'N/A')}km")
            
            # Track best city based on user criteria:
            # 1. More POIs (higher selected_count)
            # 2. Least total travel cost (lower total_travel_cost_gbp)  
            # 3. Least total distance (lower total_distance_km)
            
            current_best = best_city[1]
            if current_best is None:
                best_city = (city, res)
                print(f"      🏆 New best city: {city}")
            else:
                current_plan = current_best["best"]
                new_plan = res["best"]
                
                print(f"      🔍 COMPARING PLANS:")
                print(f"         Current best: {current_plan['hotel']} ({current_plan['selected_count']} POIs, £{current_plan['total_travel_cost_gbp']} cost, {current_plan['total_distance_km']}km)")
                print(f"         New plan: {new_plan['hotel']} ({new_plan['selected_count']} POIs, £{new_plan['total_travel_cost_gbp']} cost, {new_plan['total_distance_km']}km)")
                
                # Compare POI count first (highest wins)
                if new_plan["selected_count"] > current_plan["selected_count"]:
                    best_city = (city, res)
                    print(f"      🏆 New best city: {city} (more POIs: {new_plan['selected_count']} vs {current_plan['selected_count']})")
                elif new_plan["selected_count"] == current_plan["selected_count"]:
                    # Same POI count, compare total travel cost (lowest wins)
                    if new_plan["total_travel_cost_gbp"] < current_plan["total_travel_cost_gbp"]:
                        best_city = (city, res)
                        print(f"      🏆 New best city: {city} (lower cost: £{new_plan['total_travel_cost_gbp']} vs £{current_plan['total_travel_cost_gbp']})")
                    elif new_plan["total_travel_cost_gbp"] == current_plan["total_travel_cost_gbp"]:
                        # Same cost, compare distance (lowest wins)
                        if new_plan["total_distance_km"] < current_plan["total_distance_km"]:
                            best_city = (city, res)
                            print(f"      🏆 New best city: {city} (shorter distance: {new_plan['total_distance_km']}km vs {current_plan['total_distance_km']}km)")
                        # If distance is also same, keep the current best
                    else:
                        print(f"      ❌ {city} has higher cost - keeping current best")
                else:
                    print(f"      ❌ {city} has fewer POIs - keeping current best")

    state["per_city"] = per_city
    state["best"] = {"city": best_city[0], "result": best_city[1]["best"]} if best_city[1] else None
    
    print(f"\n🎯 Planning Results Summary:")
    print(f"   - Cities processed: {len(cities)}")
    print(f"   - Feasible plans: {len([c for c in per_city.values() if c.get('status') == 'ok'])}")
    if best_city[1]:
        print(f"   - Best city: {best_city[0]}")
        print(f"   - Best objective: {best_city[1]['best'].get('objective', 'N/A')}")
    
    log.info(f"Stage 4 completed: Planner processed {len(cities)} cities, found {len([c for c in per_city.values() if c.get('status') == 'ok'])} feasible plans")
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

    # Add nodes with clear stage names
    g.add_node("commonsense", RunnableLambda(node_commonsense))
    g.add_node("tools", RunnableLambda(node_tools))
    g.add_node("policy_generator", RunnableLambda(lambda s: node_policy_generator(s, policy_spec)))
    g.add_node("planner", RunnableLambda(lambda s: node_planner(s, cp_verbose)))

    # Set clear sequential flow
    g.set_entry_point("commonsense")
    g.add_edge("commonsense", "tools")
    g.add_edge("tools", "policy_generator")
    g.add_edge("policy_generator", "planner")
    g.add_edge("planner", END)
    return g.compile()


# ---------------
# CLI
# ---------------
def main():
    ap = argparse.ArgumentParser(description="End-to-end trip orchestration with clear sequential flow")
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
            "  1) Template-guided synthesis (recommended): path to a template JSON, e.g. policy_budget_first.json\n"
            "  2) Direct synthesis (no template): 'synth:budget_first' or 'synth:stamina_first'\n"
            "  3) Legacy load (no synthesis): path to a full CP policy JSON"
        ),
    )
    ap.add_argument("--cp-verbose", action="store_true", help="Show OR-Tools internal logs")
    ap.add_argument("--output", type=str, default="orchestrator_results.json", help="Output JSON file path")
    args = ap.parse_args()
    
    # Log startup information
    print("\n" + "="*80)
    print("🚀 ORCHESTRATOR STARTUP")
    print("="*80)
    log.info(f"=== ORCHESTRATOR STARTUP ===")
    print(f"📝 Query: {args.query}")
    print(f"🎯 Policy: {args.policy}")
    print(f"🔍 CP Verbose: {args.cp_verbose}")
    print(f"💾 Output: {args.output}")
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
        print("\n" + "="*80)
        print("❌ ORCHESTRATOR FAILED")
        print("="*80)
        print("ERROR:", result["error"])
        if "raw" in result:
            print("RAW:", result["raw"])
        save_results_to_json(result, args.output)
        return

    print("\n" + "="*80)
    print("🎉 ORCHESTRATOR COMPLETED SUCCESSFULLY")
    print("="*80)
    
    log.info(f"=== ORCHESTRATOR RESULTS ===")
    log.info(f"Commonsense policy generated: {len(result.get('policy_raw', {}))} keys")
    log.info(f"Tool bundle created: {len(result.get('bundle', {}))} cities, {len(result.get('pois', []))} POIs, {len(result.get('hotels', []))} hotels")
    
    print("\n=== FINAL RESULTS SUMMARY ===")
    print(f"✅ Commonsense Policy: {len(result.get('policy_raw', {}))} keys generated")
    print(f"✅ Tool Bundle: {len(result.get('bundle', {}).get('cities', []))} cities, {len(result.get('pois', []))} POIs, {len(result.get('hotels', []))} hotels")
    print(f"✅ Generated Policy: {result.get('planner_policy', {}).get('unique_id', 'unknown')}")
    
    if result.get("best"):
        log.info(f"Best plan found for city: {result['best']['city']}")
        log.info(f"Best plan objective: {result['best']['result'].get('objective', 'N/A')}")
        print(f"✅ Best Plan: {result['best']['city']} (Objective: {result['best']['result'].get('objective', 'N/A')})")
    else:
        log.warning("No feasible plans found")
        print("⚠️  No feasible plans found.")

    print("\n=== DETAILED RESULTS ===")
    print("📋 Generated Policy:")
    print(json.dumps(result.get("planner_policy", {}), indent=2, ensure_ascii=False))
    
    if result.get("best"):
        print("\n🏆 Best Day Plan:")
        print("City:", result["best"]["city"])
        print(json.dumps(result["best"]["result"], indent=2, ensure_ascii=False))
    
    print("\n🏙️  Per-City Results:")
    print(json.dumps(result.get("per_city", {}), indent=2, ensure_ascii=False))

    save_results_to_json(result, args.output)
    log.info(f"Results saved to {args.output}")
    log.info(f"=== ORCHESTRATOR COMPLETED SUCCESSFULLY ===")


if __name__ == "__main__":
    main()