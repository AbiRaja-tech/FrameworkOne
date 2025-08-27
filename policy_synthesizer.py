# policy_synthesizer.py
# Gemini-driven, persona-aware policy builder for the intracity CP-SAT planner.
# - Uses Google's Gemini to merge:
#     (a) a planner-compatible BASE policy,
#     (b) commonsense_agent output (`policy_raw`),
#     (c) an optional template overlay (`templates`)
#   into a single personalized policy JSON.
# - Enforces required keys and computes lexicographic priority_weights if missing.
#
# ENV:
#   GEMINI_API_KEY or GOOGLE_API_KEY   (required)
#   GEMINI_MODEL (optional, default: "gemini-1.5-pro")

from __future__ import annotations
from typing import Dict, Any, List, Optional
from copy import deepcopy
import json, os, re
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Load environment variables from .env file
load_dotenv()

# ----------------------------
# Planner-safe weighting
# ----------------------------
SAFE_LEX_BASE = 10**6
MAX_WEIGHT    = 10**12

def _lex_from_order(order: List[str]) -> Dict[str, int]:
    n = len(order or [])
    weights: Dict[str, int] = {}
    for i, k in enumerate(order or []):
        w = SAFE_LEX_BASE ** (n - i)
        if w > MAX_WEIGHT:
            w = MAX_WEIGHT
        weights[k] = int(w)
    return weights

def _merge_dict(dst: Dict[str, Any], src: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(src, dict):
        return dst
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _merge_dict(dst[k], v)
        else:
            dst[k] = v
    return dst

# ----------------------------
# Base policy (exact shape expected by intracity_planner)
# ----------------------------
def default_policy_template() -> Dict[str, Any]:
    return {
        "unique_id": "default_constraints",
        "priority_order": ["stamina", "budget", "poi_score", "experience", "comfort"],
        "budget": {
            "daily_cap": 150.0,
            "mode_fixed_cost": {
                "walk": 0.0,
                "train": 1.0,
                "bus": 2.0,
                "tram": 0.0,
                "cab": 5.0
            },
            "cab_distance_pricing": {
                "threshold_1": 2.0,
                "cost_under_2km": 5.0,
                "cost_2_to_4km": 10.0,
                "cost_over_4km": 15.0
            }
        },
        "stamina": {
            "start": 10.0,
            "poi_visit_cost": 1.5,
            "poi_liked_reduction": 1.0,
            "poi_disliked_reduction": 2.0,
            "meal_gain": 2.0,
            "transport_costs": {
                "walk_per_hour": 2.0,
                "train_per_hour": 1.0,
                "bus_per_hour": 1.0,
                "tram_per_hour": 1.0,
                "cab_per_hour": 0.5
            }
        },
        "planning": {
            "max_pois_per_day": 6,
            "min_pois_per_day": 1,
            "max_one_way_distance_from_hotel_km": 20.0,
            "max_total_distance_day_km": 50.0
        },
        "preferences": {
            "must_include": [],
            "must_avoid": []
        },
        "comfort": {
            "discomfort_per_min": {
                "walk": 1.0,
                "train": 0.6,
                "bus": 0.5,
                "tram": 0.6,
                "cab": 0.2
            },
            "transfer_penalty_per_change": 0.5
        },
        "solver": {
            "max_seconds": 10.0,
            "workers": 8
        }
    }

# ----------------------------
# Robust JSON extraction (handles code fences, extra text)
# ----------------------------
def _extract_json(text: str) -> Dict[str, Any]:
    if not isinstance(text, str):
        raise ValueError("Model response is not a string.")
    # Strip common code fences if present
    text = text.strip()
    text = re.sub(r"^\s*```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```\s*$", "", text)
    # Greedy match first JSON object
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        # Try parse as-is
        return json.loads(text)
    return json.loads(m.group(0))

# ----------------------------
# Build the Gemini prompt
# ----------------------------
def _build_prompt(base_policy: Dict[str, Any],
                  policy_raw: Dict[str, Any],
                  template_overlay: Optional[Dict[str, Any]]) -> str:
    """
    We give Gemini:
      - A strict instruction to output JSON only
      - The base policy (as defaults)
      - The commonsense_agent output (policy_raw)
      - Optional template overlay to respect/merge
      - A schema contract: allowed keys and expectations
    """
    # Optional: pass original query if upstream attached it into policy_raw
    query_hint = policy_raw.get("__query") or policy_raw.get("query") or ""

    schema_contract = {
        "required_keys": [
            "unique_id", "priority_order", "priority_weights",
            "budget", "stamina", "planning", "preferences", "comfort", "solver"
        ],
        "notes": [
            "Return STRICT JSON only. No prose. No code fences.",
            "Use the base policy as defaults, then adapt to the persona/query commonsense.",
            "If template overlay is present, respect it and merge sensibly.",
            "Compute priority_weights as lexicographic weights (1e6 base) from priority_order.",
            "Do not invent new top-level keys."
        ],
        "field_expectations": {
            "priority_order": "A list ordering some or all of: stamina, budget, poi_score, experience, comfort.",
            "priority_weights": "Large integers reflecting lexicographic priority; if missing, they will be recomputed.",
            "budget.daily_cap": "GBP per day, float.",
            "budget.mode_fixed_cost": "GBP per trip by mode.",
            "budget.cab_distance_pricing": "Pricing tiers by distance (km).",
            "stamina": "Start stamina (float), visit costs/gains, transport stamina costs/hour.",
            "planning": "max_pois_per_day, min_pois_per_day, max_one_way_distance_from_hotel_km, max_total_distance_day_km.",
            "preferences": "must_include/must_avoid: array of category keywords.",
            "comfort": "discomfort_per_min per mode, transfer_penalty_per_change.",
            "solver": "max_seconds (float), workers (int)."
        }
    }

    pieces = {
        "instruction": (
            "You are a travel-policy generator. Create a CP-SAT planner policy JSON "
            "tailored to the persona/query. Use the BASE policy as defaults; merge the "
            "commonsense context; respect the TEMPLATE overlay if provided. "
            "Output strict JSON ONLY."
        ),
        "query_hint": query_hint,
        "base_policy": base_policy,
        "commonsense_policy_raw": policy_raw,
        "template_overlay": template_overlay or {},
        "schema_contract": schema_contract
    }
    return json.dumps(pieces, ensure_ascii=False)

# ----------------------------
# Gemini call
# ----------------------------
def _call_gemini(prompt: str) -> str:
    # Lazy import to keep this file importable without the SDK installed
    try:
        import google.generativeai as genai
    except Exception as e:
        raise RuntimeError(
            "google-generativeai not installed. Install with `pip install google-generativeai`."
        ) from e

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY (or GOOGLE_API_KEY) not set in environment.")

    genai.configure(api_key=api_key)
    model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")
    model = genai.GenerativeModel(model_name)

    # Slightly cautious decoding config; we want structure, not creativity
    gen_cfg = {
        "temperature": 0.3,
        "max_output_tokens": 2048,
    }
    resp = model.generate_content(prompt, generation_config=gen_cfg)
    # Depending on SDK version, text may be at different properties:
    if hasattr(resp, "text") and resp.text:
        return resp.text
    # Fallbacks for earlier SDKs
    try:
        return "".join(p.text for p in resp.candidates[0].content.parts)
    except Exception:
        pass
    raise RuntimeError("Gemini returned an empty response.")

# ----------------------------
# Persona inference
# ----------------------------
def _infer_persona_flags(policy_raw: Dict[str, Any]) -> Dict[str, bool]:
    """Infer persona flags from policy_raw data"""
    flags = {
        "elderly": False,
        "family_with_children": False,
        "toddlers": False,
        "teens": False,
        "solo_backpacker": False,
        "luxury": False,
        "tight_budget": False,
        "mobility_issues": False,
        "business_trip": False
    }
    
    # Check ages
    ages = policy_raw.get("ages", [])
    if ages:
        avg_age = sum(ages) / len(ages)
        if avg_age > 65:
            flags["elderly"] = True
    
    # Check group type and family indicators
    group_type = policy_raw.get("group_type", "").lower()
    persona = policy_raw.get("persona", "").lower()
    if "family" in group_type or "family" in persona or "children" in persona:
        flags["family_with_children"] = True
        # Check for toddlers/teens
        if ages:
            for age in ages:
                if age < 5:
                    flags["toddlers"] = True
                elif 5 <= age <= 17:
                    flags["teens"] = True
    
    # Check budget level
    budget_level = policy_raw.get("budget_level", "").lower()
    if "tight" in budget_level or "low" in budget_level:
        flags["tight_budget"] = True
    elif "luxury" in budget_level or "high" in budget_level:
        flags["luxury"] = True
    
    # Check transport preferences
    transport = policy_raw.get("transport_intent", "").lower()
    if "walk" in transport and "tram" in transport:
        flags["solo_backpacker"] = True
    
    # Check business indicators
    if "business" in persona or "work" in persona:
        flags["business_trip"] = True
    
    return flags

# ----------------------------
# Persona rules
# ----------------------------
def _apply_rule(policy: Dict[str, Any], rule_name: str, rule_fn):
    """Apply a persona rule to the policy"""
    try:
        rule_fn(policy)
        policy["unique_id"] = f"{policy.get('unique_id', 'default')}_{rule_name}"
    except Exception as e:
        print(f"Warning: Rule {rule_name} failed: {e}")

def _rule_elderly(policy: Dict[str, Any]):
    """Apply elderly-specific adjustments"""
    # Reduce stamina costs, increase comfort
    if "stamina" in policy:
        policy["stamina"]["start"] = 8.0  # Lower starting stamina
        policy["stamina"]["poi_visit_cost"] = 1.0  # Lower POI stamina cost
    
    if "comfort" in policy:
        # Prefer more comfortable transport
        policy["comfort"]["discomfort_per_min"]["walk"] = 1.5
        policy["comfort"]["discomfort_per_min"]["cab"] = 0.1

def _rule_family_with_children(policy: Dict[str, Any], toddlers: bool, teens: bool):
    """Apply family-specific adjustments"""
    if "planning" in policy:
        policy["planning"]["max_pois_per_day"] = 4  # Fewer POIs for families
        policy["planning"]["max_total_distance_day_km"] = 30.0  # Shorter distances
    
    if "comfort" in policy:
        # Prefer family-friendly transport
        policy["comfort"]["discomfort_per_min"]["walk"] = 0.8
        policy["comfort"]["discomfort_per_min"]["tram"] = 0.4

def _rule_solo_backpacker(policy: Dict[str, Any]):
    """Apply solo backpacker adjustments"""
    if "planning" in policy:
        policy["planning"]["max_pois_per_day"] = 8  # More POIs for solo travelers
        policy["planning"]["max_total_distance_day_km"] = 60.0  # Longer distances
    
    if "comfort" in policy:
        # Prefer walking and public transport
        policy["comfort"]["discomfort_per_min"]["walk"] = 0.6
        policy["comfort"]["discomfort_per_min"]["cab"] = 0.8

def _rule_luxury(policy: Dict[str, Any]):
    """Apply luxury adjustments"""
    if "budget" in policy:
        policy["budget"]["daily_cap"] = 300.0  # Higher budget
    
    if "comfort" in policy:
        # Prefer comfortable transport
        policy["comfort"]["discomfort_per_min"]["cab"] = 0.1
        policy["comfort"]["discomfort_per_min"]["train"] = 0.3

def _rule_tight_budget(policy: Dict[str, Any]):
    """Apply tight budget adjustments"""
    if "budget" in policy:
        policy["budget"]["daily_cap"] = 80.0  # Lower budget
    
    if "comfort" in policy:
        # Prefer cheaper transport
        policy["comfort"]["discomfort_per_min"]["walk"] = 0.5
        policy["comfort"]["discomfort_per_min"]["cab"] = 1.0

def _rule_mobility_issues(policy: Dict[str, Any]):
    """Apply mobility issue adjustments"""
    if "stamina" in policy:
        policy["stamina"]["start"] = 6.0  # Lower starting stamina
        policy["stamina"]["poi_visit_cost"] = 2.0  # Higher POI stamina cost
    
    if "comfort" in policy:
        # Prefer accessible transport
        policy["comfort"]["discomfort_per_min"]["walk"] = 2.0
        policy["comfort"]["discomfort_per_min"]["cab"] = 0.1

def _rule_business_trip(policy: Dict[str, Any]):
    """Apply business trip adjustments"""
    if "planning" in policy:
        policy["planning"]["max_pois_per_day"] = 3  # Fewer POIs for business
        policy["planning"]["max_total_distance_day_km"] = 25.0  # Shorter distances
    
    if "comfort" in policy:
        # Prefer efficient transport
        policy["comfort"]["discomfort_per_min"]["cab"] = 0.2
        policy["comfort"]["discomfort_per_min"]["train"] = 0.4

# ----------------------------
# Public API
# ----------------------------
def synthesize_policy(policy_raw: Dict[str, Any],
                      profile: Optional[str] = None,
                      templates: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Build a planner-ready policy using Gemini:
      1) Start from BASE policy (planner-compatible)
      2) Merge TEMPLATE overlay (if provided)
      3) Call Gemini with BASE, policy_raw (commonsense), TEMPLATE, and schema instructions
      4) Parse JSON; enforce required keys; recompute priority_weights if missing
      5) Return policy dict
    """
    base = default_policy_template()
    # Merge overlay BEFORE sending to Gemini so the model sees the intended lean
    base_plus_overlay = deepcopy(base)
    if isinstance(templates, dict):
        _merge_dict(base_plus_overlay, deepcopy(templates))

    # Temporarily disable LLM due to API quota limits
    # prompt = _build_prompt(base_plus_overlay, policy_raw or {}, templates if isinstance(templates, dict) else None)
    # raw_text = _call_gemini(prompt)
    # try:
    #     model_json = _extract_json(raw_text)
    # except Exception as e:
    #     # As a last resort, return the base+overlay with weights; this keeps pipeline running.
    #     out = deepcopy(base_plus_overlay)
    #     out["unique_id"] = "gemini_fallback_default"
    #     out["priority_weights"] = _lex_from_order(out.get("priority_order") or [])
    #     return out
    
    # Use rule-based fallback for now
    model_json = {}
    print("INFO: Using rule-based policy synthesis (LLM disabled due to API quota)")

    # Safety: enforce required sections & fill from base if missing
    policy = deepcopy(base)  # start with base defaults
    _merge_dict(policy, base_plus_overlay)  # then overlay
    _merge_dict(policy, model_json)         # then model output

    # Ensure mandatory sections exist
    for sec in ("budget", "stamina", "planning", "preferences", "comfort", "solver"):
        policy.setdefault(sec, deepcopy(base[sec]))

    # Ensure priority_order & weights
    if "priority_order" not in policy or not policy["priority_order"]:
        policy["priority_order"] = deepcopy(base["priority_order"])
    if "priority_weights" not in policy or not policy["priority_weights"]:
        policy["priority_weights"] = _lex_from_order(policy["priority_order"])

    # Apply persona rules
    flags = _infer_persona_flags(policy_raw)
    
    if flags["elderly"]:
        _apply_rule(policy, "elderly", _rule_elderly)
    
    if flags["family_with_children"]:
        _apply_rule(policy, "family_with_children",
                    lambda pol: _rule_family_with_children(pol, flags["toddlers"], flags["teens"]))
    
    if flags["solo_backpacker"]:
        _apply_rule(policy, "solo_backpacker", _rule_solo_backpacker)
    
    if flags["luxury"]:
        _apply_rule(policy, "luxury", _rule_luxury)
    
    if flags["tight_budget"]:
        _apply_rule(policy, "tight_budget", _rule_tight_budget)
    
    if flags["mobility_issues"]:
        _apply_rule(policy, "mobility_issues", _rule_mobility_issues)
    
    if flags["business_trip"]:
        _apply_rule(policy, "business_trip", _rule_business_trip)

    # Unique id to mark provenance
    policy["unique_id"] = policy.get("unique_id") or "persona_synthesized"

    return policy

# ----------------------------
# Manual test
# ----------------------------
if __name__ == "__main__":
    # Example: emulate what your orchestrator passes here
    demo_commonsense = {
        "__query": "3-day Manchester family trip with two kids (5 and 11). Prefer trams & short walks. Medium budget.",
        "poi_preferences": {"museums": "high", "parks": "prefer", "shopping": "low"},
        "budget_constraints": {"daily_budget": 160, "currency": "GBP"},
        "transport_intent": "tram and walking",
        "ages": [35, 34, 11, 5],
        "group_type": "family",
        "persona": "family with children"
    }
    # Optional overlay: e.g., stamina-first template
    overlay = None  # or load a dict from a file if you want

    pol = synthesize_policy(demo_commonsense, templates=overlay)
    print(json.dumps(pol, indent=2, ensure_ascii=False))
