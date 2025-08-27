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
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Enhanced rule-based policy generation with DeepSeek integration
from deepseek_policy_generator import generate_policy_with_deepseek

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
            "max_one_way_distance_from_hotel_km": 15.0,
            "max_total_distance_day_km": 30.0
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
            "You are an expert travel-policy generator for CP-SAT planning. Your task is to create a highly personalized "
            "travel policy JSON that adapts the BASE policy to the specific user's needs, preferences, and constraints. "
            "Analyze the query context, user persona, and commonsense data to make intelligent adjustments. "
            "Consider factors like age, group type, budget level, transport preferences, and activity interests. "
            "Use the BASE policy as defaults, then intelligently modify values based on the user's profile. "
            "Output strict JSON ONLY - no explanations or prose."
        ),
        "query_hint": query_hint,
        "base_policy": base_policy,
        "commonsense_policy_raw": policy_raw,
        "template_overlay": template_overlay or {},
        "schema_contract": schema_contract,
        "personalization_guidelines": {
            "budget_adjustments": {
                "luxury_travelers": "Increase daily_cap by 50-100%, prefer premium transport",
                "budget_conscious": "Reduce daily_cap by 20-40%, emphasize walking/public transport",
                "family_trips": "Moderate budget with family-friendly transport preferences"
            },
            "stamina_adjustments": {
                "elderly_travelers": "Reduce start stamina by 20-30%, lower POI visit costs",
                "active_travelers": "Increase start stamina by 20-30%, higher POI visit costs",
                "family_with_children": "Moderate stamina with child-friendly planning constraints"
            },
            "planning_adjustments": {
                "solo_backpackers": "More POIs per day, longer distances, flexible planning",
                "business_travelers": "Fewer POIs, efficient routes, comfort-focused transport",
                "cultural_enthusiasts": "Prioritize museums/historic sites, adjust preferences accordingly"
            }
        }
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
    """Enhanced persona inference with more sophisticated analysis"""
    flags = {
        "elderly": False,
        "family_with_children": False,
        "toddlers": False,
        "teens": False,
        "solo_backpacker": False,
        "luxury": False,
        "tight_budget": False,
        "mobility_issues": False,
        "business_trip": False,
        "cultural_enthusiast": False,
        "active_adventurer": False,
        "comfort_seeker": False
    }
    
    # Enhanced age analysis
    ages = policy_raw.get("ages", [])
    if ages:
        avg_age = sum(ages) / len(ages)
        if avg_age > 65:
            flags["elderly"] = True
            flags["comfort_seeker"] = True
        elif avg_age < 30:
            flags["active_adventurer"] = True
        elif 30 <= avg_age <= 50:
            flags["comfort_seeker"] = True
    
    # Enhanced group analysis
    group_type = policy_raw.get("group_type", "").lower()
    persona = policy_raw.get("persona", "").lower()
    query = policy_raw.get("__query", "").lower()
    
    if any(word in group_type + persona + query for word in ["family", "children", "kids", "child"]):
        flags["family_with_children"] = True
        if ages:
            for age in ages:
                if age < 5:
                    flags["toddlers"] = True
                elif 5 <= age <= 17:
                    flags["teens"] = True
    
    # Enhanced budget analysis
    budget_level = policy_raw.get("budget_level", "").lower()
    budget_constraints = policy_raw.get("budget_constraints", {})
    daily_budget = budget_constraints.get("daily_budget", 0)
    
    if any(word in budget_level for word in ["tight", "low", "budget", "cheap"]) or daily_budget < 100:
        flags["tight_budget"] = True
    elif any(word in budget_level for word in ["luxury", "high", "premium"]) or daily_budget > 200:
        flags["luxury"] = True
    
    # Enhanced transport and activity analysis
    transport = policy_raw.get("transport_intent", "").lower()
    poi_prefs = policy_raw.get("poi_preferences", {})
    
    if "walk" in transport and ("tram" in transport or "public" in transport):
        flags["solo_backpacker"] = True
        flags["active_adventurer"] = True
    
    if any(pref == "high" for pref in poi_prefs.values()):
        flags["cultural_enthusiast"] = True
    
    # Enhanced business and mobility analysis
    if any(word in persona + query for word in ["business", "work", "meeting", "conference"]):
        flags["business_trip"] = True
        flags["comfort_seeker"] = True
    
    if any(word in query for word in ["wheelchair", "accessible", "mobility", "disability"]):
        flags["mobility_issues"] = True
        flags["comfort_seeker"] = True
    
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

def _rule_cultural_enthusiast(policy: Dict[str, Any]):
    """Apply cultural enthusiast adjustments"""
    if "planning" in policy:
        policy["planning"]["max_pois_per_day"] = 7  # More POIs for cultural exploration
        policy["planning"]["max_total_distance_day_km"] = 45.0  # Moderate distances
    
    if "preferences" in policy:
        policy["preferences"]["must_include"] = ["museum", "historic", "cultural", "arts"]
    
    if "stamina" in policy:
        policy["stamina"]["poi_visit_cost"] = 1.2  # Lower stamina cost for cultural activities

def _rule_active_adventurer(policy: Dict[str, Any]):
    """Apply active adventurer adjustments"""
    if "planning" in policy:
        policy["planning"]["max_pois_per_day"] = 8  # More POIs for active travelers
        policy["planning"]["max_total_distance_day_km"] = 70.0  # Longer distances
    
    if "stamina" in policy:
        policy["stamina"]["start"] = 12.0  # Higher starting stamina
        policy["stamina"]["poi_visit_cost"] = 1.8  # Higher stamina cost for active activities
    
    if "comfort" in policy:
        policy["comfort"]["discomfort_per_min"]["walk"] = 0.4  # Prefer walking

def _rule_comfort_seeker(policy: Dict[str, Any]):
    """Apply comfort seeker adjustments"""
    if "comfort" in policy:
        # Prefer comfortable transport modes
        policy["comfort"]["discomfort_per_min"]["cab"] = 0.1
        policy["comfort"]["discomfort_per_min"]["train"] = 0.3
        policy["comfort"]["discomfort_per_min"]["walk"] = 1.2  # Higher walking discomfort
    
    if "planning" in policy:
        policy["planning"]["max_total_distance_day_km"] = 35.0  # Shorter distances for comfort

# ----------------------------
# Detailed logging functions
# ----------------------------
def _log_ai_changes(base_policy: Dict[str, Any], ai_output: Dict[str, Any]):
    """Log detailed changes made by AI (DeepSeek)"""
    changes = []
    
    # Budget changes
    if "budget" in ai_output:
        budget_changes = ai_output["budget"]
        if "daily_cap" in budget_changes:
            old_val = base_policy["budget"]["daily_cap"]
            new_val = budget_changes["daily_cap"]
            if old_val != new_val:
                change = f"💰 Budget daily cap: £{old_val} → £{new_val}"
                if new_val > old_val:
                    change += " 📈 (INCREASED)"
                else:
                    change += " 📉 (DECREASED)"
                changes.append(change)
    
    # Stamina changes
    if "stamina" in ai_output:
        stamina_changes = ai_output["stamina"]
        if "start" in stamina_changes:
            old_val = base_policy["stamina"]["start"]
            new_val = stamina_changes["start"]
            if old_val != new_val:
                change = f"💪 Stamina start: {old_val} → {new_val}"
                if new_val > old_val:
                    change += " 📈 (INCREASED)"
                else:
                    change += " 📉 (DECREASED)"
                changes.append(change)
        
        if "poi_visit_cost" in stamina_changes:
            old_val = base_policy["stamina"]["poi_visit_cost"]
            new_val = stamina_changes["poi_visit_cost"]
            if old_val != new_val:
                change = f"🎯 POI visit cost: {old_val} → {new_val}"
                if new_val < old_val:
                    change += " 📉 (REDUCED - easier visits)"
                else:
                    change += " 📈 (INCREASED - harder visits)"
                changes.append(change)
    
    # Planning changes
    if "planning" in ai_output:
        planning_changes = ai_output["planning"]
        if "max_pois_per_day" in planning_changes:
            old_val = base_policy["planning"]["max_pois_per_day"]
            new_val = planning_changes["max_pois_per_day"]
            if old_val != new_val:
                change = f"📍 Max POIs per day: {old_val} → {new_val}"
                if new_val > old_val:
                    change += " 📈 (MORE POIs - active traveler)"
                else:
                    change += " 📉 (FEWER POIs - relaxed pace)"
                changes.append(change)
        
        if "max_total_distance_day_km" in planning_changes:
            old_val = base_policy["planning"]["max_total_distance_day_km"]
            new_val = planning_changes["max_total_distance_day_km"]
            if old_val != new_val:
                change = f"🛣️  Max daily distance: {old_val}km → {new_val}km"
                if new_val > old_val:
                    change += " 📈 (LONGER distances - adventurous)"
                else:
                    change += " 📉 (SHORTER distances - comfort-focused)"
                changes.append(change)
    
    # Preferences changes
    if "preferences" in ai_output:
        pref_changes = ai_output["preferences"]
        if "must_include" in pref_changes and pref_changes["must_include"]:
            changes.append(f"🎨 Must include categories: {pref_changes['must_include']}")
        if "must_avoid" in pref_changes and pref_changes["must_avoid"]:
            changes.append(f"❌ Must avoid categories: {pref_changes['must_avoid']}")
    
    # Comfort changes
    if "comfort" in ai_output:
        comfort_changes = ai_output["comfort"]
        if "discomfort_per_min" in comfort_changes:
            mode_changes = []
            for mode, discomfort in comfort_changes["discomfort_per_min"].items():
                if mode in base_policy["comfort"]["discomfort_per_min"]:
                    old_val = base_policy["comfort"]["discomfort_per_min"][mode]
                    if old_val != discomfort:
                        mode_changes.append(f"{mode}: {old_val} → {discomfort}")
            if mode_changes:
                changes.append(f"🚗 Transport comfort changes: {', '.join(mode_changes)}")
    
    # Log all changes
    if changes:
        for change in changes:
            print(f"   🔄 {change}")
    else:
        print("   ℹ️  No significant changes detected")
    
    # Log any new keys added by AI
    new_keys = set(ai_output.keys()) - set(base_policy.keys())
    if new_keys:
        print(f"   ✨ New sections added: {list(new_keys)}")
    
    # Log any modified sections
    modified_sections = []
    for section in ai_output:
        if section in base_policy and ai_output[section] != base_policy[section]:
            modified_sections.append(section)
    
    if modified_sections:
        print(f"   🔧 Modified sections: {modified_sections}")

# ----------------------------
# Enhanced Intelligent Policy Generation
# ----------------------------
def _load_base_policy(profile: Optional[str] = None) -> Dict[str, Any]:
    """Load base policy template based on profile"""
    base = default_policy_template()
    
    # Apply profile-specific modifications
    if profile == "budget_first":
        base["budget"]["daily_cap"] = 50.0
        base["planning"]["max_pois_per_day"] = 6
        base["planning"]["max_total_distance_day_km"] = 25.0
    elif profile == "stamina_first":
        base["stamina"]["start"] = 15.0
        base["planning"]["max_pois_per_day"] = 4
        base["planning"]["max_total_distance_day_km"] = 15.0
    
    return base


def _apply_user_preferences(base: Dict[str, Any], policy_raw: Dict[str, Any]) -> Dict[str, Any]:
    """Apply user preferences overlay to base policy"""
    result = deepcopy(base)
    
    # Extract key user preferences
    if "budget_constraints" in policy_raw:
        budget_info = policy_raw["budget_constraints"]
        if "max_budget" in budget_info:
            result["budget"]["daily_cap"] = float(budget_info["max_budget"])
        if "budget_level" in budget_info:
            if budget_info["budget_level"] == "tight":
                result["budget"]["daily_cap"] = min(result["budget"]["daily_cap"], 30.0)
            elif budget_info["budget_level"] == "luxury":
                result["budget"]["daily_cap"] = max(result["budget"]["daily_cap"], 100.0)
    
    if "transport_preferences" in policy_raw:
        transport_info = policy_raw["transport_preferences"]
        if "preferred_mode" in transport_info:
            result["comfort"]["transport_mode"] = transport_info["preferred_mode"]
    
    if "activity_preferences" in policy_raw:
        activity_info = policy_raw["activity_preferences"]
        if "interests" in activity_info:
            result["preferences"]["must_include"] = activity_info["interests"]
    
    return result


def _apply_persona_rules(policy: Dict[str, Any], persona_flags: Dict[str, bool]) -> Dict[str, Any]:
    """Apply persona-based rules to the policy"""
    result = deepcopy(policy)
    
    # Apply specific persona rules
    if persona_flags.get("elderly"):
        _apply_rule(result, "elderly", _rule_elderly)
    
    if persona_flags.get("family_with_children"):
        _apply_rule(result, "family_with_children",
                    lambda pol: _rule_family_with_children(pol, persona_flags.get("toddlers", False), persona_flags.get("teens", False)))
    
    if persona_flags.get("solo_backpacker"):
        _apply_rule(result, "solo_backpacker", _rule_solo_backpacker)
    
    if persona_flags.get("luxury"):
        _apply_rule(result, "luxury", _rule_luxury)
    
    if persona_flags.get("tight_budget"):
        _apply_rule(result, "tight_budget", _rule_tight_budget)
    
    if persona_flags.get("mobility_issues"):
        _apply_rule(result, "mobility_issues", _rule_mobility_issues)
    
    if persona_flags.get("business_trip"):
        _apply_rule(result, "business_trip", _rule_business_trip)
    
    if persona_flags.get("cultural_enthusiast"):
        _apply_rule(result, "cultural_enthusiast", _rule_cultural_enthusiast)
    
    if persona_flags.get("active_adventurer"):
        _apply_rule(result, "active_adventurer", _rule_active_adventurer)
    
    if persona_flags.get("comfort_seeker"):
        _apply_rule(result, "comfort_seeker", _rule_comfort_seeker)
    
    # Ensure required sections exist
    for sec in ("budget", "stamina", "planning", "preferences", "comfort", "solver"):
        if sec not in result:
            result[sec] = deepcopy(default_policy_template()[sec])
    
    # Ensure priority_order & weights
    if "priority_order" not in result or not result["priority_order"]:
        result["priority_order"] = deepcopy(default_policy_template()["priority_order"])
    if "priority_weights" not in result or not result["priority_weights"]:
        result["priority_weights"] = _lex_from_order(result["priority_order"])
    
    return result


def _generate_intelligent_policy(policy_raw: Dict[str, Any], 
                                base_policy: Dict[str, Any],
                                templates: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Generate intelligent policy modifications based on deep query analysis"""
    
    modifications = {}
    
    # Analyze query for specific requirements
    query = policy_raw.get("__query", "").lower()
    
    # Budget analysis and modifications
    budget_constraints = policy_raw.get("budget_constraints", {})
    if budget_constraints:
        if "max_budget" in budget_constraints:
            max_budget = float(budget_constraints["max_budget"])
            if max_budget < 50:
                modifications["budget"] = {"daily_cap": max_budget * 0.8, "budget_level": "tight"}
            elif max_budget > 100:
                modifications["budget"] = {"daily_cap": max_budget * 0.6, "budget_level": "luxury"}
            else:
                modifications["budget"] = {"daily_cap": max_budget * 0.7, "budget_level": "medium"}
    
    # Activity preferences analysis
    activity_prefs = policy_raw.get("activity_preferences", {})
    if activity_prefs:
        interests = activity_prefs.get("interests", [])
        if interests:
            modifications["preferences"] = {"must_include": interests}
            
            # Adjust planning based on interests
            if any(interest in ["museum", "historic", "cultural"] for interest in interests):
                modifications["planning"] = {"max_pois_per_day": 6, "max_total_distance_day_km": 20.0}
            elif any(interest in ["adventure", "outdoor", "sports"] for interest in interests):
                modifications["planning"] = {"max_pois_per_day": 8, "max_total_distance_day_km": 35.0}
    
    # Transport preferences
    transport_prefs = policy_raw.get("transport_preferences", {})
    if transport_prefs:
        preferred_mode = transport_prefs.get("preferred_mode", "walking")
        if preferred_mode == "walking":
            modifications["comfort"] = {"max_walking_distance_km": 15.0, "transport_mode": "walking"}
        elif preferred_mode == "public_transport":
            modifications["comfort"] = {"max_walking_distance_km": 8.0, "transport_mode": "public_transport"}
        elif preferred_mode == "taxi":
            modifications["comfort"] = {"max_walking_distance_km": 3.0, "transport_mode": "taxi"}
    
    # Apply template overlay if provided
    if templates:
        for key, value in templates.items():
            if key not in modifications:
                modifications[key] = value
            elif isinstance(value, dict) and isinstance(modifications[key], dict):
                modifications[key].update(value)
            else:
                modifications[key] = value
    
    print("🧠 Intelligent analysis completed:")
    print(f"   - Budget analysis: {modifications.get('budget', {})}")
    print(f"   - POI preferences: {modifications.get('preferences', {})}")
    print(f"   - Transport comfort: {modifications.get('comfort', {})}")
    print(f"   - Planning constraints: {modifications.get('planning', {})}")
    
    return modifications


def _force_walking_constraints(policy: Dict[str, Any], policy_raw: Dict[str, Any]):
    """Force walking constraints, POI limits, transport preferences, and time constraints based on persona detection"""
    print("🚶 FORCING PERSONA-AWARE CONSTRAINTS...")
    
    # Detect persona from query
    query = policy_raw.get("__query", "").lower()
    
    # Extract age information if available
    ages = policy_raw.get("ages", [])
    has_children = any(word in query for word in ["children", "kids", "child", "toddler", "teen"])
    has_elderly = any(word in query for word in ["elderly", "senior", "65+", "retirement", "old"])
    
    # Ensure planning section exists
    if "planning" not in policy:
        policy["planning"] = {}
    
    # Force family constraints if detected
    if any(word in query for word in ["family", "children", "kids", "child", "toddler", "teen"]):
        print("   👨‍👩‍👧‍👦 FAMILY DETECTED - Applying strict constraints")
        
        # POI limits for families
        if has_elderly:
            policy["planning"]["max_pois_per_day"] = 2  # Very limited for elderly families
            policy["planning"]["min_pois_per_day"] = 1
            print("   ✅ Elderly family POI limits: 1-2 POIs per day")
        else:
            policy["planning"]["max_pois_per_day"] = 3  # Limited for families with children
            policy["planning"]["min_pois_per_day"] = 2
            print("   ✅ Family POI limits: 2-3 POIs per day")
        
        # Walking constraints for families (very strict)
        policy["planning"]["max_walking_distance_day_km"] = 8.0  # Reduced from 15.0
        policy["planning"]["max_one_shot_walking_km"] = 1.0  # MAX 1km for families
        policy["planning"]["walking_preference_threshold_km"] = 0.5  # Prefer walking only up to 0.5km
        
        print(f"   ✅ Family walking limits: max_one_shot=1.0km, daily_max=8.0km, threshold=0.5km")
        
        # TIME CONSTRAINTS for families
        policy["planning"]["max_daily_total_time_hours"] = 6.0  # Shorter day for families
        policy["planning"]["max_daily_travel_time_hours"] = 2.5  # Limited travel time
        policy["planning"]["poi_visit_times"] = {
            "favorite_categories": 90,   # 1.5 hours for museums (shorter attention spans)
            "neutral_categories": 60,    # 1 hour for shopping, parks
            "disliked_categories": 45    # 45 minutes for disliked activities
        }
        print("   ✅ Family time limits: 6 hours total, 2.5 hours travel, shorter POI visits")
        
        # Transport preferences for families
        if "comfort" not in policy:
            policy["comfort"] = {}
        
        # Families prefer bus and cab, avoid walking
        policy["comfort"]["discomfort_per_min"] = {
            "walk": 3.0,      # High walking discomfort for families
            "train": 0.8,     # Moderate train discomfort
            "bus": 0.3,       # Low bus discomfort (preferred)
            "tram": 0.4,      # Low tram discomfort
            "cab": 0.1        # Very low cab discomfort (preferred)
        }
        
        # Age-specific transport preferences
        if ages:
            avg_child_age = sum(age for age in ages if age < 18) / len([age for age in ages if age < 18]) if any(age < 18 for age in ages) else 0
            if avg_child_age > 0:
                if 7 <= avg_child_age <= 12:
                    print(f"   🚕 Children aged 7-12: Prefer CAB over bus")
                    policy["comfort"]["discomfort_per_min"]["cab"] = 0.05  # Even lower cab discomfort
                    policy["comfort"]["discomfort_per_min"]["bus"] = 0.4   # Higher bus discomfort
                else:
                    print(f"   🚌 Children aged {avg_child_age:.1f}: Prefer BUS over cab")
                    policy["comfort"]["discomfort_per_min"]["bus"] = 0.2   # Lower bus discomfort
                    policy["comfort"]["discomfort_per_min"]["cab"] = 0.2   # Higher cab discomfort
        
        print("   ✅ Family transport preferences: BUS/CAB preferred, walking discouraged")
    
    # Force solo traveler constraints
    elif any(word in query for word in ["solo", "backpacker", "individual", "alone"]):
        print("   🎒 SOLO TRAVELER DETECTED - Applying moderate constraints")
        
        # POI limits for solo travelers
        if has_elderly:
            policy["planning"]["max_pois_per_day"] = 3  # Limited for elderly solo
            policy["planning"]["min_pois_per_day"] = 1
            print("   ✅ Elderly solo POI limits: 1-3 POIs per day")
        else:
            policy["planning"]["max_pois_per_day"] = 4  # Moderate for solo travelers
            policy["planning"]["min_pois_per_day"] = 2
            print("   ✅ Solo POI limits: 2-4 POIs per day")
        
        # Walking constraints for solo travelers
        policy["planning"]["max_walking_distance_day_km"] = 20.0
        policy["planning"]["max_one_shot_walking_km"] = 3.0  # MAX 3km for solo
        policy["planning"]["walking_preference_threshold_km"] = 2.0
        
        print(f"   ✅ Solo walking limits: max_one_shot=3.0km, daily_max=20.0km, threshold=2.0km")
        
        # TIME CONSTRAINTS for solo travelers
        policy["planning"]["max_daily_total_time_hours"] = 10.0  # Longer day for solo travelers
        policy["planning"]["max_daily_travel_time_hours"] = 4.0  # More travel time allowed
        policy["planning"]["poi_visit_times"] = {
            "favorite_categories": 180,  # 3 hours for museums (can spend more time)
            "neutral_categories": 120,   # 2 hours for shopping, parks
            "disliked_categories": 90    # 1.5 hours for disliked activities
        }
        print("   ✅ Solo time limits: 10 hours total, 4 hours travel, longer POI visits")
        
        # Transport preferences for solo travelers
        if "comfort" not in policy:
            policy["comfort"] = {}
        
        policy["comfort"]["discomfort_per_min"] = {
            "walk": 0.8,      # Low walking discomfort for solo travelers
            "train": 0.6,     # Moderate train discomfort
            "bus": 0.5,       # Moderate bus discomfort
            "tram": 0.6,      # Moderate tram discomfort
            "cab": 0.4        # Moderate cab discomfort
        }
        
        print("   ✅ Solo transport preferences: Walking encouraged, all modes balanced")
    
    # Force elderly constraints
    elif any(word in query for word in ["elderly", "senior", "65+", "retirement", "old"]):
        print("   👴 ELDERLY DETECTED - Applying very strict constraints")
        
        # POI limits for elderly
        policy["planning"]["max_pois_per_day"] = 2  # Very limited for elderly
        policy["planning"]["min_pois_per_day"] = 1
        print("   ✅ Elderly POI limits: 1-2 POIs per day")
        
        # Walking constraints for elderly
        policy["planning"]["max_walking_distance_day_km"] = 5.0  # Very limited walking
        policy["planning"]["max_one_shot_walking_km"] = 1.0  # MAX 1km for elderly
        policy["planning"]["walking_preference_threshold_km"] = 0.3  # Prefer walking only up to 0.3km
        
        print(f"   ✅ Elderly walking limits: max_one_shot=1.0km, daily_max=5.0km, threshold=0.3km")
        
        # TIME CONSTRAINTS for elderly
        policy["planning"]["max_daily_total_time_hours"] = 6.0  # Shorter day for elderly
        policy["planning"]["max_daily_travel_time_hours"] = 2.0  # Very limited travel time
        policy["planning"]["poi_visit_times"] = {
            "favorite_categories": 75,   # 1.25 hours for museums (shorter attention)
            "neutral_categories": 60,    # 1 hour for shopping, parks
            "disliked_categories": 45    # 45 minutes for disliked activities
        }
        print("   ✅ Elderly time limits: 6 hours total, 2 hours travel, shorter POI visits")
        
        # Transport preferences for elderly
        if "comfort" not in policy:
            policy["comfort"] = {}
        
        policy["comfort"]["discomfort_per_min"] = {
            "walk": 4.0,      # Very high walking discomfort for elderly
            "train": 0.8,     # Moderate train discomfort
            "bus": 0.6,       # Moderate bus discomfort
            "tram": 0.7,      # Moderate tram discomfort
            "cab": 0.1        # Very low cab discomfort (preferred)
        }
        
        print("   ✅ Elderly transport preferences: CAB strongly preferred, walking discouraged")
    
    # Force couple/group constraints (default)
    else:
        print("   👥 GROUP/COUPLE DETECTED - Applying standard constraints")
        
        # POI limits for groups
        policy["planning"]["max_pois_per_day"] = 3  # Limited for groups
        policy["planning"]["min_pois_per_day"] = 2
        print("   ✅ Group POI limits: 2-3 POIs per day")
        
        # Walking constraints for groups
        policy["planning"]["max_walking_distance_day_km"] = 15.0  # Standard walking limits
        policy["planning"]["max_one_shot_walking_km"] = 2.0  # MAX 2km for groups
        policy["planning"]["walking_preference_threshold_km"] = 1.5
        
        print(f"   ✅ Group walking limits: max_one_shot=2.0km, daily_max=15.0km, threshold=1.5km")
        
        # TIME CONSTRAINTS for groups
        policy["planning"]["max_daily_total_time_hours"] = 8.0  # Standard day for groups
        policy["planning"]["max_daily_travel_time_hours"] = 3.0  # Standard travel time
        policy["planning"]["poi_visit_times"] = {
            "favorite_categories": 120,  # 2 hours for museums (standard)
            "neutral_categories": 90,    # 1.5 hours for shopping, parks
            "disliked_categories": 60    # 1 hour for disliked activities
        }
        print("   ✅ Group time limits: 8 hours total, 3 hours travel, standard POI visits")
        
        # Transport preferences for groups
        if "comfort" not in policy:
            policy["comfort"] = {}
        
        policy["comfort"]["discomfort_per_min"] = {
            "walk": 1.5,      # Standard walking discomfort
            "train": 0.6,     # Standard train discomfort
            "bus": 0.4,       # Standard bus discomfort
            "tram": 0.5,      # Standard tram discomfort
            "cab": 0.3        # Standard cab discomfort
        }
        
        print("   ✅ Group transport preferences: Balanced across all modes")
    
    print(f"   Final Planning Constraints after forcing: {policy['planning']}")
    print(f"   Final Transport Preferences after forcing: {policy['comfort']}")


# ----------------------------
# Public API
# ----------------------------
def synthesize_policy(policy_raw: Dict[str, Any],
                      profile: Optional[str] = None,
                      templates: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Synthesize a planner-ready policy from raw policy data.
    
    Flow:
    1. Load base policy template
    2. Apply user preferences overlay
    3. Use DeepSeek to generate personalized policy
    4. Apply persona-based rules
    5. Return final policy
    """
    print("\n" + "="*60)
    print("🔧 POLICY SYNTHESIS PIPELINE")
    print("="*60)
    
    print(f"📥 Inputs:")
    print(f"   - Policy Raw: {len(policy_raw)} keys")
    print(f"   - Profile: {profile or 'None'}")
    print(f"   - Templates: {len(templates) if templates else 0} keys" if templates else "   - Templates: None")
    
    # Step 1: Load base policy
    print(f"\n1️⃣ Loading base policy template...")
    base = _load_base_policy(profile)
    print(f"   ✅ Base policy loaded: {len(base)} keys")
    print(f"   🔑 Base policy sections: {list(base.keys())}")
    
    # Step 2: Apply user preferences overlay
    print(f"\n2️⃣ Applying user preferences overlay...")
    base_plus_overlay = _apply_user_preferences(base, policy_raw)
    if isinstance(templates, dict):
        _merge_dict(base_plus_overlay, deepcopy(templates))
    print(f"   ✅ Overlay applied: {len(base_plus_overlay)} keys")
    print(f"   🔄 Modified sections: {[k for k, v in base_plus_overlay.items() if k in base and v != base[k]]}")
    
    # Step 3: Use DeepSeek for intelligent policy generation
    print(f"\n3️⃣ DeepSeek-powered policy generation...")
    try:
        print("🎯 Attempting DeepSeek policy generation...")
        model_json = generate_policy_with_deepseek(
            policy_raw=policy_raw or {},
            base_policy=base_plus_overlay,
            template_overlay=templates if isinstance(templates, dict) else None
        )
        print("🎯 DEEPSEEK SUCCESS: Generated personalized policy")
        print("=" * 80)
        print("🤖 DEEPSEEK GENERATED CONTENT:")
        print(json.dumps(model_json, indent=2, ensure_ascii=False))
        print("=" * 80)
    except Exception as e:
        print(f"⚠️  WARNING: DeepSeek API failed ({e})")
        print("🔄 Falling back to enhanced rule-based synthesis")
        # Fallback to enhanced rule-based logic if DeepSeek fails
        model_json = _generate_intelligent_policy(policy_raw or {}, base_plus_overlay, templates)
        print("🔄 FALLBACK: Using enhanced rule-based policy generation")
    
    # Step 4: Apply persona-based rules
    print(f"\n4️⃣ Applying persona-based rules...")
    if model_json:
        print("3️⃣ DEEPSEEK AI MODIFICATIONS:")
        _log_ai_changes(base, model_json)
        
        # Merge AI modifications
        final_policy = _merge_dict(base_plus_overlay, model_json)
        print(f"   ✅ AI modifications merged: {len(final_policy)} keys")
    else:
        final_policy = base_plus_overlay
        print(f"   ⚠️  No AI modifications, using base + overlay")
    
    # Apply persona rules
    persona_flags = _infer_persona_flags(policy_raw)
    print(f"   🎭 Detected persona: {persona_flags}")
    
    final_policy = _apply_persona_rules(final_policy, persona_flags)
    print(f"   ✅ Persona rules applied")

    # Force persona-aware constraints
    _force_walking_constraints(final_policy, policy_raw)
    
    # Step 5: Finalize policy
    print(f"\n5️⃣ Finalizing policy...")
    final_policy["unique_id"] = f"policy_{profile or 'default'}_{hash(str(final_policy)) % 10000:04d}"
    final_policy["synthesis_timestamp"] = str(datetime.now())
    final_policy["synthesis_method"] = "deepseek_enhanced" if model_json else "rule_based"
    
    print(f"   ✅ Policy finalized:")
    print(f"      - ID: {final_policy['unique_id']}")
    print(f"      - Method: {final_policy['synthesis_method']}")
    print(f"      - Total keys: {len(final_policy)}")
    
    print(f"\n🎯 FINAL POLICY SUMMARY:")
    print(f"   - Budget: {final_policy.get('budget', {}).get('daily_cap', 'N/A')}")
    print(f"   - Stamina: {final_policy.get('stamina', {}).get('start', 'N/A')}")
    print(f"   - Max POIs: {final_policy.get('planning', {}).get('max_pois_per_day', 'N/A')}")
    print(f"   - Max Distance: {final_policy.get('planning', {}).get('max_total_distance_day_km', 'N/A')}")
    
    print(f"\n📋 COMPLETE FINAL POLICY:")
    print(json.dumps(final_policy, indent=2, ensure_ascii=False))
    
    print(f"\n✅ POLICY SYNTHESIS COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    return final_policy

# ----------------------------
# Manual test
# ----------------------------
if __name__ == "__main__":
    # Example: emulate what your orchestrator passes here
    demo_commonsense = {
        "__query": "3-day Manchester family trip with two kids (5 and 11). Prefer trams & short walks. Medium budget. Love museums and history.",
        "poi_preferences": {"museums": "high", "parks": "prefer", "shopping": "low", "historic": "high"},
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
