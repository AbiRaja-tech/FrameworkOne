# commonsense_agent.py
import os
import time
import logging
import argparse, json, re
from typing import List, Dict, Any, Tuple

from fewshot import EXAMPLES, CATEGORY_KEYWORDS

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

SYSTEM_PROMPT = """You are a travel-planning policy generator.
Given a natural-language trip brief, reply with a single, strict JSON object
containing the knobs/constraints that downstream planners use.
No prose. No code fences. JSON only.

The JSON shape should match the assistant examples provided in context.
"""

# ------------------------
# Logging / timing helpers
# ------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("commonsense")

def _now(): return time.perf_counter()
def _timed(label: str, t0: float) -> float:
    dt = _now() - t0
    log.info(f"{label} in {dt:.2f}s")
    return _now()

# ------------------------
# Routing helpers
# ------------------------
def score_category(query: str, keywords: List[str]) -> int:
    q = query.lower()
    return sum(1 for kw in keywords if kw in q)

def route_category(query: str, min_score: int = 1) -> List[str]:
    ranked = sorted(
        ((cat, score_category(query, kws)) for cat, kws in CATEGORY_KEYWORDS.items()),
        key=lambda x: x[1],
        reverse=True
    )
    top = [cat for cat, s in ranked if s >= min_score]
    return top or [cat for cat, _ in ranked[:3]]

def select_examples(categories: List[str], k_per_cat: int = 1) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    for cat in categories:
        ids = {e["id"] for e in EXAMPLES if e.get("category") == cat}
        pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
        for _id in ids:
            u = next((e for e in EXAMPLES if e["id"] == _id and e["role"] == "user"), None)
            a = next((e for e in EXAMPLES if e["id"] == _id and e["role"] == "assistant"), None)
            if u and a:
                pairs.append((u, a))
        for (u, a) in pairs[:k_per_cat]:
            selected.extend([u, a])
    return selected

# ------------------------
# Prompt builder
# ------------------------
def build_messages(user_request: str, fewshot_msgs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    msgs: List[Dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for m in fewshot_msgs:
        content = m["content"]
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)
        msgs.append({"role": m["role"], "content": content})
    msgs.append({"role": "user", "content": user_request})
    return msgs

# ------------------------
# Mock Model call (when ML dependencies unavailable)
# ------------------------
def call_deepseek(messages: List[Dict[str, str]], device_str: str = "cuda:0", max_new_tokens: int = 256) -> str:
    """
    Mock implementation that returns a predefined policy when ML dependencies are unavailable.
    """
    log.info("Using mock DeepSeek implementation (ML dependencies unavailable)")
    
    # Extract the user query from messages (find the LAST user message - the actual query)
    user_query = ""
    for msg in messages:
        if msg.get("role") == "user":
            user_query = msg.get("content", "")
    
    # Generate a simple policy based on the query
    user_query_lower = user_query.lower()
    
    # Determine the city from the query
    city = "London"  # default
    print(f"DEBUG: Checking for cities in: {user_query_lower}")
    if "manchester" in user_query_lower:
        city = "Manchester"
        print(f"DEBUG: Found Manchester!")
    elif "edinburgh" in user_query_lower:
        city = "Edinburgh"
        print(f"DEBUG: Found Edinburgh!")
    elif "york" in user_query_lower:
        city = "York"
        print(f"DEBUG: Found York!")
    elif "bath" in user_query_lower:
        city = "Bath"
        print(f"DEBUG: Found Bath!")
    elif "cardiff" in user_query_lower:
        city = "Cardiff"
        print(f"DEBUG: Found Cardiff!")
    elif "brighton" in user_query_lower:
        city = "Brighton"
        print(f"DEBUG: Found Brighton!")
    elif "london" in user_query_lower:
        city = "London"
        print(f"DEBUG: Found London!")
    print(f"DEBUG: Final city selected: {city}")
    
    # Determine transport preferences based on city
    if city == "London":
        transport = "walking_and_tube"
    elif city in ["Manchester", "Edinburgh", "York", "Bath", "Cardiff", "Brighton"]:
        transport = "walking_and_public_transport"
    else:
        transport = "walking"
    
    # Determine budget based on query keywords
    if any(word in user_query_lower for word in ["luxury", "high", "premium", "expensive"]):
        budget = "high"
        daily_budget = 300
    elif any(word in user_query_lower for word in ["cheap", "low", "economy"]):
        budget = "low"
        daily_budget = 80
    elif "medium" in user_query_lower:
        budget = "medium"
        daily_budget = 150
    else:
        budget = "medium"
        daily_budget = 150
    
    # Determine duration from query
    duration_days = 3  # default
    for i in range(1, 15):
        if f"{i}-day" in user_query_lower or f"{i} day" in user_query_lower:
            duration_days = i
            break
    
    return json.dumps({
        "cities": {
            city: {
                "description": f"City of {city}",
                "transport": transport,
                "budget": budget
            }
        },
        "hotel_location": {
            "description": city,
            "preference": "central_location"
        },
        "poi_preferences": {
            "museums": "high" if "museum" in user_query_lower else "medium",
            "parks": "high" if "park" in user_query_lower else "medium",
            "shopping": "high" if "shopping" in user_query_lower else "low"
        },
        "budget_constraints": {
            "daily_budget": daily_budget,
            "currency": "GBP"
        },
        "time_constraints": {
            "duration_days": duration_days,
            "start_date": "2024-01-01"
        }
    }, ensure_ascii=False)

# ------------------------
# JSON extraction
# ------------------------
def extract_json(raw_text: str) -> Dict[str, Any]:
    """
    Extracts JSON from raw model output.
    """
    try:
        # Try to find JSON in the text
        json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            return json.loads(json_str)
        else:
            # If no JSON found, try to parse the entire text
            return json.loads(raw_text)
    except json.JSONDecodeError as e:
        log.error(f"JSON decode error: {e}")
        log.error(f"Raw text: {raw_text}")
        raise ValueError(f"Failed to extract valid JSON: {e}")

# ------------------------
# CLI
# ------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Commonsense Agent: route queries to policy generators")
    ap.add_argument("--query", type=str, default="4-day London trip for a solo traveler who loves museums and parks",
                    help="User trip brief")
    ap.add_argument("--device", type=str, default="cuda:0", help="CUDA device (cuda:0, cuda:1, etc.)")
    args = ap.parse_args()

    # Test the pipeline
    categories = route_category(args.query)
    print(f"Categories: {categories}")
    
    examples = select_examples(categories, k_per_cat=1)
    print(f"Selected {len(examples)} examples")
    
    messages = build_messages(args.query, examples)
    print(f"Built {len(messages)} messages")
    
    raw = call_deepseek(messages, device_str=args.device)
    print(f"Raw output: {raw[:200]}...")
    
    try:
        policy = extract_json(raw)
        print("\nExtracted policy:")
        print(json.dumps(policy, indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"Failed to extract policy: {e}")