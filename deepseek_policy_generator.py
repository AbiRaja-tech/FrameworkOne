# deepseek_policy_generator.py
"""
DeepSeek-powered policy generator for travel planning using Ollama.

This module generates personalized travel policies by calling DeepSeek through Ollama.
It takes user preferences, base policy templates, and generates customized policies.
"""

import json
import logging
import re
from typing import Dict, Any, Optional
from copy import deepcopy

# Configure logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("deepseek_policy_generator")

# Import Ollama client
try:
    from ollama_client import create_deepseek_client
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    log.warning("Ollama client not available. Install with: pip install requests")

def _extract_json(text: str) -> Dict[str, Any]:
    """Extract JSON object from model text."""
    if not isinstance(text, str):
        raise ValueError("Response is not a string.")
    
    # Clean the text
    text = text.strip()
    
    # Remove code fences if present
    text = re.sub(r'^```(?:json)?\s*', '', text)
    text = re.sub(r'\s*```\s*$', '', text)
    
    # Find JSON object
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        json_str = json_match.group(0)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
    
    # Try to parse the entire text
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        log.error(f"Failed to extract JSON: {e}")
        log.error(f"Raw text: {text}")
        raise ValueError(f"Failed to extract valid JSON: {e}")

def _build_deepseek_prompt(policy_raw: Dict[str, Any], 
                           base_policy: Dict[str, Any],
                           template_overlay: Optional[Dict[str, Any]] = None) -> str:
    """
    Build a comprehensive prompt for DeepSeek to generate a personalized travel policy.
    """
    
    # Extract key information from the query and commonsense agent output
    query = policy_raw.get("__query", "")
    ages = policy_raw.get("ages", [])
    group_type = policy_raw.get("group_type", "")
    budget_level = policy_raw.get("budget_level", "")
    transport_prefs = policy_raw.get("transport_preferences", {})
    activity_prefs = policy_raw.get("activity_preferences", {})
    
    # Extract commonsense agent data
    cities_info = policy_raw.get("cities", {})
    budget_constraints = policy_raw.get("budget_constraints", {})
    time_constraints = policy_raw.get("time_constraints", {})
    poi_preferences = policy_raw.get("poi_preferences", {})
    
    # Build the prompt
    prompt = f"""You are an expert travel policy generator. Create a personalized travel policy JSON based on this user query and commonsense agent analysis:

USER QUERY: {query}

COMMONSENSE AGENT ANALYSIS:
- Cities: {json.dumps(cities_info, indent=2) if cities_info else 'Not specified'}
- Budget Constraints: {json.dumps(budget_constraints, indent=2) if budget_constraints else 'Not specified'}
- Time Constraints: {json.dumps(time_constraints, indent=2) if time_constraints else 'Not specified'}
- POI Preferences: {json.dumps(poi_preferences, indent=2) if poi_preferences else 'Not specified'}

USER DETAILS:
- Ages: {ages if ages else 'Not specified'}
- Group Type: {group_type if group_type else 'Not specified'}
- Budget Level: {budget_level if budget_level else 'Not specified'}
- Transport Preferences: {json.dumps(transport_prefs, indent=2) if transport_prefs else 'Not specified'}
- Activity Preferences: {json.dumps(activity_prefs, indent=2) if activity_prefs else 'Not specified'}

PERSONA DETECTION (CRITICAL):
Based on the query "{query}", this is a:
- FAMILY WITH CHILDREN (if query contains: family, children, kids, child, toddler, teen)
- SOLO TRAVELER (if query contains: solo, backpacker, individual)
- ELDERLY (if query contains: elderly, senior, 65+, retirement)
- COUPLE (if query contains: couple, romantic, honeymoon)
- BUSINESS (if query contains: business, work, meeting, conference)
- ACTIVE ADVENTURER (if query contains: adventure, outdoor, sports, hiking)
- COMFORT SEEKER (if query contains: comfortable, luxury, premium, relaxing)
- CULTURAL ENTHUSIAST (if query contains: museum, historic, cultural, arts)

BASE POLICY TEMPLATE:
{json.dumps(base_policy, indent=2)}

TEMPLATE OVERLAY (if provided):
{json.dumps(template_overlay, indent=2) if template_overlay else 'None'}

INSTRUCTIONS:
1. Analyze the user's query, commonsense agent output, and preferences carefully
2. Adapt the base policy values based on user needs and characteristics
3. Consider factors like age, group type, budget level, transport preferences, activity interests
4. Respect the template overlay if provided
5. Output ONLY a valid JSON policy with these exact sections: budget, stamina, planning, preferences, comfort, solver
6. Do NOT include cities, hotels, or other non-policy data
7. Focus on constraint values that will guide the travel planner
8. Make the policy truly personalized based on the user's specific needs

WALKING CONSTRAINTS & STAMINA (CRITICAL - MUST FOLLOW):
- Set realistic walking limits based on persona and group type
- Adjust walking_stamina_cost_per_km based on fitness level and age
- Set max_walking_distance_day_km based on group capabilities
- Set max_one_shot_walking_km for single walking segments
- Use walking_preference_threshold_km to balance walking vs transport

CRITICAL WALKING LIMITS (MUST RESPECT):
- Solo travelers: max_one_shot_walking_km = 3.0 MAX
- Couples/Groups: max_one_shot_walking_km = 2.0 MAX  
- Families with children: max_one_shot_walking_km = 2.0 MAX
- Elderly: max_one_shot_walking_km = 1.5 MAX
- Active adventurers: max_one_shot_walking_km = 3.0 MAX
- Comfort seekers: max_one_shot_walking_km = 2.0 MAX

WALKING PREFERENCE THRESHOLDS (MUST RESPECT):
- Solo travelers: walking_preference_threshold_km = 2.0
- Couples/Groups: walking_preference_threshold_km = 1.5
- Families with children: walking_preference_threshold_km = 1.0
- Elderly: walking_preference_threshold_km = 0.5
- Active adventurers: walking_preference_threshold_km = 2.5
- Comfort seekers: walking_preference_threshold_km = 0.8

DAILY WALKING LIMITS (MUST RESPECT):
- Solo travelers: max_walking_distance_day_km = 20-25
- Couples/Groups: max_walking_distance_day_km = 15-20
- Families with children: max_walking_distance_day_km = 10-15
- Elderly: max_walking_distance_day_km = 5-10
- Active adventurers: max_walking_distance_day_km = 25-35

TIME CONSTRAINTS & POI VISIT TIMES (CRITICAL - MUST FOLLOW):
- Set realistic daily time limits based on persona and group type
- Adjust POI visit times based on user preferences and interests
- Consider attention spans and stamina for different age groups

DAILY TIME LIMITS (MUST RESPECT):
- Solo travelers: max_daily_total_time_hours = 10-12 (longer day, more POIs)
- Couples/Groups: max_daily_total_time_hours = 8-10 (standard day)
- Families with children: max_daily_total_time_hours = 6-8 (shorter attention spans)
- Elderly: max_daily_total_time_hours = 6-8 (lower stamina)
- Active adventurers: max_daily_total_time_hours = 10-12 (high energy)
- Comfort seekers: max_daily_total_time_hours = 7-9 (balanced)

TRAVEL TIME LIMITS (MUST RESPECT):
- Solo travelers: max_daily_travel_time_hours = 4-5 (more travel allowed)
- Couples/Groups: max_daily_travel_time_hours = 3-4 (standard travel)
- Families with children: max_daily_travel_time_hours = 2-3 (limited travel)
- Elderly: max_daily_travel_time_hours = 2-3 (minimal travel)
- Active adventurers: max_daily_travel_time_hours = 4-5 (more travel)
- Comfort seekers: max_daily_travel_time_hours = 2-3 (minimal travel)

POI VISIT TIMES (MUST RESPECT):
- Favorite categories (museums, historic, cultural, arts): 120-180 minutes
- Neutral categories (shopping, parks, entertainment): 90-120 minutes  
- Disliked categories (if any): 60-90 minutes

PERSONA-SPECIFIC POI VISIT TIMES:
- Solo travelers: favorite=180min, neutral=120min, disliked=90min
- Couples/Groups: favorite=120min, neutral=90min, disliked=60min
- Families with children: favorite=90min, neutral=60min, disliked=45min
- Elderly: favorite=75min, neutral=60min, disliked=45min
- Active adventurers: favorite=150min, neutral=120min, disliked=90min
- Comfort seekers: favorite=100min, neutral=75min, disliked=60min

PERSONALIZATION GUIDELINES:
- Budget-conscious travelers: Reduce daily_cap by 20-40%, emphasize walking/public transport, but set realistic walking limits
- Luxury travelers: Increase daily_cap by 50-100%, prefer premium transport, higher walking stamina costs
- Active travelers: Increase stamina, more POIs per day, longer distances, lower walking stamina costs
- Family trips: Moderate constraints, family-friendly transport preferences, lower walking stamina costs for children
- Cultural enthusiasts: Prioritize museums/historic sites, adjust preferences accordingly
- Solo backpackers: More POIs per day, longer distances, flexible planning, moderate walking stamina costs
- Elderly travelers: Fewer POIs (~3), short distances, increase transfer penalty, higher walking stamina costs
- Children/Teens: Lower walking stamina costs, shorter max walking distances, more POI variety

STAMINA REASONING:
- Young adults (18-35): walking_stamina_cost_per_km = 1.5-2.0
- Middle-aged (36-55): walking_stamina_cost_per_km = 2.0-2.5
- Elderly (65+): walking_stamina_cost_per_km = 3.0-4.0
- Children (under 12): walking_stamina_cost_per_km = 1.0-1.5
- Active/fit: Reduce stamina costs by 20-30%
- Sedentary: Increase stamina costs by 20-30%

WALKING DISTANCE REASONING:
- Solo backpackers: max_walking_distance_day_km = 20-30, max_one_shot_walking_km = 8-10
- Families with children: max_walking_distance_day_km = 8-15, max_one_shot_walking_km = 3-5
- Elderly: max_walking_distance_day_km = 5-10, max_one_shot_walking_km = 2-3
- Active adventurers: max_walking_distance_day_km = 25-40, max_one_shot_walking_km = 10-15
- Comfort seekers: max_walking_distance_day_km = 10-20, max_one_shot_walking_km = 4-6

EXAMPLE OUTPUT FORMAT:
{{
  "budget": {{
    "daily_cap": 80.0,
    "mode_fixed_cost": {{
      "walk": 0.0,
      "train": 1.0,
      "bus": 2.0,
      "tram": 0.0,
      "cab": 8.0
    }}
  }},
  "stamina": {{
    "start": 12.0,
    "poi_visit_cost": 1.2,
    "walking_stamina_cost_per_km": 1.8
  }},
  "planning": {{
    "max_pois_per_day": 8,
    "max_total_distance_day_km": 60.0,
    "max_walking_distance_day_km": 25.0,
    "max_one_shot_walking_km": 8.0,
    "walking_preference_threshold_km": 2.5
  }},
  "preferences": {{
    "must_include": ["adventure", "outdoor", "cultural"]
  }},
  "comfort": {{
    "discomfort_per_min": {{
      "walk": 0.8,
      "tram": 0.4
    }}
  }},
  "solver": {{
    "max_seconds": 10.0,
    "workers": 8
  }}
}}

IMPORTANT: Return ONLY the JSON object, no explanations or additional text. Use the commonsense agent data to inform your decisions."""

    return prompt

def generate_policy_with_deepseek(policy_raw: Dict[str, Any],
                                 base_policy: Dict[str, Any],
                                 template_overlay: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Generate a personalized travel policy using DeepSeek through Ollama.
    
    Args:
        policy_raw: Raw policy data from commonsense agent
        base_policy: Base policy template
        template_overlay: Optional template overlay
        
    Returns:
        Generated personalized policy
    """
    
    if not OLLAMA_AVAILABLE:
        raise RuntimeError("Ollama client not available. Install required dependencies.")
    
    try:
        # Create DeepSeek client
        client = create_deepseek_client()
        
        # Test connection
        if not client.test_connection():
            raise RuntimeError("Cannot connect to Ollama server. Make sure Ollama is running on localhost:11434")
        
        log.info("✅ Ollama connection successful!")
        
        # Build the prompt
        prompt = _build_deepseek_prompt(policy_raw, base_policy, template_overlay)
        
        log.info("🎯 Calling DeepSeek through Ollama for policy generation...")
        log.info("=" * 80)
        
        # Generate policy using DeepSeek
        response = client.generate(
            prompt=prompt,
            system="You are an expert travel policy generator. Always respond with valid JSON only.",
            temperature=0.3,  # Low temperature for consistent policy generation
            max_tokens=2048
        )
        
        log.info("📥 DeepSeek response received:")
        log.info(f"Response length: {len(response)} characters")
        log.info(f"Response preview: {response[:200]}...")
        
        # Extract JSON from response
        try:
            policy_json = _extract_json(response)
            log.info("✅ JSON extraction successful!")
            
            # Validate the policy structure
            required_sections = ["budget", "stamina", "planning", "preferences", "comfort", "solver"]
            missing_sections = [sec for sec in required_sections if sec not in policy_json]
            
            if missing_sections:
                log.warning(f"⚠️ Missing sections in generated policy: {missing_sections}")
                # Fill missing sections from base policy
                for section in missing_sections:
                    if section in base_policy:
                        policy_json[section] = deepcopy(base_policy[section])
                        log.info(f"🔧 Filled missing section '{section}' from base policy")
            
            log.info("🎯 DEEPSEEK SUCCESS: Generated personalized policy")
            log.info("=" * 80)
            log.info("🤖 DEEPSEEK GENERATED CONTENT:")
            log.info(json.dumps(policy_json, indent=2, ensure_ascii=False))
            log.info("=" * 80)
            
            return policy_json
            
        except Exception as e:
            log.error(f"❌ JSON extraction failed: {e}")
            log.error(f"Raw response: {response}")
            raise RuntimeError(f"Failed to extract valid JSON from DeepSeek response: {e}")
            
    except Exception as e:
        log.error(f"❌ DeepSeek policy generation failed: {e}")
        raise RuntimeError(f"DeepSeek policy generation failed: {e}")

def test_deepseek_integration():
    """Test the DeepSeek integration."""
    
    # Test data
    test_policy_raw = {
        "__query": "1-day Manchester trip for a solo 25-year-old backpacker, loves adventure and outdoor activities, tight budget, prefers walking and public transport.",
        "ages": [25],
        "group_type": "solo",
        "budget_level": "tight",
        "transport_preferences": {"preferred_mode": "walking"},
        "activity_preferences": {"interests": ["adventure", "outdoor", "sports"]}
    }
    
    test_base_policy = {
        "budget": {"daily_cap": 150.0},
        "stamina": {"start": 10.0, "poi_visit_cost": 1.5},
        "planning": {"max_pois_per_day": 6, "max_total_distance_day_km": 30.0},
        "preferences": {"must_include": [], "must_avoid": []},
        "comfort": {"discomfort_per_min": {"walk": 1.0}},
        "solver": {"max_seconds": 10.0, "workers": 8}
    }
    
    try:
        log.info("🧪 Testing DeepSeek integration...")
        result = generate_policy_with_deepseek(test_policy_raw, test_base_policy)
        log.info("✅ Test successful!")
        return result
    except Exception as e:
        log.error(f"❌ Test failed: {e}")
        return None

if __name__ == "__main__":
    # Test the integration
    print("🧪 Testing DeepSeek integration...")
    print("Note: This is a test function. For actual use, run the orchestrator.")
    
    # Uncomment the lines below to test locally
    # result = test_deepseek_integration()
    # if result:
    #     print("\n🎯 FINAL TEST RESULT:")
    #     print(json.dumps(result, indent=2, ensure_ascii=False))
    # else:
    #     print("\n❌ Test failed. Check the logs above for details.")
