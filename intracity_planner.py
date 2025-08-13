# intracity_planner.py
#
# Intracity CP-SAT itinerary planner (single city, single day per run, multi-hotel)
# with your corrected rules and strict priority objective, now with MTZ subtour elimination
# and a hard minimum-POI constraint.
#
# PRIORITY (lexicographic via big weights):
#   1) Budget (cheaper is better)
#   2) Stamina (higher remaining is better, never negative)
#   3) POI score (sum of ratings)
#   4) Experience score (must_include + variety bonus − must_avoid)
#   5) Comfort score (more comfortable is better)
#
# Stamina rules (your spec):
#   Start=10; -1/60min walk; -1/180min cab; -1/120min inside POIs; +2 per meal POI; 0<=S<=Smax
#
# Comfort score:
#   We treat "discomfort per minute" (walk highest, cab lowest, bus moderate, tram slightly > bus)
#   comfort_score = - Σ(discomfort_per_min[mode] * minutes_in_mode)
#
# Experience score:
#   +5  for POIs in must_include categories
#   -1  for POIs (category/name match) in must_avoid
#   +3  for each category that appears ≥ 3 times among selected
#
# Budget:
#   daily_cap (default 120) includes hotel + travel (per-minute + fixed/leg) + tickets
#
# This variant still optimizes selection/order/modes, but no clocked opening hours.
# (We can add time windows back later if you want.)
#
# Usage:
#   python intracity_planner.py
#   LOG_LEVEL=DEBUG python intracity_planner.py --data city.json --policy policy.json --cp-verbose

import os
import json
import math
import time
import argparse
import logging
from typing import Dict, Any, List, Optional, Tuple
from ortools.sat.python import cp_model

# ---------- Logging ----------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("intracity_planner")


# ---------- Helpers ----------
def haversine_km(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    R = 6371.0088
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return 2 * R * math.asin(math.sqrt(h))

def time_from_km(distance_km: float, kmph: float) -> int:
    return int(round(60.0 * distance_km / max(1e-6, kmph)))


# ---------- Defaults / Loading ----------
def default_policy() -> Dict[str, Any]:
    return {
        # Costs (GBP)
        "budget": {
            "daily_cap": 250.0,
            "mode_cost_per_min": {"walk": 0.0, "bus": 0.25, "tram": 0.35, "cab": 0.9},
            "mode_fixed_cost": {"walk": 0.0, "bus": 0.0, "tram": 0.0, "cab": 1.0},
        },
        # Stamina rules  (Start=10; drains & gains per your spec)
        "stamina": {
            "start": 10.0,
            "max": 12.0,
            "walk_hr_cost": 1.0,     # -1 per 1 hr walking
            "cab_hr_cost": 1.0/3.0,  # -1 per 3 hrs cab
            "poi_2hr_cost": 0.5,     # -1 per 2 hrs inside POIs => 0.5 per hour
            "meal_gain": 2.0         # +2 per meal POI
        },
        # Discomfort per min (lower = more comfortable; comfort_score = - Σ(discomfort * min))
        "comfort_discomfort_per_min": {"walk": 1.0, "tram": 0.6, "bus": 0.5, "cab": 0.2},
        # Speeds
        "transport_speeds_kmph": {"walk": 4.5, "bus": 14.0, "tram": 18.0, "cab": 22.0},
        # POI prefs
        "must_include": ["Museum", "Heritage"],
        "must_avoid": ["theme_parks", "Noisy bar"],
        # Priority weights (lexicographic by magnitude)
        "priority_weights": {
            "budget": 10_000_000,
            "stamina": 50_000,
            "poi_score": 5_000,
            "experience": 1_000,
            "comfort": 100,
        },
        # Selection limits and geometry
        "max_pois_per_day": 6,
        "min_pois_per_day": 3,
        "max_one_way_distance_from_hotel_km": 6.0,
        "max_total_distance_day_km": 18.0,
        # Solver
        "solver_max_seconds": 10.0,
        "solver_workers": 8
    }

def load_data(path: Optional[str]) -> Dict[str, Any]:
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    # Built-in sample
    pois = [
        {"name": "British Museum", "category": "Museum",   "rating": 4.8, "duration_min": 120, "price": 0.0,  "latitude": 51.5194, "longitude": -0.1270, "is_meal": False},
        {"name": "Tower of London","category": "Heritage", "rating": 4.7, "duration_min": 120, "price": 34.0, "latitude": 51.5081, "longitude": -0.0759, "is_meal": False},
        {"name": "Borough Market","category": "Food",      "rating": 4.3, "duration_min": 60,  "price": 0.0,  "latitude": 51.5054, "longitude": -0.0911, "is_meal": True},
        {"name": "Covent Garden", "category": "Shopping",  "rating": 4.4, "duration_min": 60,  "price": 0.0,  "latitude": 51.5123, "longitude": -0.1223, "is_meal": False},
        {"name": "National Gallery","category":"Museum",   "rating": 4.6, "duration_min": 90,  "price": 0.0,  "latitude": 51.5089, "longitude": -0.1283, "is_meal": False},
        {"name": "St Paul’s Cathedral","category":"Heritage","rating":4.6,"duration_min": 75,  "price": 21.0, "latitude": 51.5138, "longitude": -0.0984, "is_meal": False},
        {"name": "Hyde Park","category":"Park",            "rating": 4.6, "duration_min": 60,  "price": 0.0,  "latitude": 51.5073, "longitude": -0.1657, "is_meal": False},
        {"name": "Theme Park (avoid)","category":"theme_parks","rating": 4.2, "duration_min": 120, "price": 40.0, "latitude": 51.4051, "longitude": -0.5126, "is_meal": False}
    ]
    hotels = [
        {"name": "Hotel Bloomsbury", "price_per_night": 95.0, "user_rating": 4.4, "latitude": 51.5200, "longitude": -0.1250},
        {"name": "Hotel South Bank", "price_per_night":110.0, "user_rating": 4.6, "latitude": 51.5065, "longitude": -0.1198}
    ]

    N = len(pois)
    dist = [[0.0]*N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            if i != j:
                dist[i][j] = haversine_km((pois[i]["latitude"], pois[i]["longitude"]),
                                          (pois[j]["latitude"], pois[j]["longitude"]))

    return {"hotels": hotels, "pois": pois, "poi_poi_distance_km": dist}

def load_policy(path: Optional[str]) -> Dict[str, Any]:
    base = default_policy()
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            override = json.load(f)
        for k, v in override.items():
            if isinstance(v, dict) and isinstance(base.get(k), dict):
                base[k].update(v)
            else:
                base[k] = v

    # If priority_order is specified, generate weights automatically
    if "priority_order" in base:
        order = base["priority_order"]
        # Start from a big base to preserve lexicographic dominance
        base_multiplier = 10 ** 6
        weights = {}
        current_weight = base_multiplier ** (len(order))
        # Assign descending weights to preserve order importance
        for idx, key in enumerate(order):
            weights[key] = base_multiplier ** (len(order) - idx)
        base["priority_weights"] = weights
        log.info(f"Priority weights auto-generated from order {order}: {weights}")

    return base



# ---------- Core single-day, single-hotel solver ----------
def solve_intracity_day_for_hotel(
    hotel: Dict[str, Any],
    pois: List[Dict[str, Any]],
    poi_poi_distance_km: List[List[float]],
    policy: Dict[str, Any],
    cp_verbose: bool = False,
) -> Dict[str, Any]:
    hname = hotel.get("name", "hotel")
    log.info(f"[{hname}] Optimize intracity day (CP-SAT)")

    # Unpack policy
    budget = policy["budget"]
    stamina = policy["stamina"]
    speeds = policy["transport_speeds_kmph"]
    discomfort = policy["comfort_discomfort_per_min"]
    must_include = [s.lower() for s in policy.get("must_include", [])]
    must_avoid   = [s.lower() for s in policy.get("must_avoid",   [])]
    weights = policy["priority_weights"]

    max_pois = int(policy.get("max_pois_per_day", 6))
    min_pois = int(policy.get("min_pois_per_day", 3))
    radius_km = float(policy.get("max_one_way_distance_from_hotel_km", 6.0))
    max_total_km_day = float(policy.get("max_total_distance_day_km", 18.0))

    # Build extended graph: 0 = hotel(start), 1..N = POIs, N+1 = end
    N = len(pois)
    hotel_pos = (float(hotel["latitude"]), float(hotel["longitude"]))
    hotel_to_poi = [haversine_km(hotel_pos, (p["latitude"], p["longitude"])) for p in pois]

    dist_ext = [[0.0]*(N+2) for _ in range(N+2)]
    for i in range(N):
        dist_ext[0][i+1] = hotel_to_poi[i]
        dist_ext[i+1][0] = hotel_to_poi[i]
    for i in range(N):
        for j in range(N):
            dist_ext[i+1][j+1] = poi_poi_distance_km[i][j]
    # End node: distance back to hotel
    for i in range(N+1):
        dist_ext[i][N+1] = dist_ext[i][0]

    # Travel time matrices per mode (in minutes)
    TT = {m: [[0]*(N+2) for _ in range(N+2)] for m in speeds}
    for m, kmph in speeds.items():
        for i in range(N+2):
            for j in range(N+2):
                if i != j:
                    TT[m][i][j] = time_from_km(dist_ext[i][j], kmph)

    # Candidate filter
    candidate = []
    poi_price = [0.0]*(N+2)
    poi_dur_min = [0]*(N+2)
    poi_is_meal = [0]*(N+2)
    poi_cat     = [""]*(N+2)
    poi_name    = [""]*(N+2)

    for idx, p in enumerate(pois, start=1):
        name = (p.get("name") or "").lower()
        cat  = (p.get("category") or "").lower()
        poi_name[idx] = name
        poi_cat[idx] = cat

        if dist_ext[0][idx] > radius_km:
            continue
        candidate.append(idx)
        poi_price[idx] = float(p.get("price", 0.0) or 0.0)
        poi_dur_min[idx] = int(p.get("duration_min", 60))
        poi_is_meal[idx] = 1 if p.get("is_meal", False) else 0

    if not candidate:
        return {"status": "infeasible", "hotel": hname, "reason": "no candidate POIs in radius"}

    # ---------- CP model ----------
    model = cp_model.CpModel()
    start, end = 0, N+1

    # Decision variables
    y = {(i,j): model.NewBoolVar(f"y_{i}_{j}") for i in range(N+2) for j in range(N+2) if i != j}
    x = {i: model.NewBoolVar(f"x_{i}") for i in candidate}
    mvar = {(i,j,m): model.NewBoolVar(f"m_{i}_{j}_{m}") for i in range(N+2) for j in range(N+2) if i != j for m in speeds}

    # ---------- Flow constraints ----------
    # Start: exactly one outgoing
    model.Add(sum(y[(start,j)] for j in range(N+2) if j != start) == 1)
    # End: exactly one incoming
    model.Add(sum(y[(i,end)] for i in range(N+2) if i != end) == 1)
    # Every visited POI has exactly one in and one out
    for k in candidate:
        model.Add(sum(y[(i,k)] for i in range(N+2) if i != k) == x[k])
        model.Add(sum(y[(k,j)] for j in range(N+2) if j != k) == x[k])
    # Disable edges into start and out of end
    for i in range(N+2):
        if (i,start) in y: model.Add(y[(i,start)] == 0)
        if (end,i) in y:   model.Add(y[(end,i)] == 0)

    # One mode iff an arc is used
    for i in range(N+2):
        for j in range(N+2):
            if i == j: continue
            model.Add(sum(mvar[(i,j,m)] for m in speeds) == y[(i,j)])

    # Visit limits
    model.Add(sum(x[i] for i in candidate) <= max_pois)
    model.Add(sum(x[i] for i in candidate) >= min_pois)

    # ---------- Subtour elimination (MTZ) ----------
    # Order variables u_i (0..|candidate|). u_start=0. For unselected, u_i must be 0.
    u = {i: model.NewIntVar(0, len(candidate), f"u_{i}") for i in range(N+2)}
    model.Add(u[start] == 0)
    # Link u_i with selection x_i (for POIs only)
    for i in candidate:
        model.Add(u[i] >= x[i])                               # if selected, u_i >= 1
        model.Add(u[i] <= len(candidate) * x[i])              # if not selected, u_i == 0
    
    # MTZ constraints for proper tour structure
    for i in candidate:
        for j in candidate:
            if i == j:
                continue
            model.Add(u[i] - u[j] + (len(candidate)) * y[(i,j)] <= len(candidate) - 1)

    # ---------- Distance cap ----------
    model.Add(
        sum(int(round(dist_ext[i][j]*100))*y[(i,j)] for i in range(N+2) for j in range(N+2) if i != j
        ) <= int(round(max_total_km_day*100))
    )

    # ---------- Budget ----------
    travel_cost_cents_terms = []
    for i in range(N+2):
        for j in range(N+2):
            if i == j: continue
            for m in speeds:
                tmin = TT[m][i][j]
                per_min = float(budget["mode_cost_per_min"].get(m, 0.0))
                fixed = float(budget["mode_fixed_cost"].get(m, 0.0))
                cost_cents = int(round(100.0 * (per_min * tmin + fixed)))
                travel_cost_cents_terms.append(cost_cents * mvar[(i,j,m)])

    ticket_cost_cents = sum(int(round(100.0 * poi_price[i])) * x[i] for i in candidate)
    hotel_cost_cents = int(round(100.0 * float(hotel.get("price_per_night", 0.0))))
    daily_spend_cents = sum(travel_cost_cents_terms) + ticket_cost_cents + hotel_cost_cents
    model.Add(daily_spend_cents <= int(round(100.0 * float(budget["daily_cap"]))))
    
    # Temporary simple budget constraint
    # daily_spend_cents = 0  # No budget constraint for now

    # ---------- Stamina ----------
    # Drain: walk 1hr->1, cab 3hr->1, POI 2hr->1 ; Gain: +2 per meal
    start_stamina = float(stamina.get("start", 10.0))
    max_stamina = float(stamina.get("max", 12.0))
    walk_hr_cost = float(stamina.get("walk_hr_cost", 1.0))
    cab_hr_cost  = float(stamina.get("cab_hr_cost", 1.0/3.0))
    poi_2hr_cost = float(stamina.get("poi_2hr_cost", 0.5))
    meal_gain    = float(stamina.get("meal_gain", 2.0))

    walk_minutes = sum(TT["walk"][i][j] * mvar[(i,j,"walk")] for i in range(N+2) for j in range(N+2) if i != j) if "walk" in speeds else 0
    cab_minutes  = sum(TT["cab"][i][j]  * mvar[(i,j,"cab")]  for i in range(N+2) for j in range(N+2) if i != j) if "cab"  in speeds else 0
    poi_minutes  = sum(poi_dur_min[i] * x[i] for i in candidate)
    meals_count  = sum(x[i] for i in candidate if poi_is_meal[i] == 1)

    SCALE = 100
    S0_i   = int(round(SCALE * start_stamina))
    Smax_i = int(round(SCALE * max_stamina))
    walk_drain_i = int(round(SCALE * (walk_hr_cost/60.0)))
    cab_drain_i  = int(round(SCALE * (cab_hr_cost/60.0)))
    poi_drain_i  = int(round(SCALE * (poi_2hr_cost/120.0)))
    meal_gain_i  = int(round(SCALE * meal_gain))

    stamina_end = model.NewIntVar(0, Smax_i, "stamina_end")
    model.Add(
        stamina_end
        == S0_i
           - walk_drain_i * walk_minutes
           - cab_drain_i  * cab_minutes
           - poi_drain_i  * poi_minutes
           + meal_gain_i  * meals_count
    )

    # ---------- POI score ----------
    poi_score = sum(int(100 * float(pois[i-1].get("rating", 1.0))) * x[i] for i in candidate)

    # ---------- Experience score ----------
    include_bonus = 5
    avoid_penalty = 1
    variety_bonus = 3

    include_term = sum(
        include_bonus * x[i]
        for i in candidate
        if any(kw in poi_cat[i] for kw in must_include)
    )
    avoid_term = sum(
        avoid_penalty * x[i]
        for i in candidate
        if any(kw in poi_cat[i] or kw in poi_name[i] for kw in must_avoid)
    )

    cats = sorted({poi_cat[i] for i in candidate})
    cat_count = {c: model.NewIntVar(0, len(candidate), f"cat_count_{c}") for c in cats}
    for c in cats:
        indices = [i for i in candidate if poi_cat[i] == c]
        if indices:
            model.Add(cat_count[c] == sum(x[i] for i in indices))
        else:
            model.Add(cat_count[c] == 0)
    bonus_flag = {}
    for c in cats:
        b = model.NewBoolVar(f"variety_bonus_{c}")
        model.Add(cat_count[c] >= 3).OnlyEnforceIf(b)
        model.Add(cat_count[c] <= 2).OnlyEnforceIf(b.Not())
        bonus_flag[c] = b

    variety_term = variety_bonus * sum(bonus_flag[c] for c in cats)
    experience_score = 100 * (include_term - avoid_term + variety_term)

    # ---------- Comfort score ----------
    comfort_score = 0
    for m in speeds:
        per_min_dis = float(discomfort.get(m, 0.0))
        minutes_sum = sum(TT[m][i][j] * mvar[(i,j,m)] for i in range(N+2) for j in range(N+2) if i != j)
        comfort_score += - int(round(100 * per_min_dis)) * minutes_sum

    # ---------- Objective ----------
    # Multi-objective optimization with reasonable weights
    objective = (
        - 1000 * daily_spend_cents      # Budget: minimize cost (high priority)
        + 100 * stamina_end             # Stamina: maximize remaining stamina (high priority)
        + 10 * poi_score                # POI score: maximize ratings (medium priority)
        + 5 * experience_score          # Experience: maximize bonuses (medium priority)
        + 1 * comfort_score             # Comfort: minimize discomfort (low priority)
    )
    model.Maximize(objective)

    # ---------- Solve ----------
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(policy.get("solver_max_seconds", 10.0))
    solver.parameters.num_search_workers = int(policy.get("solver_workers", 8))
    solver.parameters.log_to_stdout = bool(cp_verbose)

    t0 = time.perf_counter()
    status = solver.Solve(model)
    solve_s = time.perf_counter() - t0
    log.info(f"[{hname}] Status={solver.StatusName(status)} solve_time={solve_s:.2f}s")

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return {"status": "infeasible", "hotel": hname}

    # ---------- Extract the path from start to end ----------
    # Build successor map
    succ = {i: None for i in range(N+2)}
    for i in range(N+2):
        for j in range(N+2):
            if i != j and solver.Value(y[(i,j)]) == 1:
                succ[i] = j
                break

    # Walk from start
    path = []
    cur = start
    seen = set([cur])
    steps = 0
    while cur != end and steps <= N+1:
        nxt = succ[cur]
        if nxt is None or nxt in seen:
            break
        if nxt not in (start, end):
            chosen_mode = None
            for m in speeds:
                if solver.Value(mvar[(cur,nxt,m)]) == 1:
                    chosen_mode = m
                    break
            p = pois[nxt-1]
            path.append({
                "poi": p["name"],
                "category": p["category"],
                "arrive_via": chosen_mode
            })
        seen.add(nxt)
        cur = nxt
        steps += 1

    # Diagnostic: selected count vs extracted path length
    selected_count = int(sum(solver.Value(x[i]) for i in candidate))
    if len(path) != selected_count:
        log.warning(
            f"[{hname}] Extracted path length ({len(path)}) != selected_count ({selected_count}). "
            "This should not happen with MTZ; please report if you see it."
        )

    spend_gbp = solver.Value(daily_spend_cents) / 100.0
    return {
        "status": "ok",
        "hotel": hname,
        "objective": solver.ObjectiveValue(),
        "budget_spend": round(spend_gbp, 2),
        "stamina_end": round(solver.Value(stamina_end) / 100.0, 2),
        "poi_score": solver.Value(poi_score) / 100.0,
        "experience_score": solver.Value(experience_score) / 100.0,
        "comfort_score": solver.Value(comfort_score) / 100.0,
        "selected_count": selected_count,
        "visits": path,
        "solve_time_s": round(solve_s, 2)
    }


# ---------- Multi-hotel wrapper ----------
def solve_intracity_city_day(
    data: Dict[str, Any],
    policy: Dict[str, Any],
    cp_verbose: bool = False
) -> Dict[str, Any]:
    hotels = data["hotels"]
    pois = data["pois"]
    dist = data.get("poi_poi_distance_km")

    if not pois or not hotels:
        return {"status": "infeasible", "reason": "missing hotels/pois"}

    # If distance matrix absent, compute
    if not dist:
        N = len(pois)
        dist = [[0.0]*N for _ in range(N)]
        for i in range(N):
            for j in range(N):
                if i != j:
                    dist[i][j] = haversine_km((pois[i]["latitude"], pois[i]["longitude"]),
                                              (pois[j]["latitude"], pois[j]["longitude"]))

    results = []
    for h in hotels:
        res = solve_intracity_day_for_hotel(h, pois, dist, policy, cp_verbose=cp_verbose)
        results.append(res)

    feas = [r for r in results if r.get("status") == "ok"]
    if not feas:
        return {"status": "infeasible", "candidates": results}

    best = max(feas, key=lambda r: r["objective"])
    return {"status": "ok", "best": best, "candidates": results}


# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="Intracity CP-SAT planner (with budget, stamina, experience, comfort).")
    ap.add_argument("--data", type=str, default=None, help="Path to dataset JSON.")
    ap.add_argument("--policy", type=str, default=None, help="Path to policy JSON.")
    ap.add_argument("--cp-verbose", action="store_true", help="Enable CP-SAT internal logs.")
    args = ap.parse_args()

    data = load_data(args.data)
    policy = load_policy(args.policy)
    out = solve_intracity_city_day(data, policy, cp_verbose=args.cp_verbose)

    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()