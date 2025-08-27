#!/usr/bin/env python3
"""
Intracity day planner (CP-SAT) — Updated for fixed costs and POI-based stamina
================================================================================
This version implements the user's exact requirements:
- Fixed transport costs per trip (not per minute)
- POI-based stamina costs (liked/disliked categories)
- Distance-based cab pricing
- Proper transport mode handling (walk, bus, tram, train, cab)
"""

from __future__ import annotations
from typing import Dict, Any, Tuple, List, Optional
from ortools.sat.python import cp_model
import argparse, json, os, time, logging

# --------------------------------
# Logging
# --------------------------------
log = logging.getLogger("intracity")
if not log.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    log.addHandler(h)
log.setLevel(logging.INFO if os.getenv("LOG_LEVEL","INFO").upper()!="DEBUG" else logging.DEBUG)

# --------------------------------
# Small helpers
# --------------------------------
def _namekey(s: str) -> str:
    """Lowercased, trimmed key for dict lookups."""
    return (s or "").strip().casefold()

def _num(v) -> Optional[float]:
    """Parse possibly-string cell like 'X', '', '  ', '28' → float or None."""
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    if s == "" or s.lower() in {"x", "na", "n/a", "-"}:
        return None
    try:
        return float(s)
    except Exception:
        return None

def _pick(rec: Dict[str, Any], *candidates: str) -> Optional[float]:
    """Return first numeric value among candidate keys (float) or None."""
    for k in candidates:
        if k in rec:
            val = _num(rec[k])
            if val is not None:
                return v
    return None

def _pick_edge_km(rec: Dict[str, Any]) -> Optional[float]:
    """Choose a sensible edge distance (km) from any available distance-like keys."""
    # Priority order — cab distance tends to correspond to road distance
    for ks in [
        ("edge_km",),
        ("cab_km","Cab_dist","Cab_distance"),
        ("walk_km","Walk_dist","Walk_distance"),
        ("bus_km","Bus_dist","Bus_distance"),
        ("tram_km","Tram_dist","Tram_distance","tram/tube_dist","tram/tube_distance")
    ]:
        v = _pick(rec, *ks)
        if v is not None:
            return v
    return None

def _get_time_fields(rec: Dict[str, Any], mode: str) -> Optional[int]:
    """Return minutes for mode from various sheet key variants, as integer minutes."""
    if mode == "walk":
        v = _pick(rec, "walk_min", "Walk_time", "Walk_time(mins)", "walk_time", "walk_time(mins)")
    elif mode == "cab":
        v = _pick(rec, "cab_min", "Cab_time")
    elif mode == "bus":
        v = _pick(rec, "bus_min", "Bus_time")
    elif mode == "tram":
        v = _pick(rec, "tram_min", "Tram_time", "tram/tube_time")
    elif mode == "train":
        v = _pick(rec, "train_min", "Train_time")
    else:
        v = None
    if v is None:
        return None
    return int(round(v))

def _get_transfer_cnt(rec: Dict[str, Any], mode: str) -> int:
    if mode == "bus":
        v = _pick(rec, "bus_cnt", "Bus_count")
    elif mode == "tram":
        v = _pick(rec, "tram_cnt", "Tram_count", "tram/tube_count")
    else:
        v = 0
    return int(round(v or 0))

# --------------------------------
# Default policy + loader
# --------------------------------
def default_policy() -> Dict[str, Any]:
    """Default policy with user's exact requirements"""
    return {
        "unique_id": "user_requirements",
        "priority_order": ["stamina", "budget", "poi_score", "experience", "comfort"],
        "priority_weights": {
            "stamina": 10**12, "budget": 10**12, "poi_score": 10**12, 
            "experience": 10**12, "comfort": 10**6
        },
        
        # Budget system - FIXED COSTS PER TRIP (not per minute)
        "budget": {
            "daily_cap": 300.0,
            "mode_fixed_cost": {
                "walk": 0.0,    # Free
                "train": 1.0,   # £1 per trip
                "bus": 2.0,     # £2 per trip
                "tram": 0.0,    # Free (or adjust as needed)
                "cab": 5.0      # Base cost
            },
            "cab_distance_pricing": {
                "threshold_1": 2.0,
                "cost_under_2km": 5.0,
                "cost_2_to_4km": 10.0,
                "cost_over_4km": 15.0
            }
        },
        
        # Stamina system - POI-BASED COSTS (not time-based)
        "stamina": {
            "start": 10.0,
            "max": 12.0,
            "poi_visit_cost": 1.5,           # Base stamina cost per POI
            "poi_liked_reduction": 1.0,      # Museums, historic = -1 stamina
            "poi_disliked_reduction": 2.0,   # Theme parks, bars = -2 stamina
            "meal_gain": 2.0,                # +2 stamina recovery
            "transport_costs": {
                "walk_per_hour": 2.0,        # Highest penalty
                "train_per_hour": 1.0,       # Medium penalty
                "bus_per_hour": 1.0,         # Medium penalty
                "tram_per_hour": 1.0,        # Medium penalty
                "cab_per_hour": 0.5          # Lowest penalty
            }
        },
        
        # Transport speeds (only used when Google Sheets data is missing)
        "transport_speeds_kmph": {
            "walk": 4.5,      # 4.5 km/h walking speed
            "bus": 14.0,      # 14 km/h average bus speed
            "tram": 18.0,     # 18 km/h average tram speed
            "train": 25.0,    # 25 km/h average train speed
            "cab": 22.0       # 22 km/h average cab speed
        },
        
        # Comfort factors (used in objective function)
        "comfort_discomfort_per_min": {
            "walk": 1.0,      # Highest discomfort
            "train": 0.6,     # Medium discomfort
            "bus": 0.5,       # Medium discomfort
            "tram": 0.6,      # Medium discomfort
            "cab": 0.2        # Lowest discomfort
        },
        
        # POI preferences
        "must_include": ["museum", "historic"],      # +3 points each
        "must_avoid": ["theme_parks", "noisy", "bar"], # -2 points each
        
        # Planning constraints
        "max_pois_per_day": 6,
        "min_pois_per_day": 2,
        "max_one_way_distance_from_hotel_km": 20.0,
        "max_total_distance_day_km": 50.0,
        "walk_only_threshold_km": 0.3,
        "transfer_penalty_per_change": 0.5,
        
        # Solver settings
        "solver_max_seconds": 10.0,
        "solver_workers": 8
    }

def _merge_dict(dst: Dict[str, Any], src: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge source dict into destination dict recursively"""
    if not isinstance(src, dict): 
        return dst
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _merge_dict(dst[k], v)
        else:
            dst[k] = v
    return dst

def _lex_from_order(order: List[str]) -> Dict[str, int]:
    """Generate lexicographic weights from priority order"""
    SAFE_LEX_BASE = 10**6
    MAX_WEIGHT = 10**12
    n = len(order)
    weights: Dict[str, int] = {}
    for i, k in enumerate(order):
        w = SAFE_LEX_BASE ** (n - i)
        if w > MAX_WEIGHT: 
            w = MAX_WEIGHT
        weights[k] = int(w)
    return weights

def load_policy(path: Optional[str]) -> Dict[str, Any]:
    """Load policy from file or use default"""
    pol = default_policy()
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            _merge_dict(pol, json.load(f))
    
    # Generate weights if not present
    if "priority_weights" not in pol and isinstance(pol.get("priority_order"), list):
        pol["priority_weights"] = _lex_from_order(pol["priority_order"])
    
    return pol

# --------------------------------
# Core solve (one hotel)
# --------------------------------
def _solve_one_hotel(data: Dict[str, Any], policy: Dict[str, Any], hname: str,
                     cp_verbose: bool = False) -> Dict[str, Any]:
    """Solve planning for one hotel with updated cost and stamina systems"""
    
    hotels = data["hotels"]
    pois = data["pois"]
    N = len(pois)
    
    # POI name→index map (1..N). Start=0, End=N+1
    idx_of = {_namekey(p["name"]): i+1 for i,p in enumerate(pois)}
    start, end = 0, N+1
    
    # Select hotel (fallback to first if not exact match)
    H = next((h for h in hotels if _namekey(h.get("name")) == _namekey(hname)), hotels[0])
    
    # Init structures
    modes = ["walk", "bus", "tram", "train", "cab"]
    TT = {m: [[None]*(N+2) for _ in range(N+2)] for m in modes}
    TRANSF = {m: [[0]*(N+2) for _ in range(N+2)] for m in ("bus","tram")}
    EDGEKM = [[None]*(N+2) for _ in range(N+2)]
    
    # --- HOTEL ↔ POI from bundle
    hotel_times_all = (data.get("hotel_poi_mode_times") or {})
    hrow = hotel_times_all.get(_namekey(hname)) or hotel_times_all.get(_namekey(H.get("name","")))
    if hrow:
        for poi_name, rec in hrow.items():
            j = idx_of.get(_namekey(poi_name))
            if not j:
                continue
            edge_km = _pick_edge_km(rec)
            if edge_km is not None:
                EDGEKM[start][j] = edge_km
                EDGEKM[j][end]   = edge_km
            for m in modes:
                t = _get_time_fields(rec, m)
                if t is not None and t > 0:
                    TT[m][start][j] = int(t)
                    TT[m][j][end]   = int(t)
            TRANSF["bus"][start][j]  = _get_transfer_cnt(rec, "bus")
            TRANSF["bus"][j][end]    = _get_transfer_cnt(rec, "bus")
            TRANSF["tram"][start][j] = _get_transfer_cnt(rec, "tram")
            TRANSF["tram"][j][end]   = _get_transfer_cnt(rec, "tram")
    
    # --- POI ↔ POI from bundle
    poi_times = (data.get("poi_mode_times") or {})
    def _iter_poi_pairs():
        for k, rec in poi_times.items():
            if isinstance(k, (list, tuple)) and len(k)==2:
                yield _namekey(k[0]), _namekey(k[1]), rec
            elif isinstance(k, str) and "|||" in k:
                a,b = k.split("|||",1)
                yield _namekey(a), _namekey(b), rec
            else:
                continue
    
    for src, dst, rec in _iter_poi_pairs():
        i = idx_of.get(src)
        j = idx_of.get(dst)
        if not i or not j or i==j:
            continue
        EDGEKM[i][j] = _pick_edge_km(rec)
        for m in modes:
            t = _get_time_fields(rec, m)
            if t is not None and t > 0:
                TT[m][i][j] = int(t)
        TRANSF["bus"][i][j]  = _get_transfer_cnt(rec, "bus")
        TRANSF["tram"][i][j] = _get_transfer_cnt(rec, "tram")
    
    # --- Determine available edges and modes
    max_one_way_hotel_km = float(policy.get("max_one_way_distance_from_hotel_km", 10**9))
    
    available_modes: Dict[Tuple[int,int], List[str]] = {}
    for i in range(N+2):
        for j in range(N+2):
            if i == j:
                continue
            # must have edge distance
            edge_km = EDGEKM[i][j]
            if edge_km is None:
                continue
            
            # disallow hotel↔POI edges beyond the policy one-way range
            if (i == start or j == end) and edge_km > max_one_way_hotel_km:
                continue
            
            ms: List[str] = []
            for m in modes:
                tmin = TT[m][i][j]
                if tmin is None or tmin <= 0:
                    continue
                ms.append(m)
            
            if ms:
                available_modes[(i,j)] = ms
    
    if not available_modes:
        return {"status": "infeasible", "hotel": H.get("name",""), "reason": "no valid edges"}
    
    # --------------------------------
    # CP-SAT model
    # --------------------------------
    model = cp_model.CpModel()
    
    # POI selection variables
    x = {i: model.NewBoolVar(f"x_{i}") for i in range(1, N+1)}
    
    # Edge (flow) variables only for allowed edges
    y = {(i,j): model.NewBoolVar(f"y_{i}_{j}") for (i,j) in available_modes.keys()}
    
    # Mode selection for each allowed edge
    x_mode = {}
    for (i,j), ms in available_modes.items():
        for m in ms:
            x_mode[(i,j,m)] = model.NewBoolVar(f"x_mode_{i}_{j}_{m}")
    
    # Start has exactly 1 outgoing; End has exactly 1 incoming
    model.Add(sum(y[(start,j)] for (i,j) in y if i == start) == 1)
    model.Add(sum(y[(i,end)]   for (i,j) in y if j == end)   == 1)
    
    # Exactly 1 in and 1 out for each selected POI; 0 otherwise
    for i in range(1, N+1):
        out_edges = [y[(i,j)] for (ii,j) in y if ii == i]
        in_edges  = [y[(k,i)] for (k,jj) in y if jj == i]
        if out_edges:
            model.Add(sum(out_edges) == x[i])
        else:
            model.Add(x[i] == 0)
        if in_edges:
            model.Add(sum(in_edges)  == x[i])
        else:
            model.Add(x[i] == 0)
    
    # Tie modes to edges
    for (i,j), ms in available_modes.items():
        model.Add(sum(x_mode[(i,j,m)] for m in ms) == y[(i,j)])
    
    # MTZ subtour elimination
    u = {i: model.NewIntVar(0, N, f"u_{i}") for i in range(N+2)}
    model.Add(u[start] == 0)
    for i in range(1, N+1):
        model.Add(u[i] >= x[i])
        model.Add(u[i] <= N * x[i])
    poi_ids = list(range(1, N+1))
    for i in poi_ids:
        for j in poi_ids:
            if i == j: 
                continue
            if (i,j) in y:
                model.Add(u[i] - u[j] + (N) * y[(i,j)] <= N - 1)
    
    # POI count constraints
    model.Add(sum(x.values()) >= int(policy.get("min_pois_per_day", 1)))
    model.Add(sum(x.values()) <= int(policy.get("max_pois_per_day", 6)))
    
    # Distance cap (km)
    max_total_km_day = float(policy.get("max_total_distance_day_km", 50.0))
    dist_terms = []
    for (i,j) in y:
        km = EDGEKM[i][j]
        if km is None:
            continue
        dist_terms.append(int(round(km * 100)) * y[(i,j)])
    if dist_terms:
        model.Add(sum(dist_terms) <= int(round(max_total_km_day * 100)))
    
    # ---------- Budget constraints (FIXED COSTS) ----------
    mode_fixed_cost = {k: float(v) for k, v in policy.get("budget", {}).get("mode_fixed_cost", {}).items()}
    cab_pricing = policy.get("budget", {}).get("cab_distance_pricing", {})
    daily_cap_cents = int(round(float(policy.get("budget", {}).get("daily_cap", 150.0)) * 100.0))
    travel_cost_cents_terms: List[cp_model.IntVar] = []
    
    def _intc(v: float) -> int:
        return int(round(v))
    
    for (i,j), ms in available_modes.items():
        for m in ms:
            tmin = TT[m][i][j]
            if tmin is None or tmin <= 0:
                continue
            
            # Calculate cost based on transport mode and distance
            if m == "cab" and cab_pricing:
                # Cab pricing based on distance
                km = EDGEKM[i][j]
                if km is not None:
                    if km < cab_pricing.get("threshold_1", 2.0):
                        cost = cab_pricing.get("cost_under_2km", 5.0)
                    elif km <= 4.0:
                        cost = cab_pricing.get("cost_2_to_4km", 10.0)
                    else:
                        cost = cab_pricing.get("cost_over_4km", 15.0)
                else:
                    cost = mode_fixed_cost.get(m, 5.0)
            else:
                # Fixed cost for other modes
                cost = mode_fixed_cost.get(m, 0.0)
            
            c_fixed = _intc(cost * 100)
            v = model.NewIntVar(0, c_fixed, f"cost_{i}_{j}_{m}")
            model.Add(v == c_fixed * x_mode[(i,j,m)])
            travel_cost_cents_terms.append(v)
    
    # POI costs
    poi_price = [0.0]*(N+2)
    for i,p in enumerate(pois, start=1):
        poi_price[i] = float(p.get("price", 0.0))
    poi_ticket_cents = sum(_intc(poi_price[i] * 100) * x[i] for i in range(1, N+1))
    
    daily_spend_cents = model.NewIntVar(0, 10**9, "daily_spend_cents")
    model.Add(daily_spend_cents == poi_ticket_cents + sum(travel_cost_cents_terms))
    model.Add(daily_spend_cents <= daily_cap_cents)
    
    # ---------- Stamina constraints (POI-BASED) ----------
    S0 = float(policy.get("stamina", {}).get("start", 10.0))
    Smax = float(policy.get("stamina", {}).get("max", 12.0))
    
    stamina_terms: List[cp_model.IntVar] = []
    mode_minutes = {m: [] for m in modes}
    
    # Transport stamina costs
    for (i,j), ms in available_modes.items():
        for m in ms:
            tmin = TT[m][i][j]
            if tmin is None:
                continue
            
            # Get stamina cost per hour for this transport mode
            stamina_key = f"{m}_per_hour"
            if stamina_key in policy.get("stamina", {}).get("transport_costs", {}):
                drain_per_hour = float(policy["stamina"]["transport_costs"][stamina_key])
            else:
                drain_per_hour = 0.0
            
            # Convert to per-minute and calculate total drain
            drain_per_min = drain_per_hour / 60.0
            drain_int = int(round(drain_per_min * tmin * 100))
            
            z = model.NewIntVar(0, max(0, drain_int), f"stam_{i}_{j}_{m}")
            model.Add(z == drain_int * x_mode[(i,j,m)])
            stamina_terms.append(z)
            
            # minutes in mode
            mm = model.NewIntVar(0, int(tmin), f"min_{i}_{j}_{m}")
            model.Add(mm == int(tmin) * x_mode[(i,j,m)])
            mode_minutes[m].append(mm)
    
    # POI stamina costs based on user preferences
    poi_base_cost = float(policy["stamina"].get("poi_visit_cost", 1.5))
    poi_liked_reduction = float(policy["stamina"].get("poi_liked_reduction", 1.0))
    poi_disliked_reduction = float(policy["stamina"].get("poi_disliked_reduction", 2.0))
    
    # Determine if POI is liked/disliked based on category
    poi_stamina_costs = []
    for i in range(1, N+1):
        poi = pois[i-1]
        category = str(poi.get("category", "")).lower()
        
        # Check if POI matches user preferences
        is_liked = any(pref in category for pref in ["museum", "historic", "cultural"])
        is_disliked = any(pref in category for pref in ["theme_park", "noisy", "bar"])
        
        if is_liked:
            cost = poi_liked_reduction
        elif is_disliked:
            cost = poi_disliked_reduction
        else:
            cost = poi_base_cost
        
        poi_stamina_costs.append(int(round(cost * 100)))
    
    # Calculate total POI stamina cost
    poi_stamina_terms = [poi_stamina_costs[i-1] * x[i] for i in range(1, N+1)]
    
    # Meal stamina recovery
    poi_is_meal = [0]*(N+2)
    for i,p in enumerate(pois, start=1):
        poi_is_meal[i] = 1 if str(p.get("category","")).lower() in {"restaurant","food","meal"} else 0
    
    meals_count = sum(x[i] * poi_is_meal[i] for i in range(1, N+1))
    meal_gain_i = int(round(float(policy["stamina"].get("meal_gain", 2.0)) * 100))
    
    # Final stamina calculation
    stamina_end = model.NewIntVar(0, int(round(Smax * 100)), "stamina_end")
    model.Add(
        stamina_end
        == int(round(S0 * 100))
           - sum(stamina_terms)
           - sum(poi_stamina_terms)
           + meal_gain_i * meals_count
    )
    
    # Stamina must be non-negative
    model.Add(stamina_end >= 0)
    
    # ---------- POI scoring ----------
    poi_score = sum(int(round(100 * float(pois[i-1].get("rating", 1.0)))) * x[i] for i in range(1, N+1))
    
    # Experience scoring based on preferences
    must_include = [s.lower() for s in (policy.get("must_include") or [])]
    must_avoid   = [s.lower() for s in (policy.get("must_avoid") or [])]
    include_bonus = 3  # +3 for must-include
    avoid_penalty = 2   # -2 for must-avoid
    
    poi_cat = [""]*(N+2)
    for i,p in enumerate(pois, start=1):
        poi_cat[i] = str(p.get("category","")).lower()
    
    # Must-include bonus
    include_terms = []
    for i in range(1, N+1):
        if any(kw in poi_cat[i] for kw in must_include):
            include_terms.append(include_bonus * x[i])
    include_term = model.NewIntVar(0, include_bonus * N, "include_term")
    model.Add(include_term == (sum(include_terms) if include_terms else 0))
    
    # Must-avoid penalty
    avoid_terms = []
    for i in range(1, N+1):
        if any(kw in poi_cat[i] for kw in must_avoid):
            avoid_terms.append(avoid_penalty * x[i])
    avoid_term = model.NewIntVar(0, avoid_penalty * N, "avoid_term")
    model.Add(avoid_term == (sum(avoid_terms) if avoid_terms else 0))
    
    # Experience score
    experience_score = model.NewIntVar(-avoid_penalty * N, include_bonus * N, "experience_score")
    model.Add(experience_score == include_term - avoid_term)
    
    # ---------- Comfort penalties ----------
    discomfort_per_min = policy.get("comfort_discomfort_per_min", {})
    transport_discomfort_terms: List[cp_model.IntVar] = []
    for (i,j), ms in available_modes.items():
        for m in ms:
            tmin = TT[m][i][j]
            if tmin is None:
                continue
            dpm = float(discomfort_per_min.get(m, 1.0))
            cost_i = int(round(dpm * tmin * 100))
            v = model.NewIntVar(0, cost_i, f"disc_{i}_{j}_{m}")
            model.Add(v == cost_i * x_mode[(i,j,m)])
            transport_discomfort_terms.append(v)
    transport_discomfort = sum(transport_discomfort_terms)
    
    # Transfer penalty
    transfer_penalty = int(round(float(policy.get("transfer_penalty_per_change", 0.0)) * 100))
    transfer_terms: List[cp_model.IntVar] = []
    if transfer_penalty > 0:
        for (i,j), ms in available_modes.items():
            for m in ("bus","tram"):
                if m not in ms:
                    continue
                cnt = int(TRANSF[m][i][j])
                if cnt <= 0: 
                    continue
                v = model.NewIntVar(0, transfer_penalty * cnt, f"trans_{i}_{j}_{m}")
                model.Add(v == transfer_penalty * cnt * x_mode[(i,j,m)])
                transfer_terms.append(v)
    transfer_cost = sum(transfer_terms)
    
    # ---------- Objective function ----------
    W = policy.get("priority_weights") or _lex_from_order(
        policy.get("priority_order", ["stamina","budget","poi_score","experience","comfort"])
    )
    objective = (
        + int(W.get("stamina", 0))  * stamina_end           # Maximize stamina
        - int(W.get("budget", 0))   * daily_spend_cents     # Minimize spending
        + int(W.get("poi_score", 0))* poi_score             # Maximize POI ratings
        + int(W.get("experience",0))* experience_score      # Maximize preferences
        - int(W.get("comfort", 0))  * (transport_discomfort + transfer_cost)  # Minimize discomfort
    )
    model.Maximize(objective)
    
    # ---------- Solve ----------
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(policy.get("solver_max_seconds", 10.0))
    solver.parameters.num_search_workers = int(policy.get("solver_workers", 8))
    solver.parameters.log_to_stdout = bool(cp_verbose)
    
    t0 = time.perf_counter()
    status = solver.Solve(model)
    dt = time.perf_counter() - t0
    log.info(f"[{H.get('name','')}] Status={solver.StatusName(status)} solve_time={dt:.2f}s")
    
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return {"status": "infeasible", "hotel": H.get("name",""), "reason": "no feasible solution"}
    
    # Reconstruct path
    succ = {i: None for i in range(N+2)}
    for (i,j) in y:
        if solver.Value(y[(i,j)]) == 1:
            succ[i] = j
    
    path = []
    cur = start
    total_travel_cost = 0.0
    total_travel_time = 0
    total_distance_km = 0.0
    mode_usage = {"walk":0.0,"bus":0.0,"tram":0.0,"train":0.0,"cab":0.0}
    
    while cur is not None and cur != end:
        nxt = succ.get(cur)
        if nxt is None:
            break
        chosen_mode = None
        tmin = 0
        for m in modes:
            if (cur,nxt,m) in x_mode and solver.Value(x_mode[(cur,nxt,m)]) == 1:
                chosen_mode = m
                break
        if chosen_mode:
            tmin = TT[chosen_mode][cur][nxt] or 0
            mode_usage[chosen_mode] += tmin
            # Calculate cost based on mode and distance
            if chosen_mode == "cab" and cab_pricing:
                km = EDGEKM[cur][nxt]
                if km is not None:
                    if km < cab_pricing.get("threshold_1", 2.0):
                        cost = cab_pricing.get("cost_under_2km", 5.0)
                    elif km <= 4.0:
                        cost = cab_pricing.get("cost_2_to_4km", 10.0)
                    else:
                        cost = cab_pricing.get("cost_over_4km", 15.0)
                else:
                    cost = mode_fixed_cost.get(chosen_mode, 5.0)
            else:
                cost = mode_fixed_cost.get(chosen_mode, 0.0)
            total_travel_cost += cost
        
        total_travel_time += int(tmin)
        total_distance_km += float(EDGEKM[cur][nxt] or 0.0)
        poi_name_h = (pois[nxt-1]["name"] if 1 <= nxt <= N else ("Hotel" if nxt==end else "Start"))
        path.append({
            "from": ("Hotel" if cur==start else pois[cur-1]["name"]),
            "to": poi_name_h,
            "mode": chosen_mode or "walk",
            "travel_min": int(tmin),
            "cost": cost if chosen_mode else 0.0
        })
        cur = nxt
    
    selected_count = sum(solver.Value(x[i]) for i in range(1, N+1))
    out = {
        "status": "ok",
        "hotel": H.get("name",""),
        "objective": solver.ObjectiveValue(),
        "budget_spend": round(solver.Value(daily_spend_cents)/100.0, 2),
        "stamina_end": round(solver.Value(stamina_end)/100.0, 2),
        "poi_score": solver.Value(poi_score)/100.0,
        "experience_score": solver.Value(experience_score),
        "comfort_penalty": (solver.Value(transport_discomfort)+solver.Value(transfer_cost))/100.0,
        "selected_count": int(selected_count),
        "visits": path,
        "solve_time_s": round(dt, 2),
        "modal_split": {k: round(v,1) for k,v in mode_usage.items() if v>0},
        "total_travel_time_min": int(total_travel_time),
        "total_travel_cost_gbp": round(total_travel_cost, 2),
        "total_distance_km": round(total_distance_km, 1),
    }
    return out

# --------------------------------
# Public API: solve across hotels
# --------------------------------
def solve_intracity_city_day(data: Dict[str, Any], policy: Dict[str, Any], cp_verbose: bool=False) -> Dict[str, Any]:
    """Solve once per hotel and return the best result by objective value."""
    hotels = data.get("hotels", [])
    if not hotels:
        return {"status": "error", "error": "no hotels provided"}
    
    best = None
    per_hotel = []
    for H in hotels:
        name = H.get("name","")
        res = _solve_one_hotel(data, policy, name, cp_verbose=cp_verbose)
        per_hotel.append(res)
        if res.get("status") == "ok" and (best is None or res["objective"] > best["objective"]):
            best = res
    
    if best is None:
        return {"status": "infeasible", "per_hotel": per_hotel}
    
    return {"status":"ok", "best": best, "per_hotel": per_hotel}

# --------------------------------
# CLI
# --------------------------------
def main():
    ap = argparse.ArgumentParser(description="Intracity CP-SAT day planner (updated for fixed costs)")
    ap.add_argument("--data", type=str, required=True, help="Path to data bundle JSON")
    ap.add_argument("--policy", type=str, default=None, help="Path to policy JSON (optional)")
    ap.add_argument("--cp-verbose", action="store_true", help="Show OR-Tools search logs")
    args = ap.parse_args()
    
    with open(args.data, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    policy = load_policy(args.policy) if args.policy else default_policy()
    res = solve_intracity_city_day(data, policy, cp_verbose=args.cp_verbose)
    print(json.dumps(res, indent=2))
    
if __name__ == "__main__":
    main()
