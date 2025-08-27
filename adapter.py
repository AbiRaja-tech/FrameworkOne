# adapter.py
from typing import Dict, Any, List, Tuple, Optional

# --------------------------
# Small helpers
# --------------------------
def _clean_city(s: str) -> str:
    return (s or "").strip()

def _safe_float(x, default: Optional[float] = None) -> Optional[float]:
    try:
        s = str(x).strip()
        if not s or s.lower() in {"na", "n/a", "null"}:
            return default
        return float(s)
    except Exception:
        return default

def _num_from_money_or_text(x, default: float = 0.0) -> float:
    """Extract a numeric value from strings like '£139', '139', '4.3/5', etc."""
    if x is None:
        return default
    s = str(x).strip()
    if "/" in s:  # e.g. "4.5/5" -> 4.5
        try:
            return float(s.split("/")[0])
        except Exception:
            return default
    # keep digits, dot and minus
    cleaned = "".join(ch for ch in s if ch.isdigit() or ch in ".-")
    try:
        return float(cleaned) if cleaned else default
    except Exception:
        return default

def _synthesize_poi_name(p: Dict[str, Any]) -> str:
    cat = (p.get("category") or "POI").strip().title()
    addr = (p.get("address") or "").split(",")[0].strip()
    if addr:
        return f"{cat} – {addr}"
    lat = _safe_float(p.get("latitude"))
    lon = _safe_float(p.get("longitude"))
    if lat is not None and lon is not None:
        return f"{cat} @ {round(lat,4)},{round(lon,4)}"
    return cat

# --------------------------
# Public: sanitize the bundle
# --------------------------
def sanitize_bundle(bundle: Dict[str, Any]) -> Dict[str, Any]:
    """
    - Trim/normalize city names in distance matrices.
    - Normalize POIs (city/name/category/coords).
    - Keep hotels even if coords are missing (coords can be filled later).
    """
    # 1) Distance matrices: normalize city keys
    dmat = bundle.get("poi_distance_matrices", {}) or {}
    folded = {}
    for k, v in dmat.items():
        folded[_clean_city(k)] = v
    bundle["poi_distance_matrices"] = folded

    # 2) POIs: normalize & backfill names
    for p in bundle.get("pois", []) or []:
        p["city"] = _clean_city(p.get("city"))
        if not p.get("name"):
            p["name"] = _synthesize_poi_name(p)
        if isinstance(p.get("category"), str):
            p["category"] = p["category"].strip()
        # ensure floats where possible
        p["latitude"]  = _safe_float(p.get("latitude"))
        p["longitude"] = _safe_float(p.get("longitude"))

    # 3) Hotels: keep them, normalize city and just parse what we can
    hotels = []
    for h in bundle.get("hotels", []) or []:
        h["city"] = _clean_city(h.get("city"))
        # Don't drop for missing coords—these will be filled from POI centroid later
        h["latitude"]  = _safe_float(h.get("latitude"))
        h["longitude"] = _safe_float(h.get("longitude"))
        # Ensure a name
        if not h.get("name"):
            h["name"] = (h.get("hotel_name")
                         or f"Hotel in {h['city'] or 'city'}").strip()
        hotels.append(h)
    bundle["hotels"] = hotels

    return bundle

# --------------------------
# Internal: select per-city POIs and DM
# --------------------------
def _select_city_subset(bundle: Dict[str, Any], city: str) -> Tuple[List[Dict[str, Any]], List[int], List[List[float]]]:
    """
    Returns (pois_for_city, poi_indices_in_bundle_order, distance_matrix_for_those_indices).
    Order of POIs matches the order from 'poi_indices' in the tool bundle.
    """
    dm_info = (bundle.get("poi_distance_matrices") or {}).get(city)
    if not dm_info:
        # Fallback: create distance matrix from mode times data
        print(f"   [ADAPTER] No poi_distance_matrices for {city}, creating from mode times data")
        return _create_distance_matrix_from_mode_times(bundle, city)
    
    idxs: List[int] = dm_info.get("poi_indices") or []
    all_pois = bundle.get("pois") or []
    pois_for_city = [all_pois[i] for i in idxs if i < len(all_pois)]
    distance_matrix = dm_info.get("distance_km") or []
    
    print(f"   [ADAPTER] Using poi_distance_matrices for {city}: {len(pois_for_city)} POIs, {len(distance_matrix)} distance rows")
    return pois_for_city, idxs, distance_matrix

def _create_distance_matrix_from_mode_times(bundle: Dict[str, Any], city: str) -> Tuple[List[Dict[str, Any]], List[int], List[List[float]]]:
    """
    Create a distance matrix from mode times data when poi_distance_matrices is empty.
    This handles the case where POIs lack coordinates.
    """
    print(f"   [ADAPTER] Creating distance matrix from mode times for {city}")
    
    # Get all POIs for this city
    all_pois = bundle.get("pois", [])
    pois_for_city = [p for p in all_pois if p.get("city", "").strip().casefold() == city.casefold()]
    
    if not pois_for_city:
        print(f"   [ADAPTER] No POIs found for city {city}")
        return [], [], []
    
    print(f"   [ADAPTER] Found {len(pois_for_city)} POIs for {city}")
    
    # Create POI name to index mapping
    poi_names = [p.get("name", "").strip() for p in pois_for_city]
    poi_name_to_idx = {name: i for i, name in enumerate(poi_names) if name}
    
    # Get mode times data
    poi_mode_times = bundle.get("poi_mode_times", {}).get(city, {})
    hotel_poi_mode_times = bundle.get("hotel_poi_mode_times", {}).get(city, {})
    
    # Initialize distance matrix
    N = len(pois_for_city)
    distance_matrix = [[float('inf')] * N for _ in range(N)]
    
    # Set diagonal to 0
    for i in range(N):
        distance_matrix[i][i] = 0.0
    
    # Fill in distances from POI-POI mode times
    print(f"   [ADAPTER] Processing {len(poi_mode_times)} POI-POI connections")
    for connection_key, mode_data in poi_mode_times.items():
        # Parse connection key (format: "src->dst")
        if "->" in connection_key:
            src, dst = connection_key.split("->", 1)
        else:
            # Fallback for old format
            src, dst = connection_key, connection_key
            
        if src in poi_name_to_idx and dst in poi_name_to_idx:
            src_idx = poi_name_to_idx[src]
            dst_idx = poi_name_to_idx[dst]
            
            # Get distance from mode data
            edge_km = mode_data.get("edge_km")
            if edge_km is not None and edge_km > 0:
                distance_matrix[src_idx][dst_idx] = edge_km
                distance_matrix[dst_idx][src_idx] = edge_km  # Symmetric
                print(f"     [ADAPTER] {src} -> {dst}: {edge_km} km")
    
    # Fill in missing distances using walking time estimates
    print(f"   [ADAPTER] Estimating missing distances from walking times")
    for connection_key, mode_data in poi_mode_times.items():
        # Parse connection key (format: "src->dst")
        if "->" in connection_key:
            src, dst = connection_key.split("->", 1)
        else:
            # Fallback for old format
            src, dst = connection_key, connection_key
            
        if src in poi_name_to_idx and dst in poi_name_to_idx:
            src_idx = poi_name_to_idx[src]
            dst_idx = poi_name_to_idx[dst]
            
            # If distance is still infinity, try to estimate from walking time
            if distance_matrix[src_idx][dst_idx] == float('inf'):
                walk_min = mode_data.get("walk_min")
                if walk_min is not None and walk_min > 0:
                    # Estimate distance from walking time (4.5 km/h)
                    estimated_km = (walk_min / 60.0) * 4.5
                    distance_matrix[src_idx][dst_idx] = estimated_km
                    distance_matrix[dst_idx][src_idx] = estimated_km
                    print(f"     [ADAPTER] Estimated {src} -> {dst}: {estimated_km:.2f} km (from {walk_min} min walk)")
    
    # CRITICAL FIX: Create a complete network by inferring missing connections
    print(f"   [ADAPTER] Creating complete network by inferring missing connections")
    
    # Strategy 1: Use hotel as central hub to estimate POI-to-POI distances
    if hotel_poi_mode_times:
        print(f"     [ADAPTER] Using hotel as central hub to estimate missing connections")
        for hotel, hotel_data in hotel_poi_mode_times.items():
            # Get distances from this hotel to all POIs
            hotel_to_poi_distances = {}
            for poi, mode_data in hotel_data.items():
                if poi in poi_name_to_idx:
                    poi_idx = poi_name_to_idx[poi]
                    edge_km = mode_data.get("edge_km")
                    if edge_km is not None and edge_km > 0:
                        hotel_to_poi_distances[poi_idx] = edge_km
            
            # Estimate POI-to-POI distances using triangle inequality
            for i in range(N):
                for j in range(N):
                    if i != j and distance_matrix[i][j] == float('inf'):
                        # Try to estimate using hotel as intermediate point
                        if i in hotel_to_poi_distances and j in hotel_to_poi_distances:
                            # Estimate: POI_i -> Hotel -> POI_j
                            estimated_km = hotel_to_poi_distances[i] + hotel_to_poi_distances[j]
                            # Add some penalty for the indirect route
                            estimated_km = estimated_km * 0.8  # 20% penalty
                            
                            if estimated_km < 10.0:  # Reasonable distance limit
                                distance_matrix[i][j] = estimated_km
                                distance_matrix[j][i] = estimated_km
                                print(f"       [ADAPTER] Estimated {poi_names[i]} -> {poi_names[j]}: {estimated_km:.2f} km (via hotel)")
    
    # Strategy 2: Fill remaining gaps with reasonable estimates based on POI order
    print(f"     [ADAPTER] Filling remaining gaps with reasonable estimates")
    for i in range(N):
        for j in range(N):
            if i != j and distance_matrix[i][j] == float('inf'):
                # Estimate based on POI order and typical city distances
                base_distance = 1.0 + (abs(i - j) * 0.3)  # Base 1km + 0.3km per POI difference
                # Add some randomness to avoid identical distances
                import random
                random.seed(i * 1000 + j)  # Deterministic but varied
                variation = 0.5 + random.random() * 0.5  # 0.5 to 1.0 multiplier
                estimated_km = base_distance * variation
                
                distance_matrix[i][j] = estimated_km
                distance_matrix[j][i] = estimated_km
                print(f"       [ADAPTER] Fallback estimate {poi_names[i]} -> {poi_names[j]}: {estimated_km:.2f} km")
    
    # Create poi_indices list (sequential indices for the city subset)
    poi_indices = list(range(N))
    
    print(f"   [ADAPTER] Created complete distance matrix: {N}x{N} with {sum(1 for i in range(N) for j in range(N) if distance_matrix[i][j] != float('inf'))} valid distances")
    
    return pois_for_city, poi_indices, distance_matrix

# --------------------------
# Internal: converters to planner schema
# --------------------------
def _hotel_to_planner(h: Dict[str, Any], default_lat: float, default_lon: float) -> Dict[str, Any]:
    """
    Convert tool-agent hotel dict to the minimal shape the planner needs.
    Accepts either 'price_per_night'/'user_rating' (your sheet) or legacy 'price_band'/'rating'.
    Fills coords if missing with (default_lat, default_lon).
    """
    # Price
    if "price_per_night" in h:
        price = _num_from_money_or_text(h.get("price_per_night"), 0.0)
    elif "price" in h:
        price = _num_from_money_or_text(h.get("price"), 0.0)
    else:
        # Try 'price_band' (usually £££) -> can map to a rough numeric if you want; here: 0
        price = 0.0

    # Rating
    if "user_rating" in h:
        rating = _num_from_money_or_text(h.get("user_rating"), 0.0)
    else:
        rating = _num_from_money_or_text(h.get("rating"), 0.0)

    lat = _safe_float(h.get("latitude"), default_lat)
    lon = _safe_float(h.get("longitude"), default_lon)

    return {
        "name": h.get("name", "Hotel"),
        "price_per_night": price,
        "user_rating": rating,
        "latitude": lat,
        "longitude": lon,
    }

def _poi_to_planner(p: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert tool-agent POI to planner shape.
    Adds safe defaults (rating, duration, ticket price, meal heuristic).
    """
    # rating like "4.5/5" or number
    rating = _num_from_money_or_text(p.get("rating") or p.get("user_rating") or "4.4", 4.4)

    # duration default
    duration_min = 90

    # ticket price
    price_str = (p.get("price") or "").strip().lower()
    if price_str in {"", "free", "na", "n/a", "varies"}:
        price = 0.0
    else:
        price = _num_from_money_or_text(price_str, 0.0)

    # coords with sensible defaults (central London) if missing
    lat = _safe_float(p.get("latitude"), 51.5074)
    lon = _safe_float(p.get("longitude"), -0.1278)

    cat = (p.get("category") or p.get("type") or "Attraction").strip()
    is_meal = any(kw in cat.lower() for kw in ["food", "market", "street food"])

    return {
        "name": p.get("name") or p.get("poi_name") or "POI",
        "category": cat,
        "rating": rating,
        "duration_min": duration_min,
        "price": price,
        "latitude": lat,
        "longitude": lon,
        "is_meal": is_meal,
    }

# --------------------------
# Public: bundle -> planner data per city
# --------------------------
def bundle_city_to_planner_data(bundle: Dict[str, Any], city: str) -> Dict[str, Any]:
    """
    Transform the tool-agent bundle into the exact JSON the CP-SAT planner expects:
      {
        "hotels": [...],
        "pois": [...],
        "poi_poi_distance_km": [[...], ...]
      }
    * Preserves POI order consistent with the precomputed distance matrix.
    * For hotels with missing coords, uses the POI centroid as a fallback.
    """
    city = _clean_city(city)
    pois_for_city, idxs, dm = _select_city_subset(bundle, city)
    if not pois_for_city:
        return {"hotels": [], "pois": [], "poi_poi_distance_km": []}

    # POIs (keep lockstep with DM)
    pois_planner = [_poi_to_planner(p) for p in pois_for_city]

    # Compute a default (lat, lon) from POIs for hotels missing coords
    lat_vals = [p["latitude"] for p in pois_planner if p.get("latitude") is not None]
    lon_vals = [p["longitude"] for p in pois_planner if p.get("longitude") is not None]
    if lat_vals and lon_vals:
        default_lat = sum(lat_vals) / len(lat_vals)
        default_lon = sum(lon_vals) / len(lon_vals)
    else:
        # fallback: central London
        default_lat, default_lon = 51.5074, -0.1278

    # Hotels: prefer city-matching; if none, allow all and still fill coords
    hotels_all = bundle.get("hotels", []) or []
    hotels_city = [h for h in hotels_all if _clean_city(h.get("city", "")).casefold() == city.casefold()]
    chosen_hotels = hotels_city if hotels_city else hotels_all

    hotels_planner = [_hotel_to_planner(h, default_lat, default_lon) for h in chosen_hotels]
    
    # adapter.py (end of bundle_city_to_planner_data)  ─ update return:
    poi_mode_times_all = (bundle.get("poi_mode_times") or {}).get(city)
    hotel_poi_mode_all = (bundle.get("hotel_poi_mode_times") or {}).get(city)

    return {
        "hotels": hotels_planner,
        "pois": pois_planner,
        "poi_poi_distance_km": dm,
        # NEW: forward mode times keyed by canonical (lowercased) names
        "poi_mode_times": poi_mode_times_all,        # {(poi_i, poi_j)->{walk_min,...,bus_cnt,...,edge_km}}
        "hotel_poi_mode_times": hotel_poi_mode_all   # {hotel_name->{poi_name->{walk_min,...}}}
    }