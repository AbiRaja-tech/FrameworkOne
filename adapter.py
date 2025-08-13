# adapters.py
from typing import Dict, Any, List, Tuple

def _select_city_subset(bundle: Dict[str, Any], city: str) -> Tuple[List[Dict[str, Any]], List[int], List[List[float]]]:
    """
    Returns (pois_for_city, poi_indices_in_bundle_order, distance_matrix_for_those_indices).
    The order of pois matches the order of 'poi_indices' coming from Tool Agent.
    """
    dm_info = bundle["poi_distance_matrices"].get(city)
    if not dm_info:
        return [], [], []

    idxs: List[int] = dm_info["poi_indices"]
    # Tool agent stores flat POI list in bundle["pois"]
    all_pois = bundle["pois"]

    pois_for_city: List[Dict[str, Any]] = [all_pois[i] for i in idxs if i < len(all_pois)]
    dist = dm_info["distance_km"]
    return pois_for_city, idxs, dist


def _hotel_to_planner(h: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert tool-agent hotel record to intracity_planner's expected shape.
    """
    # price_per_night is a string in sheets; try to parse number, otherwise default 0
    p = h.get("price_per_night", "")
    try:
        # Strip currency symbols like "£139"
        pp = "".join(ch for ch in str(p) if ch.isdigit() or ch == "." or ch == "-")
        price = float(pp) if pp else 0.0
    except Exception:
        price = 0.0

    return {
        "name": h.get("name", "Hotel"),
        "price_per_night": price,
        "user_rating": float(str(h.get("user_rating","0")).split("/")[0]) if h.get("user_rating") else 0.0,
        # Many hotel rows won’t have lat/lon. If missing, the solver won’t use them anyway for back-to-hotel legs.
        # You can later enrich from Google/ORS geocoding.
        "latitude": float(h.get("latitude")) if h.get("latitude") not in (None, "", "NA") else 51.5074,
        "longitude": float(h.get("longitude")) if h.get("longitude") not in (None, "", "NA") else -0.1278,
    }


def _poi_to_planner(p: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert tool-agent POI row to intracity_planner's expected shape.
    Adds a few safe defaults (ratings, durations, price).
    """
    def _f(x, default):
        try:
            return float(x)
        except Exception:
            return default

    # rating: extract like "4.5/5" → 4.5
    raw_rating = p.get("rating") or p.get("user_rating") or ""
    rating = 4.4
    if isinstance(raw_rating, (int, float)):
        rating = float(raw_rating)
    elif isinstance(raw_rating, str) and "/" in raw_rating:
        try:
            rating = float(raw_rating.split("/")[0])
        except Exception:
            rating = 4.4

    # duration: if we don’t have one, use a conservative default
    duration_min = 90

    # ticket price: parse simple "free" / "£xx" / "varies"
    price_str = (p.get("price") or "").strip().lower()
    price = 0.0
    if price_str and price_str not in ("free", "na", "n/a", "varies"):
        try:
            price = float("".join(ch for ch in price_str if ch.isdigit() or ch == "." or ch == "-"))
        except Exception:
            price = 0.0

    return {
        "name": p.get("name") or p.get("poi_name") or "POI",
        "category": p.get("category") or "Attraction",
        "rating": rating,
        "duration_min": duration_min,
        "price": price,
        "latitude": _f(p.get("latitude"), 51.5074),
        "longitude": _f(p.get("longitude"), -0.1278),
        # meal heuristic: treat notable markets as meal-capable
        "is_meal": any(kw in (p.get("category","").lower()) for kw in ["food","market","street food"])
    }


def bundle_city_to_planner_data(bundle: Dict[str, Any], city: str) -> Dict[str, Any]:
    """
    Transform the tool agent bundle into the exact data json that intracity_planner.solve_intracity_city_day expects.
    {
      "hotels": [...],
      "pois": [...],
      "poi_poi_distance_km": [[...], ...]
    }
    """
    pois_for_city, idxs, dm = _select_city_subset(bundle, city)
    if not pois_for_city:
        return {"hotels": [], "pois": [], "poi_poi_distance_km": []}

    # Keep order in lockstep with the distance matrix (indexes from tool agent)
    pois_planner = [_poi_to_planner(p) for p in pois_for_city]

    # Hotels: take hotels in that city; if none, fall back to all
    hotels_all = bundle.get("hotels", [])
    hotels_city = [h for h in hotels_all if h.get("city","").strip().casefold() == city.casefold()]
    hotels_planner = [_hotel_to_planner(h) for h in (hotels_city if hotels_city else hotels_all)]

    return {
        "hotels": hotels_planner,
        "pois": pois_planner,
        "poi_poi_distance_km": dm
    }
