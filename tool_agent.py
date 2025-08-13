# tool_agent.py
# --------------------------------------
# Tool Agent: Google Sheets fetchers + POI distance matrices
# --------------------------------------

import os
import math
import time
import base64
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from functools import lru_cache
from typing import List, Dict, Any, Optional, Tuple

# .env loader
from dotenv import load_dotenv

# Google Sheets
import gspread
from google.oauth2.service_account import Credentials


# ============================
# .env / Config
# ============================
BASE_DIR = Path(__file__).resolve().parent
_ = load_dotenv(BASE_DIR / ".env") or load_dotenv()  # try local .env first, then cwd

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive.readonly"
]

def _resolve_service_account_path() -> str:
    """
    Resolves a filesystem path to service-account JSON.
    Priority:
      1) GOOGLE_SERVICE_ACCOUNT_JSON (path)
      2) GOOGLE_SERVICE_ACCOUNT_JSON_B64 (base64-encoded JSON content) -> writes ./ .secrets/service_account.json
    """
    path = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
    if path:
        p = Path(path)
        if p.exists():
            return str(p)

    b64 = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON_B64")
    if b64:
        try:
            raw = base64.b64decode(b64)
            secrets_dir = BASE_DIR / ".secrets"
            secrets_dir.mkdir(parents=True, exist_ok=True)
            outfile = secrets_dir / "service_account.json"
            outfile.write_bytes(raw)
            return str(outfile)
        except Exception as e:
            raise RuntimeError(f"Failed to decode GOOGLE_SERVICE_ACCOUNT_JSON_B64: {e}")

    raise RuntimeError(
        "No Google credentials found. Set GOOGLE_SERVICE_ACCOUNT_JSON (file path) "
        "or GOOGLE_SERVICE_ACCOUNT_JSON_B64 (base64 of the JSON) in your .env."
    )

SERVICE_ACCOUNT_JSON = _resolve_service_account_path()

SPREADSHEET_NAME    = os.getenv("TRAVEL_SPREADSHEET_NAME", "Travel Planning")
POI_SHEET           = os.getenv("POI_SHEET", "POIs")
HOTEL_SHEET         = os.getenv("HOTEL_SHEET", "Hotels")
RESTAURANT_SHEET    = os.getenv("RESTAURANT_SHEET", "Restaurants")  # optional tab
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY", "")           # reserved for future travel-time tool


# ============================
# Logging
# ============================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("tool_agent")


# ============================
# Data Models
# ============================
@dataclass
class POI:
    city: str
    name: str
    category: str
    short_desc: str
    address: str
    latitude: Optional[float]
    longitude: Optional[float]
    opening_hours: str
    price: str
    booking: str

@dataclass
class Hotel:
    city: str
    name: str
    star_rating: str
    price_per_night: str
    user_rating: str
    description: str

@dataclass
class Restaurant:
    city: str
    name: str
    cuisine: str
    price_band: str
    rating: str
    address: str
    latitude: Optional[float]
    longitude: Optional[float]
    opening_hours: str


# ============================
# Column Maps (edit if your headers differ)
# ============================
POI_COLS = {
    "city": "city",
    "poi_name": "name",
    "category": "category",
    "short_desc": "short_desc",
    "address": "address",
    "latitude": "latitude",
    "longitude": "longitude",
    "opening_hours": "opening_hours",
    "price": "price",
    "booking": "booking",
}
HOTEL_COLS = {
    "hotel_name": "name",
    "star_rating": "star_rating",
    "price_per_night": "price_per_night",
    "user_rating": "user_rating",
    "description": "description",
    "city": "city",
}
RESTAURANT_COLS = {
    "city": "city",
    "name": "name",
    "cuisine": "cuisine",
    "price_band": "price_band",
    "rating": "rating",
    "address": "address",
    "latitude": "latitude",
    "longitude": "longitude",
    "opening_hours": "opening_hours",
}


# ============================
# Google Sheets helpers
# ============================
@lru_cache(maxsize=1)
def _get_gspread_client():
    t0 = time.perf_counter()
    creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_JSON, scopes=SCOPES)
    gc = gspread.authorize(creds)
    log.info(f"Authorized gspread in {time.perf_counter()-t0:.2f}s")
    return gc

@lru_cache(maxsize=1)
def _open_sheet(spreadsheet_name: str):
    t0 = time.perf_counter()
    sh = _get_gspread_client().open(spreadsheet_name)
    log.info(f"Opened spreadsheet '{spreadsheet_name}' in {time.perf_counter()-t0:.2f}s")
    return sh

def _get_all_records(sheet_name: str) -> List[Dict[str, Any]]:
    t0 = time.perf_counter()
    ws = _open_sheet(SPREADSHEET_NAME).worksheet(sheet_name)
    rows = ws.get_all_records()  # list[dict]
    log.info(f"Read {len(rows)} rows from '{sheet_name}' in {time.perf_counter()-t0:.2f}s")
    return rows

def _to_float(x: Any) -> Optional[float]:
    try:
        s = str(x).strip()
        if s == "" or s.lower() == "na" or s.lower() == "null":
            return None
        return float(s)
    except Exception:
        return None


# ============================
# Fetchers
# ============================
def fetch_pois(cities: List[str]) -> List[POI]:
    rows = _get_all_records(POI_SHEET)
    want = {c.casefold() for c in cities}
    out: List[POI] = []
    for r in rows:
        if r.get("city", "").strip().casefold() in want:
            data: Dict[str, Any] = {}
            for col, field in POI_COLS.items():
                v = r.get(col, "")
                if field in ("latitude", "longitude"):
                    v = _to_float(v)
                data[field] = v
            out.append(POI(**data))
    log.info(f"fetch_pois → {len(out)} rows (cities={cities})")
    return out

def fetch_hotels(cities: List[str]) -> List[Hotel]:
    rows = _get_all_records(HOTEL_SHEET)
    want = {c.casefold() for c in cities}
    out: List[Hotel] = []
    for r in rows:
        if r.get("city", "").strip().casefold() in want:
            data = {field: r.get(col, "") for col, field in HOTEL_COLS.items()}
            out.append(Hotel(**data))
    log.info(f"fetch_hotels → {len(out)} rows (cities={cities})")
    return out

def fetch_restaurants(cities: List[str]) -> List[Restaurant]:
    try:
        rows = _get_all_records(RESTAURANT_SHEET)
    except gspread.exceptions.WorksheetNotFound:
        log.warning("Restaurants worksheet not found; returning empty list.")
        return []

    want = {c.casefold() for c in cities}
    out: List[Restaurant] = []
    for r in rows:
        if r.get("city", "").strip().casefold() in want:
            data: Dict[str, Any] = {}
            for col, field in RESTAURANT_COLS.items():
                v = r.get(col, "")
                if field in ("latitude", "longitude"):
                    v = _to_float(v)
                data[field] = v
            out.append(Restaurant(**data))
    log.info(f"fetch_restaurants → {len(out)} rows (cities={cities})")
    return out


# ============================
# Distances (Haversine)
# ============================
def _haversine_km(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    # a=(lat, lon), b=(lat, lon)
    R = 6371.0088
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * R * math.asin(math.sqrt(h))

def distance_matrix(points: List[Tuple[float, float]]) -> List[List[float]]:
    """
    Symmetric distance matrix in km using Haversine. Fast and API-free.
    Swap this for Google Distance Matrix/OSRM later if you want travel times.
    """
    n = len(points)
    M = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = _haversine_km(points[i], points[j])
            M[i][j] = M[j][i] = d
    return M


# ============================
# Tool Agent entrypoint
# ============================
def _infer_cities_from_policy(policy: Dict[str, Any]) -> List[str]:
    if not isinstance(policy, dict):
        raise ValueError("Policy must be a dict.")

    if "cities" in policy and isinstance(policy["cities"], dict):
        return list(policy["cities"].keys())

    if "city_sequence" in policy and isinstance(policy["city_sequence"], list):
        return [str(x) for x in policy["city_sequence"] if isinstance(x, (str, int))]

    # simple intracity fallback
    if "hotel_location" in policy and isinstance(policy["hotel_location"], dict):
        desc = policy["hotel_location"].get("description")
        if isinstance(desc, str) and desc.strip():
            return [desc.strip()]

    # last-resort default
    return ["London"]

def run_tool_agent(policy: Dict[str, Any]) -> Dict[str, Any]:
    """
    Input: policy JSON from commonsense agent.
    Output: dict with normalized lists and POI distance matrices per city.
    """
    t0 = time.perf_counter()
    cities = [c.strip() for c in _infer_cities_from_policy(policy) if c and isinstance(c, str)]
    log.info(f"ToolAgent: target cities → {cities}")

    pois = fetch_pois(cities)
    hotels = fetch_hotels(cities)
    restaurants = fetch_restaurants(cities)

    # Build per-city POI distance matrices
    poi_points: List[Tuple[float, float]] = []
    poi_city_of: List[str] = []
    city_poi_index: Dict[str, List[int]] = {}

    for p in pois:
        if p.latitude is None or p.longitude is None:
            continue
        poi_points.append((p.latitude, p.longitude))
        poi_city_of.append(p.city)
        city_poi_index.setdefault(p.city, []).append(len(poi_points) - 1)

    dm_all = distance_matrix(poi_points)

    per_city_dm: Dict[str, Any] = {}
    for city, idxs in city_poi_index.items():
        sub = [[dm_all[i][j] for j in idxs] for i in idxs]
        per_city_dm[city] = {
            "poi_indices": idxs,
            "distance_km": sub,
        }

    bundle = {
        "cities": cities,
        "pois": [asdict(p) for p in pois],
        "hotels": [asdict(h) for h in hotels],
        "restaurants": [asdict(r) for r in restaurants],
        "poi_distance_matrices": per_city_dm,
        "meta": {
            "source": "google_sheets",
            "spreadsheet": SPREADSHEET_NAME,
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    }
    log.info(f"ToolAgent finished in {time.perf_counter()-t0:.2f}s")
    return bundle


# ============================
# CLI (smoke test)
# ============================
if __name__ == "__main__":
    import argparse, json as _json

    ap = argparse.ArgumentParser(description="Tool Agent: fetch data from Google Sheets and build POI distance matrices.")
    ap.add_argument("--policy", type=str, help="Path to a JSON file containing a policy.")
    ap.add_argument("--cities", type=str, help="Comma-separated city list (overrides policy).")
    ap.add_argument("--print", action="store_true", help="Print a compact summary.")
    args = ap.parse_args()

    if args.cities:
        cities = [c.strip() for c in args.cities.split(",") if c.strip()]
        policy = {"city_sequence": cities}
    elif args.policy:
        with open(args.policy, "r", encoding="utf-8") as f:
            policy = _json.load(f)
    else:
        # default quick test
        policy = {"city_sequence": ["London", "Manchester", "Edinburgh"]}

    out = run_tool_agent(policy)

    if args.print:
        print("Cities:", out["cities"])
        print("POIs:", len(out["pois"]), "Hotels:", len(out["hotels"]), "Restaurants:", len(out["restaurants"]))
        for c, dm in out["poi_distance_matrices"].items():
            n = len(dm["poi_indices"])
            print(f"Distance matrix: {c} — {n}x{n}")
    else:
        print(_json.dumps(out, indent=2, ensure_ascii=False))
