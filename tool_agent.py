# tool_agent.py
# --------------------------------------
# Tool Agent: Google Sheets fetchers + POI distance matrices
# --------------------------------------

import os, math, time, base64, logging
from pathlib import Path
from dataclasses import dataclass, asdict
from functools import lru_cache
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv
import json

# Google Sheets (optional)
try:
    import gspread
    from google.oauth2.service_account import Credentials
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False
    gspread = None
    Credentials = None

BASE_DIR = Path(__file__).resolve().parent
_ = load_dotenv(BASE_DIR / ".env") or load_dotenv()

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive.readonly",
]

def _resolve_service_account_path() -> Optional[str]:
    if not GOOGLE_AVAILABLE:
        return None
    path = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
    if path and Path(path).exists():
        return path
    b64 = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON_B64")
    if b64:
        try:
            raw = base64.b64decode(b64)
            outdir = BASE_DIR / ".secrets"
            outdir.mkdir(parents=True, exist_ok=True)
            outfile = outdir / "service_account.json"
            outfile.write_bytes(raw)
            return str(outfile)
        except Exception as e:
            logging.warning(f"Failed to decode GOOGLE_SERVICE_ACCOUNT_JSON_B64: {e}")
    return None

SERVICE_ACCOUNT_JSON = _resolve_service_account_path()

SPREADSHEET_NAME = os.getenv("TRAVEL_SPREADSHEET_NAME", "Travel Planning")
POI_SHEET        = os.getenv("POI_SHEET", "POIs")
HOTEL_SHEET      = os.getenv("HOTEL_SHEET", "Hotels")
RESTAURANT_SHEET = os.getenv("RESTAURANT_SHEET", "Restaurants")
POI_POI_SHEET   = os.getenv("POI_POI_SHEET", "POI_POI_Distance")
HOTEL_POI_SHEET = os.getenv("HOTEL_POI_SHEET", "Hotel_POI_Distance")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("tool_agent")

# -------------------- Data models --------------------
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
    category: str
    short_desc: str
    address: str
    latitude: Optional[float]
    longitude: Optional[float]
    price_band: str
    rating: str
    opening_hours: str

@dataclass
class Restaurant:
    city: str
    name: str
    category: str
    short_desc: str
    address: str
    latitude: Optional[float]
    longitude: Optional[float]
    price_band: str
    rating: str
    opening_hours: str

# tool_agent.py  (replace the HOTEL_COLS dict)
HOTEL_COLS = {
    "city": "city",
    "hotel_name": "name",            # <— map your header to name
    "star_rating": "category",       # <— reasonable place to store “4-star”
    "description": "short_desc",
    "price_per_night": "price_band", # <— keep as text (£139, £86, …)
    "user_rating": "rating",
    "latitude": "latitude",          # may not exist in the sheet; stays blank
    "longitude": "longitude",        # may not exist in the sheet; stays blank
    "opening_hours": "opening_hours" # optional / absent is fine
}


# -------------------- Helpers --------------------
def _get_gspread_client():
    if not GOOGLE_AVAILABLE or not SERVICE_ACCOUNT_JSON:
        return None
    try:
        t0 = time.perf_counter()
        creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_JSON, scopes=SCOPES)
        gc = gspread.authorize(creds)
        log.info(f"Authorized gspread in {time.perf_counter()-t0:.2f}s")
        return gc
    except Exception as e:
        log.warning(f"Google auth failed: {e}")
        return None

@lru_cache(maxsize=1)
def _open_sheet(spreadsheet_name: str):
    gc = _get_gspread_client()
    if not gc:
        return None
    try:
        t0 = time.perf_counter()
        sh = gc.open(spreadsheet_name)
        log.info(f"Opened spreadsheet '{spreadsheet_name}' in {time.perf_counter()-t0:.2f}s")
        return sh
    except Exception as e:
        log.warning(f"Open spreadsheet failed: {e}")
        return None

def _get_all_records(sheet_name: str) -> List[Dict[str, Any]]:
    sh = _open_sheet(SPREADSHEET_NAME)
    if not sh:
        return []
    try:
        t0 = time.perf_counter()
        ws = sh.worksheet(sheet_name)
        
        # Try to get all records normally first
        try:
            rows = ws.get_all_records()  # list[dict], headers normalized from row 1
            log.info(f"Read {len(rows)} rows from '{sheet_name}' in {time.perf_counter()-t0:.2f}s")
            return rows
        except Exception as e:
            if "duplicates" in str(e).lower():
                log.warning(f"Duplicate headers detected in '{sheet_name}', using manual parsing")
                return _get_all_records_manual(ws, sheet_name, t0)
            else:
                raise e
                
    except Exception as e:
        log.warning(f"Read sheet '{sheet_name}' failed: {e}")
        return []

def _get_all_records_manual(ws, sheet_name: str, t0: float) -> List[Dict[str, Any]]:
    """
    Manually read sheet data to handle duplicate headers.
    """
    try:
        # Get all values as raw data
        all_values = ws.get_all_values()
        if not all_values or len(all_values) < 2:
            return []
        
        # Get headers from first row
        headers = all_values[0]
        
        # Handle duplicate headers by appending numbers
        seen_headers = {}
        processed_headers = []
        
        for i, header in enumerate(headers):
            header_str = str(header).strip()
            if not header_str:
                header_str = f"column_{i}"
            
            if header_str in seen_headers:
                seen_headers[header_str] += 1
                header_str = f"{header_str}_{seen_headers[header_str]}"
            else:
                seen_headers[header_str] = 0
            
            processed_headers.append(header_str)
        
        print(f"   [MANUAL_READ] Original headers: {headers}")
        print(f"   [MANUAL_READ] Processed headers: {processed_headers}")
        
        # Process data rows
        rows = []
        for row_data in all_values[1:]:  # Skip header row
            if not row_data or all(not cell for cell in row_data):
                continue  # Skip empty rows
                
            # Create row dict, padding with empty strings if needed
            row_dict = {}
            for i, header in enumerate(processed_headers):
                value = row_data[i] if i < len(row_data) else ""
                row_dict[header] = value
            
            rows.append(row_dict)
        
        log.info(f"Manually read {len(rows)} rows from '{sheet_name}' in {time.perf_counter()-t0:.2f}s")
        return rows
        
    except Exception as e:
        log.warning(f"Manual read of '{sheet_name}' failed: {e}")
        return []

def _to_float(x: Any) -> Optional[float]:
    try:
        s = str(x).strip().replace("£", "").replace(",", "")
        if s == "" or s.lower() in {"na", "null"}:
            return None
        return float(s)
    except Exception:
        return None

def _pick(row: Dict[str, Any], *candidates: str, default="") -> Any:
    """
    Pick the first key present in row from candidates (case-insensitive).
    """
    lower = {k.lower(): k for k in row.keys()}
    for c in candidates:
        k = lower.get(c.lower())
        if k is not None:
            return row.get(k, default)
    return default

def _pick_num(row: Dict[str, Any], *candidates: str) -> Optional[float]:
    return _to_float(_pick(row, *candidates, default=""))

# tool_agent.py (add)
def _num_or_none(v):
    s = str(v).strip().lower()
    if not s or s in {"x","na","n/a","-"}: return None
    try: return float(s)
    except: return None

def _norm(s):  # casefolded key for matching POI/hotel names
    return str(s or "").strip().casefold()


# -------------------- Data loading --------------------
def _load_local_data(city: str) -> Optional[Dict[str, Any]]:
    """Load local data for cities not available in Google Sheets."""
    try:
        # Check for local data files
        local_files = {
            "Manchester": "manchester_data.json",
            "Edinburgh": "edinburgh_data.json",
            "York": "york_data.json",
            "Bath": "bath_data.json"
        }
        
        if city in local_files:
            file_path = BASE_DIR / local_files[city]
            if file_path.exists():
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    log.info(f"Loaded local data for {city} from {file_path}")
                    return data
    except Exception as e:
        log.warning(f"Failed to load local data for {city}: {e}")
    
    return None

# -------------------- Fetchers (robust mappings) --------------------
def fetch_pois(cities: List[str]) -> List[POI]:
    rows = _get_all_records(POI_SHEET)
    if not rows:
        log.warning("POI sheet empty or unavailable.")
    want = {c.casefold().strip() for c in cities}
    out: List[POI] = []
    for r in rows:
        city = str(_pick(r, "city")).strip()
        if city.casefold() not in want:
            continue
        poi = POI(
            city=city,
            name=str(_pick(r, "name", "poi_name")).strip(),
            category=str(_pick(r, "category")).strip(),
            short_desc=str(_pick(r, "short_desc", "description")).strip(),
            address=str(_pick(r, "address", "addr")).strip(),
            latitude=_pick_num(r, "latitude", "lat"),
            longitude=_pick_num(r, "longitude", "lon", "lng"),
            opening_hours=str(_pick(r, "opening_hours", "hours")).strip(),
            price=str(_pick(r, "price", "ticket_price")).strip(),
            booking=str(_pick(r, "booking", "booking_required")).strip(),
        )
        out.append(poi)
    log.info(f"fetch_pois → {len(out)} rows (cities={cities})")
    return out

def fetch_hotels(cities: List[str]) -> List[Hotel]:
    rows = _get_all_records(HOTEL_SHEET)
    if not rows:
        log.warning("Hotel sheet empty or unavailable.")
    want = {c.casefold().strip() for c in cities}
    out: List[Hotel] = []
    for r in rows:
        city = str(_pick(r, "city")).strip()
        if city.casefold() not in want:
            continue
        hotel = Hotel(
            city=city,
            name=str(_pick(r, "name", "hotel_name")).strip(),
            category=str(_pick(r, "category", "star_rating")).strip(),
            short_desc=str(_pick(r, "short_desc", "description")).strip(),
            address=str(_pick(r, "address")).strip(),
            latitude=_pick_num(r, "latitude", "lat"),
            longitude=_pick_num(r, "longitude", "lon", "lng"),
            # many sheets have price_per_night; map it into price_band if price_band missing
            price_band=str(_pick(r, "price_band", "price_per_night")).strip(),
            rating=str(_pick(r, "rating", "user_rating")).strip(),
            opening_hours=str(_pick(r, "opening_hours", "hours", "24/7")).strip() or "24/7",
        )
        out.append(hotel)
    log.info(f"fetch_hotels → {len(out)} rows (cities={cities})")
    return out

def fetch_restaurants(cities: List[str]) -> List[Restaurant]:
    rows = _get_all_records(RESTAURANT_SHEET)
    if not rows:
        log.info("Restaurants sheet empty — skipping (no mock fallback here).")
        return []
    want = {c.casefold().strip() for c in cities}
    out: List[Restaurant] = []
    for r in rows:
        city = str(_pick(r, "city")).strip()
        if city.casefold() not in want:
            continue
        rest = Restaurant(
            city=city,
            name=str(_pick(r, "name", "restaurant_name")).strip(),
            category=str(_pick(r, "category", "cuisine")).strip(),
            short_desc=str(_pick(r, "short_desc", "description")).strip(),
            address=str(_pick(r, "address")).strip(),
            latitude=_pick_num(r, "latitude", "lat"),
            longitude=_pick_num(r, "longitude", "lon", "lng"),
            price_band=str(_pick(r, "price_band", "price_range")).strip(),
            rating=str(_pick(r, "rating", "user_rating")).strip(),
            opening_hours=str(_pick(r, "opening_hours", "hours")).strip(),
        )
        out.append(rest)
    log.info(f"fetch_restaurants → {len(out)} rows (cities={cities})")
    return out

# tool_agent.py (add)
def fetch_poi_poi_mode_times(cities: List[str]):
    rows = _get_all_records(POI_POI_SHEET)
    want = {c.casefold().strip() for c in cities}
    # Map: city -> {(src,dst) -> {mode_time_min, mode_count, edge_km}}
    per_city = {}
    
    if not rows:
        log.warning(f"No data found in {POI_POI_SHEET}")
        return per_city
    
    # Get the processed headers from the manual read function
    # We need to access the actual header names, not the first data row
    processed_headers = []
    if hasattr(rows, '_processed_headers'):
        processed_headers = rows._processed_headers
    else:
        # Fallback: try to get headers from the first row keys
        if rows:
            processed_headers = list(rows[0].keys())
    
    # Handle duplicate headers by checking if we have tram_time twice
    tram_time_columns = [i for i, h in enumerate(processed_headers) if 'tram_time' in str(h).lower()]
    
    if len(tram_time_columns) > 1:
        log.warning(f"Found duplicate tram_time headers at columns {tram_time_columns}")
        # Use the first tram_time for time, second for count
        tram_time_col = tram_time_columns[0]
        tram_count_col = tram_time_columns[1]
    else:
        tram_time_col = None
        tram_count_col = None
        # Try to find tram_count column (including the new _1, _2 suffixes)
        for i, h in enumerate(processed_headers):
            if any(pattern in str(h).lower() for pattern in ['tram_count', 'tram_transfer', 'tram_time_1', 'tram_time_2']):
                tram_count_col = i
                break
    
    print(f"   [TOOL_AGENT] Processing POI-POI sheet with {len(rows)} rows")
    print(f"   [TOOL_AGENT] Processed headers: {processed_headers}")
    print(f"   [TOOL_AGENT] Tram time columns: {tram_time_columns}")
    print(f"   [TOOL_AGENT] Using tram_time_col: {tram_time_col}, tram_count_col: {tram_count_col}")
    
    for r in rows:  # Process all rows (first row is data, not headers)
        city = str(_pick(r,"city")).strip()
        if city.casefold() not in want: 
            continue
            
        src = _norm(_pick(r,"poi","start","start point","start_point"))
        dst = _norm(_pick(r,"end point","end","end_point","poi_2"))
        if not src or not dst or src == dst: 
            continue
            
        # Use string key instead of tuple for JSON serialization
        key = f"{src}->{dst}"
        d = per_city.setdefault(city, {}).setdefault(key, {})
        
        # Enhanced time parsing with validation - using actual column names from sheets
        d["walk_min"] = _num_or_none(_pick(r,"walk_time","walk_time(mins)","walk_min"))
        d["bus_min"]  = _num_or_none(_pick(r,"bus_time","bus_min"))
        
        # Handle tram time from the correct column
        if tram_time_col is not None and tram_time_col < len(processed_headers):
            header_name = processed_headers[tram_time_col]
            d["tram_min"] = _num_or_none(r.get(header_name))
        else:
            d["tram_min"] = _num_or_none(_pick(r,"tram_time","tube_time"))
        
        d["metro_min"]= d["tram_min"]  # alias
        d["cab_min"]  = _num_or_none(_pick(r,"cab time","cab_time","taxi_time"))
        
        # Enhanced count parsing - handle the duplicate header issue
        d["bus_cnt"]  = _num_or_none(_pick(r,"bus_count","bus_cnt")) or 0
        
        # Handle tram count from duplicate header or separate column
        if tram_count_col is not None and tram_count_col < len(processed_headers):
            header_name = processed_headers[tram_count_col]
            d["tram_cnt"] = _num_or_none(r.get(header_name)) or 0
        else:
            d["tram_cnt"] = _num_or_none(_pick(r,"tram_count","tram_cnt")) or 0
        
        # Enhanced distance parsing - try multiple column names from actual sheets
        edge_km = None
        for k in ("walk_distance","walk_distance (r)","walk_distance (km)","cab_distance","cab distance","bus_distance","tram_distance"):
            km = _num_or_none(_pick(r,k))
            if km is not None:
                edge_km = km
                break
        
        # If no explicit distance, try to derive from walk time and speed
        if edge_km is None and d["walk_min"] is not None:
            # Assume 4.5 km/h walking speed
            edge_km = (d["walk_min"] / 60.0) * 4.5
            
        d["edge_km"] = edge_km
        
        # Validate data quality
        if d["walk_min"] is not None and d["walk_min"] > 0:
            d["walk_available"] = True
        else:
            d["walk_available"] = False
            
        if d["bus_min"] is not None and d["bus_min"] > 0:
            d["bus_available"] = True
        else:
            d["bus_available"] = False
            
        if d["tram_min"] is not None and d["tram_min"] > 0:
            d["tram_available"] = True
        else:
            d["tram_available"] = False
            
        if d["cab_min"] is not None and d["cab_min"] > 0:
            d["cab_available"] = True
        else:
            d["cab_available"] = False
    
    log.info(f"Fetched POI-POI mode times for {len(per_city)} cities")
    return per_city


# tool_agent.py (enhance fetch_hotel_poi_mode_times)
def fetch_hotel_poi_mode_times(cities: List[str]):
    rows = _get_all_records(HOTEL_POI_SHEET)
    want = {c.casefold().strip() for c in cities}
    per_city = {}
    
    if not rows:
        log.warning(f"No data found in {HOTEL_POI_SHEET}")
        return per_city
    
    for r in rows[1:]:  # Skip header row
        # Hotel tab often doesn't repeat city; derive from hotel name row if present, else accept
        hotel = _norm(_pick(r,"hotel","hotel_name","name"))
        poi   = _norm(_pick(r,"poi","poi_name","attraction"))
        if not hotel or not poi: 
            continue
            
        city = str(_pick(r,"city")).strip() or ""  # optional
        if city and city.casefold() not in want: 
            continue
        city = city or cities[0]  # best effort
        
        d = per_city.setdefault(city, {}).setdefault(hotel, {}).setdefault(poi, {})
        
        # Enhanced time parsing - using actual column names from sheets
        d["walk_min"] = _num_or_none(_pick(r,"Walk_time","walk_time","walk_time(mins)","walk_min"))
        d["bus_min"]  = _num_or_none(_pick(r,"Bus_time","bus_time","bus_min"))
        d["tram_min"] = _num_or_none(_pick(r,"Tram_time","tram_time","tram/tube_time","tube_time"))
        d["metro_min"]= d["tram_min"]
        d["cab_min"]  = _num_or_none(_pick(r,"Cab_time","cab_time","taxi_time"))
        
        # Enhanced count parsing - using actual column names from sheets
        d["bus_cnt"]  = _num_or_none(_pick(r,"Bus_count","bus_count","bus_cnt")) or 0
        d["tram_cnt"] = _num_or_none(_pick(r,"Tram_count","tram_count","tram/tube_count","tube_count")) or 0
        
        # Enhanced distance parsing - using actual column names from sheets
        edge_km = None
        for k in ("Walk_dist","walk_dist","walk_distance","walk_distance (r)","Cab_dist","cab_dist","cab_distance","bus_dist","bus_distance","tram_dist","tram_distance"):
            km = _num_or_none(_pick(r,k))
            if km is not None:
                edge_km = km
                break
                
        # If no explicit distance, try to derive from walk time and speed
        if edge_km is None and d["walk_min"] is not None:
            # Assume 4.5 km/h walking speed
            edge_km = (d["walk_min"] / 60.0) * 4.5
            
        d["edge_km"] = edge_km
        
        # Validate data quality
        if d["walk_min"] is not None and d["walk_min"] > 0:
            d["walk_available"] = True
        else:
            d["walk_available"] = False
            
        if d["bus_min"] is not None and d["bus_min"] > 0:
            d["bus_available"] = True
        else:
            d["bus_available"] = False
            
        if d["tram_min"] is not None and d["tram_min"] > 0:
            d["tram_available"] = True
        else:
            d["tram_available"] = False
            
        if d["cab_min"] is not None and d["cab_min"] > 0:
            d["cab_available"] = True
        else:
            d["cab_available"] = False
    
    log.info(f"Fetched Hotel-POI mode times for {len(per_city)} cities")
    return per_city


# -------------------- Distances --------------------
def _haversine_km(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return 6371.0088 * (2 * math.asin(math.sqrt(h)))

def distance_matrix(points: List[Tuple[float, float]]) -> List[List[float]]:
    n = len(points)
    m = [[0.0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            d = _haversine_km(points[i], points[j])
            m[i][j] = m[j][i] = d
    return m

# -------------------- City inference --------------------
def _infer_cities_from_policy(policy: Dict[str, Any]) -> List[str]:
    if isinstance(policy.get("cities"), dict):
        return list(policy["cities"].keys())
    if isinstance(policy.get("city_sequence"), list):
        return [str(x) for x in policy["city_sequence"] if isinstance(x, (str, int))]
    if isinstance(policy.get("hotel_location"), dict):
        desc = policy["hotel_location"].get("description")
        if isinstance(desc, str) and desc.strip():
            return [desc.strip()]
    
    # Try to extract city from query if available
    query = policy.get("query", "")
    if query:
        query_lower = query.lower()
        if "manchester" in query_lower:
            return ["Manchester"]
        elif "edinburgh" in query_lower:
            return ["Edinburgh"]
        elif "york" in query_lower:
            return ["York"]
        elif "bath" in query_lower:
            return ["Bath"]
        elif "cardiff" in query_lower:
            return ["Cardiff"]
        elif "brighton" in query_lower:
            return ["Brighton"]
        elif "london" in query_lower:
            return ["London"]
    
    # Default to London only if no other city is specified
    return ["London"]

# -------------------- Main entry --------------------
def run_tool_agent(policy: Dict[str, Any]) -> Dict[str, Any]:
    t0 = time.perf_counter()
    cities = [c.strip() for c in _infer_cities_from_policy(policy) if c]
    log.info(f"ToolAgent: target cities → {cities}")

    # Check if we have local data for any cities
    local_data = None
    for city in cities:
        local_data = _load_local_data(city)
        if local_data:
            log.info(f"Using local data for {city}")
            break
    
    if local_data:
        # Use local data instead of fetching from Google Sheets
        return local_data

    # Fall back to Google Sheets fetching
    pois = fetch_pois(cities)
    hotels = fetch_hotels(cities)
    restaurants = fetch_restaurants(cities)

    # per-city distance matrices for POIs
    poi_points: List[Tuple[float, float]] = []
    poi_city_of: List[str] = []
    city_idx: Dict[str, List[int]] = {}
    for p in pois:
        if p.latitude is None or p.longitude is None:
            continue
        poi_points.append((p.latitude, p.longitude))
        poi_city_of.append(p.city)
        city_idx.setdefault(p.city, []).append(len(poi_points)-1)

    dm_all = distance_matrix(poi_points)
    per_city_dm: Dict[str, Any] = {}
    for city, idxs in city_idx.items():
        sub = [[dm_all[i][j] for j in idxs] for i in idxs]
        per_city_dm[city] = {"poi_indices": idxs, "distance_km": sub}

    bundle = {
        "cities": cities,
        "pois": [asdict(p) for p in pois],
        "hotels": [asdict(h) for h in hotels],
        "restaurants": [asdict(r) for r in restaurants],
        "poi_distance_matrices": per_city_dm,
        "meta": {
            "source": "google_sheets" if (GOOGLE_AVAILABLE and SERVICE_ACCOUNT_JSON) else "local",
            "spreadsheet": SPREADSHEET_NAME,
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    }
    log.info(f"ToolAgent finished in {time.perf_counter()-t0:.2f}s")
    # tool_agent.py (inside run_tool_agent, before return)  ─ add:
    poi_mode = fetch_poi_poi_mode_times(cities)
    hotel_poi_mode = fetch_hotel_poi_mode_times(cities)
    bundle.update({
    "poi_mode_times": poi_mode,
    "hotel_poi_mode_times": hotel_poi_mode
    })
    return bundle

# -------------------- CLI smoke test --------------------
if __name__ == "__main__":
    import argparse, json as _json
    ap = argparse.ArgumentParser()
    ap.add_argument("--cities", type=str, default="London")
    args = ap.parse_args()
    policy = {"city_sequence": [c.strip() for c in args.cities.split(",")]}
    out = run_tool_agent(policy)
    print(_json.dumps(out, indent=2, ensure_ascii=False))
