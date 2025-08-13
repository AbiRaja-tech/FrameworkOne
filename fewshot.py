# Simple routing hints per category (cheap & transparent).
CATEGORY_KEYWORDS = {
    "single_male_intracity": ["single", "male", "solo", "intracity", "within city", "london only"],
    "single_female_intracity": ["single", "female", "solo", "intracity", "within city"],
    "single_female_intercity": ["single", "female", "intercity", "multiple cities", "train", "rail"],
    "couple_intercity": ["couple", "intercity", "train", "rail"],
    "couple_combination": ["couple", "combination", "intercity", "intracity", "london", "york", "edinburgh"],
    "family_combination_children_l10": ["family", "children", "kids", "stroller", "combination"],
    "family_teens_intracity": ["family", "teens", "intracity"],
    "family_elderly_combination": ["elderly", "grandparent", "assistance", "combination"],
    "family_baby_intracity": ["baby", "infant", "stroller", "intracity", "gentle"],
    "group_male_intercity": ["group", "male", "friends", "intercity", "nightlife", "football"],
    "group_female_combination": ["group", "female", "combination", "spa", "shopping"],
    "family_multigen_intercity": ["multigenerational", "grandparents", "kids", "intercity"],
    "elderly_healthy_intracity": ["elderly", "healthy", "intracity", "york"],
    "elderly_assistance_combination": ["wheelchair", "assistance", "combination", "accessible"],
    "mixed_group_combination": ["mixed", "friends", "family", "teen", "combination", "cardiff"],
}

EXAMPLES = [
  #1 Single Male — Intracity (London)
  {
    "id": "1",
    "category": "single_male_intracity",
    "role": "user",
    "content": "4-day London trip for a single 32-year-old male, loves history and museums, medium budget, prefers walking and Tube. Only intracity travel."
  },
  {
    "id": "1",
    "category": "single_male_intracity",
    "role": "assistant",
    "content": {
      "pacing": "standard",
      "stamina_profile": "high",
      "budget_band": "medium",
      "max_pois_per_day": 5,
      "min_dwell_minutes_per_poi": 45,
      "max_total_walk_minutes_per_day": 200,
      "max_single_walk_minutes": 30,
      "meal_windows": { "breakfast": ["08:00","09:00"], "lunch": ["12:00","13:30"], "dinner": ["18:00","20:00"] },
      "quiet_hours": ["22:00","07:30"],
      "transport_policy": ["tube","walk_if_<2km","bus"],
      "weather_fallback_policy": ["prefer_indoor_museums","swap_days_if_rain"],
      "accessibility": { "child_friendly": False, "wheelchair_accessible": False },
      "must_include": ["British Museum","Imperial War Museum","Westminster cluster"],
      "must_avoid": ["theme_parks"],
      "hard_windows": [],
      "timeboxing": { "buffer_minutes_between_pois": 15 },
      "hotel_location": { "description": "Bloomsbury", "latitude": 51.52, "longitude": -0.125 },
      "max_one_way_distance_from_hotel_km": 5,
      "max_distance_from_hotel_per_day_km": 10,
      "avoid_backtracking": True,
      "local_transport": ["walk","tube","bus"],
      "passes_and_payments": { "contactless_or_oyster": True, "zone_cap_strategy": "zones_1_2_if_possible" }
    }
  },

  #2 Couple — Intercity (London, Manchester, Edinburgh)
  {
    "id": "2",
    "category": "couple_intercity",
    "role": "user",
    "content": "7-day UK trip for a couple (aged 29 and 31), visiting London, Manchester, Edinburgh. Medium budget, prefers rail, avoid overnight travel. Intercity only."
  },
  {
    "id": "2",
    "category": "couple_intercity",
    "role": "assistant",
    "content": {
      "city_sequence": ["London","Manchester","Edinburgh"],
      "city_nights": { "London": 3, "Manchester": 2, "Edinburgh": 2 },
      "intercity_prefs": {
        "preferred_mode_order": ["rail"],
        "max_leg_hours": 5,
        "depart_window": ["09:00","11:00"],
        "arrive_window": ["15:00","18:00"],
        "buffer_minutes_at_station": 20,
        "seat_pref": "table",
        "luggage": { "bags": 2, "heavy": False },
        "railcard": "TwoTogether",
        "avoid_overnight_travel": True
      },
      "ticketing_rules": { "prefer_advance_fares": True, "avoid_multiple_changes": True, "split_tickets_ok": False },
      "station_preferences": {
        "London": ["Euston","Kings Cross"],
        "Manchester": ["Piccadilly"],
        "Edinburgh": ["Waverley"]
      },
      "rest_strategy": { "light_day_after_leg_hours_gt": 3 },
      "sunday_service_awareness": True
    }
  },

  #3 Family with young children (<10) — Combination (Manchester, London, York, Edinburgh)
  {
    "id": "3",
    "category": "family_combination_children_l10",
    "role": "user",
    "content": "10-day UK trip for a family (parents 35, 33, children 8 and 5), starting Manchester, visiting London, York, Edinburgh, ending in London. Medium budget, prefers rail between cities, stroller-friendly. Combination intercity and intracity."
  },
  {
    "id": "3",
    "category": "family_combination_children_l10",
    "role": "assistant",
    "content": {
      "city_sequence": ["Manchester","London","York","Edinburgh","London"],
      "city_nights": { "Manchester": 2, "London": 4, "York": 1, "Edinburgh": 3 },
      "intercity_prefs": {
        "preferred_mode_order": ["rail"],
        "max_leg_hours": 4,
        "depart_window": ["09:00","10:30"],
        "arrive_window": ["14:00","18:00"],
        "buffer_minutes_at_station": 25,
        "seat_pref": "forward_facing",
        "luggage": { "bags": 3, "heavy": True },
        "railcard": "Family&Friends",
        "avoid_overnight_travel": True
      },
      "ticketing_rules": { "prefer_advance_fares": True, "avoid_multiple_changes": True, "split_tickets_ok": False },
      "station_preferences": {
        "London": ["Kings Cross","Euston"],
        "Manchester": ["Piccadilly"],
        "York": ["York"],
        "Edinburgh": ["Waverley"]
      },
      "rest_strategy": { "light_day_after_leg_hours_gt": 3 },
      "sunday_service_awareness": True,
      "cities": {
        "Manchester": {
          "hotel_location": { "description": "City Centre", "latitude": 53.48, "longitude": -2.24 },
          "max_one_way_distance_from_hotel_km": 3,
          "max_distance_from_hotel_per_day_km": 8,
          "avoid_backtracking": True,
          "local_transport": ["walk","Metrolink","bus"],
          "passes_and_payments": { "use_metrolink": True },
          "pacing": "standard",
          "max_pois_per_day": 5,
          "min_dwell_minutes_per_poi": 45,
          "meal_windows": { "breakfast": ["08:00","09:30"], "lunch": ["12:00","13:30"], "dinner": ["18:00","19:30"] },
          "quiet_hours": ["20:30","07:30"],
          "weather_fallback_policy": ["prefer_indoor_museums","swap_days_if_rain"],
          "must_include": ["Science & Industry Museum","family park"],
          "must_avoid": ["late-night events"],
          "accessibility": { "child_friendly": True, "wheelchair_accessible": True },
          "timeboxing": { "buffer_minutes_between_pois": 20 }
        },
        "London": {
          "hotel_location": { "description": "South Bank", "latitude": 51.50, "longitude": -0.11 },
          "max_one_way_distance_from_hotel_km": 5,
          "max_distance_from_hotel_per_day_km": 9,
          "avoid_backtracking": True,
          "local_transport": ["walk","tube","bus"],
          "passes_and_payments": { "contactless_or_oyster": True, "zone_cap_strategy": "zones_1_2_if_possible" },
          "pacing": "standard",
          "max_pois_per_day": 5,
          "min_dwell_minutes_per_poi": 50,
          "meal_windows": { "breakfast": ["08:00","09:30"], "lunch": ["12:00","13:30"], "dinner": ["18:00","20:00"] },
          "quiet_hours": ["20:30","07:30"],
          "weather_fallback_policy": ["prefer_indoor_museums","swap_days_if_rain"],
          "must_include": ["Natural History Museum","Hyde Park"],
          "must_avoid": ["high-traffic nightlife"],
          "accessibility": { "child_friendly": True, "wheelchair_accessible": True },
          "timeboxing": { "buffer_minutes_between_pois": 20 }
        }
      }
    }
  },

  #4 Single Female — Intracity (London)
  {
    "id": "4",
    "category": "single_female_intracity",
    "role": "user",
    "content": "5-day London trip for a solo 27-year-old female traveler, enjoys art galleries, boutique shopping, and quiet cafés. Prefers public transport and walking, medium budget. Only intracity travel."
  },
  {
    "id": "4",
    "category": "single_female_intracity",
    "role": "assistant",
    "content": {
      "pacing": "standard",
      "stamina_profile": "medium",
      "budget_band": "medium",
      "max_pois_per_day": 5,
      "min_dwell_minutes_per_poi": 45,
      "max_total_walk_minutes_per_day": 150,
      "max_single_walk_minutes": 25,
      "meal_windows": { "breakfast": ["08:30","09:30"], "lunch": ["12:30","14:00"], "dinner": ["18:30","20:00"] },
      "quiet_hours": ["21:30","07:30"],
      "transport_policy": ["tube","walk_if_<2km","bus"],
      "weather_fallback_policy": ["prefer_indoor_galleries","swap_days_if_rain"],
      "accessibility": { "child_friendly": False, "wheelchair_accessible": False },
      "must_include": ["Tate Modern","Saatchi Gallery","Covent Garden boutiques"],
      "must_avoid": ["overcrowded_tourist_markets"],
      "hard_windows": [],
      "timeboxing": { "buffer_minutes_between_pois": 15 },
      "hotel_location": { "description": "Kensington", "latitude": 51.499, "longitude": -0.193 },
      "max_one_way_distance_from_hotel_km": 5,
      "max_distance_from_hotel_per_day_km": 9,
      "avoid_backtracking": True,
      "local_transport": ["walk","tube","bus"],
      "passes_and_payments": { "contactless_or_oyster": True, "zone_cap_strategy": "zones_1_2_if_possible" }
    }
  },

  #5 Single Female — Intercity (London, Manchester, Edinburgh)
  {
    "id": "5",
    "category": "single_female_intercity",
    "role": "user",
    "content": "6-day UK trip for a solo 29-year-old female traveler, loves historic sites, art galleries, and local food. Visiting London, Manchester, and Edinburgh by train. Medium budget, prefers morning departures, avoids overnight travel."
  },
  {
    "id": "5",
    "category": "single_female_intercity",
    "role": "assistant",
    "content": {
      "city_sequence": ["London","Manchester","Edinburgh"],
      "city_nights": { "London": 2, "Manchester": 2, "Edinburgh": 2 },
      "intercity_prefs": {
        "preferred_mode_order": ["rail"],
        "max_leg_hours": 4.5,
        "depart_window": ["09:00","10:30"],
        "arrive_window": ["13:00","17:00"],
        "buffer_minutes_at_station": 20,
        "seat_pref": "forward_facing",
        "luggage": { "bags": 1, "heavy": False },
        "railcard": "26-30",
        "avoid_overnight_travel": True
      },
      "ticketing_rules": {
        "prefer_advance_fares": True,
        "avoid_multiple_changes": True,
        "split_tickets_ok": True
      },
      "station_preferences": {
        "London": ["Euston","Kings Cross"],
        "Manchester": ["Piccadilly"],
        "Edinburgh": ["Waverley"]
      },
      "rest_strategy": { "light_day_after_leg_hours_gt": 3 },
      "sunday_service_awareness": True
    }
  },
  
  #6 Couple — Combination (London, York, Edinburgh)
  {
    "id": "6",
    "category": "couple_combination",
    "role": "user",
    "content": "9-day UK trip for a couple (aged 34 and 36), starting in London, visiting York and Edinburgh, ending in London. Medium-high budget, prefers rail, enjoys history, fine dining, and light walking."
  },
  {
    "id": "6",
    "category": "couple_combination",
    "role": "assistant",
    "content": {
      "city_sequence": ["London","York","Edinburgh","London"],
      "city_nights": { "London": 3, "York": 2, "Edinburgh": 3, "London_return": 1 },
      "intercity_prefs": {
        "preferred_mode_order": ["rail"],
        "max_leg_hours": 4,
        "depart_window": ["09:00","10:30"],
        "arrive_window": ["13:00","17:30"],
        "buffer_minutes_at_station": 20,
        "seat_pref": "table",
        "luggage": { "bags": 2, "heavy": False },
        "railcard": "TwoTogether",
        "avoid_overnight_travel": True
      },
      "ticketing_rules": {
        "prefer_advance_fares": True,
        "avoid_multiple_changes": True,
        "split_tickets_ok": False
      },
      "station_preferences": {
        "London": ["Kings Cross"],
        "York": ["York"],
        "Edinburgh": ["Waverley"]
      },
      "rest_strategy": { "light_day_after_leg_hours_gt": 3 },
      "sunday_service_awareness": True,
      "cities": {
        "London": {
          "hotel_location": { "description": "South Kensington", "latitude": 51.494, "longitude": -0.174 },
          "pacing": "standard",
          "max_pois_per_day": 5,
          "min_dwell_minutes_per_poi": 50,
          "meal_windows": { "breakfast": ["08:00","09:30"], "lunch": ["12:30","14:00"], "dinner": ["18:30","21:00"] },
          "quiet_hours": ["22:30","07:30"],
          "weather_fallback_policy": ["prefer_indoor_museums","swap_days_if_rain"],
          "must_include": ["Tower of London","British Museum","Michelin dining experience"],
          "must_avoid": ["crowded street markets"],
          "accessibility": { "child_friendly": False, "wheelchair_accessible": False },
          "timeboxing": { "buffer_minutes_between_pois": 15 },
          "max_one_way_distance_from_hotel_km": 5,
          "max_distance_from_hotel_per_day_km": 9,
          "avoid_backtracking": True,
          "local_transport": ["tube","walk","uber"],
          "passes_and_payments": { "contactless_or_oyster": True, "zone_cap_strategy": "zones_1_2_if_possible" }
        },
        "York": {
          "hotel_location": { "description": "City Centre near Museum Gardens", "latitude": 53.961, "longitude": -1.086 },
          "pacing": "light",
          "max_pois_per_day": 4,
          "min_dwell_minutes_per_poi": 60,
          "meal_windows": { "breakfast": ["08:30","09:30"], "lunch": ["12:00","13:30"], "dinner": ["18:00","20:00"] },
          "quiet_hours": ["21:30","07:30"],
          "weather_fallback_policy": ["prefer_indoor_historic_sites","swap_days_if_rain"],
          "must_include": ["York Minster","Shambles","City Walls (short section)"],
          "must_avoid": ["steep stairs"],
          "accessibility": { "child_friendly": False, "wheelchair_accessible": True },
          "timeboxing": { "buffer_minutes_between_pois": 10 },
          "max_one_way_distance_from_hotel_km": 3,
          "max_distance_from_hotel_per_day_km": 6,
          "avoid_backtracking": True,
          "local_transport": ["walk","bus"]
        },
        "Edinburgh": {
          "hotel_location": { "description": "Old Town near Royal Mile", "latitude": 55.949, "longitude": -3.19 },
          "pacing": "standard",
          "max_pois_per_day": 5,
          "min_dwell_minutes_per_poi": 50,
          "meal_windows": { "breakfast": ["08:00","09:30"], "lunch": ["12:00","13:30"], "dinner": ["18:30","20:30"] },
          "quiet_hours": ["22:00","07:30"],
          "weather_fallback_policy": ["prefer_indoor_galleries","swap_days_if_heavy_wind"],
          "must_include": ["Edinburgh Castle","Royal Mile","Fine dining in New Town"],
          "must_avoid": ["long steep climbs on rainy days"],
          "accessibility": { "child_friendly": False, "wheelchair_accessible": False },
          "timeboxing": { "buffer_minutes_between_pois": 15 },
          "max_one_way_distance_from_hotel_km": 4,
          "max_distance_from_hotel_per_day_km": 8,
          "avoid_backtracking": True,
          "local_transport": ["walk","bus","tram"]
        }
      }
    }
  },

  #7 Family with Teens (>10) — Intracity (London)
  {
    "id": "7",
    "category": "family_teens_intracity",
    "role": "user",
    "content": "5-day London trip for a family (parents 45 & 43, teens 15 & 17). Medium budget. They like history, science museums, football stadiums, street food, and photography. Prefer Tube and walking, okay with short taxi rides at night. Avoid adult-only venues. Only intracity travel."
  },
  {
    "id": "7",
    "category": "family_teens_intracity",
    "role": "assistant",
    "content": {
      "pacing": "standard",
      "stamina_profile": "medium",
      "budget_band": "medium",
      "max_pois_per_day": 6,
      "min_dwell_minutes_per_poi": 45,
      "max_total_walk_minutes_per_day": 160,
      "max_single_walk_minutes": 25,
      "meal_windows": { "breakfast": ["08:30","09:30"], "lunch": ["12:30","14:00"], "dinner": ["18:30","20:00"] },
      "quiet_hours": ["22:30","07:30"],
      "transport_policy": ["tube","walk_if_<2km","short_taxi_ok"],
      "weather_fallback_policy": ["prefer_indoor_museums","swap_days_if_rain"],
      "accessibility": { "child_friendly": True, "wheelchair_accessible": False },
      "must_include": [
        "British Museum",
        "Science Museum",
        "Tower of London",
        "Emirates Stadium or Stamford Bridge tour",
        "Borough Market street food",
        "Greenwich & Cutty Sark"
      ],
      "must_avoid": ["adult-only venues","nightclubs"],
      "hard_windows": [],
      "timeboxing": { "buffer_minutes_between_pois": 15 },
      "hotel_location": { "description": "Waterloo area for easy Tube access", "latitude": 51.503, "longitude": -0.113 },
      "max_one_way_distance_from_hotel_km": 6,
      "max_distance_from_hotel_per_day_km": 12,
      "avoid_backtracking": True,
      "local_transport": ["tube","walk","taxi"],
      "passes_and_payments": { "contactless_or_oyster": True, "zone_cap_strategy": "zones_1_2_if_possible" }
    }
  },

  #8 Family with Elderly Parent — Combination (London, Bath, Oxford)
  {
    "id": "8",
    "category": "family_elderly_combination",
    "role": "user",
    "content": "7-day UK trip for a family of 4 (parents 42 & 40, child 12, elderly parent 72 with medium mobility). Starting in London, visiting Bath and Oxford, ending in London. Medium budget, prefers rail, avoid overnight travel. Combination intercity and intracity."
  },
  {
    "id": "8",
    "category": "family_elderly_combination",
    "role": "assistant",
    "content": {
      "city_sequence": ["London","Bath","Oxford","London"],
      "city_nights": { "London": 3, "Bath": 2, "Oxford": 2 },
      "intercity_prefs": {
        "preferred_mode_order": ["rail"],
        "max_leg_hours": 2.5,
        "depart_window": ["09:30","11:00"],
        "arrive_window": ["12:00","15:00"],
        "buffer_minutes_at_station": 25,
        "seat_pref": "forward_facing",
        "luggage": { "bags": 3, "heavy": True },
        "railcard": "Family&Friends",
        "avoid_overnight_travel": True
      },
      "ticketing_rules": { "prefer_advance_fares": True, "avoid_multiple_changes": True, "split_tickets_ok": False },
      "station_preferences": {
        "London": ["Paddington"],
        "Bath": ["Bath Spa"],
        "Oxford": ["Oxford"]
      },
      "rest_strategy": { "light_day_after_leg_hours_gt": 2 },
      "sunday_service_awareness": True,
      "cities": {
        "London": {
          "hotel_location": { "description": "South Kensington", "latitude": 51.494, "longitude": -0.174 },
          "pacing": "light",
          "max_pois_per_day": 4,
          "min_dwell_minutes_per_poi": 60,
          "meal_windows": { "breakfast": ["08:30","09:30"], "lunch": ["12:30","14:00"], "dinner": ["18:00","20:00"] },
          "quiet_hours": ["21:30","07:30"],
          "weather_fallback_policy": ["prefer_indoor_historic_sites","swap_days_if_rain"],
          "must_include": ["Tower of London","Victoria & Albert Museum"],
          "must_avoid": ["steep_stairs"],
          "accessibility": { "child_friendly": True, "wheelchair_accessible": True },
          "timeboxing": { "buffer_minutes_between_pois": 20 },
          "max_one_way_distance_from_hotel_km": 4,
          "max_distance_from_hotel_per_day_km": 8,
          "avoid_backtracking": True,
          "local_transport": ["tube","walk","taxi"],
          "passes_and_payments": { "contactless_or_oyster": True, "zone_cap_strategy": "zones_1_2_if_possible" }
        },
        "Bath": {
          "hotel_location": { "description": "City Centre near Roman Baths", "latitude": 51.381, "longitude": -2.359 },
          "pacing": "light",
          "max_pois_per_day": 3,
          "min_dwell_minutes_per_poi": 60,
          "meal_windows": { "breakfast": ["08:30","09:30"], "lunch": ["12:30","14:00"], "dinner": ["18:00","20:00"] },
          "quiet_hours": ["21:00","07:30"],
          "weather_fallback_policy": ["prefer_indoor_spa","swap_days_if_rain"],
          "must_include": ["Roman Baths","Thermae Bath Spa (family friendly hours)"],
          "must_avoid": ["steep_walks_to_Royal_Crescent"],
          "accessibility": { "child_friendly": True, "wheelchair_accessible": True },
          "timeboxing": { "buffer_minutes_between_pois": 20 },
          "max_one_way_distance_from_hotel_km": 3,
          "max_distance_from_hotel_per_day_km": 6,
          "avoid_backtracking": True,
          "local_transport": ["walk","taxi"]
        },
        "Oxford": {
          "hotel_location": { "description": "City Centre near Christ Church", "latitude": 51.752, "longitude": -1.258 },
          "pacing": "light",
          "max_pois_per_day": 3,
          "min_dwell_minutes_per_poi": 60,
          "meal_windows": { "breakfast": ["08:30","09:30"], "lunch": ["12:30","14:00"], "dinner": ["18:00","20:00"] },
          "quiet_hours": ["21:00","07:30"],
          "weather_fallback_policy": ["prefer_indoor_colleges","swap_days_if_rain"],
          "must_include": ["Christ Church College","Ashmolean Museum"],
          "must_avoid": ["long_cobbled_walks"],
          "accessibility": { "child_friendly": True, "wheelchair_accessible": True },
          "timeboxing": { "buffer_minutes_between_pois": 15 },
          "max_one_way_distance_from_hotel_km": 3,
          "max_distance_from_hotel_per_day_km": 5,
          "avoid_backtracking": True,
          "local_transport": ["walk","bus","taxi"]
        }
      }
    }
  },

  #9 Family with Baby — Intracity (Manchester)
  {
    "id": "9",
    "category": "family_baby_intracity",
    "role": "user",
    "content": "4-day Manchester trip for a family (parents 33 & 31, baby aged 9 months). Medium budget, stroller-friendly, prefers short walks and indoor attractions. Avoid noisy venues. Only intracity travel."
  },
  {
    "id": "9",
    "category": "family_baby_intracity",
    "role": "assistant",
    "content": {
      "pacing": "gentle",
      "stamina_profile": "low",
      "budget_band": "medium",
      "max_pois_per_day": 3,
      "min_dwell_minutes_per_poi": 60,
      "max_total_walk_minutes_per_day": 90,
      "max_single_walk_minutes": 15,
      "meal_windows": { "breakfast": ["08:30","09:30"], "lunch": ["12:00","13:30"], "dinner": ["17:30","19:00"] },
      "quiet_hours": ["20:00","07:00"],
      "transport_policy": ["walk_if_<1km","tram","short_taxi_ok"],
      "weather_fallback_policy": ["prefer_indoor_attractions","swap_days_if_rain"],
      "accessibility": { "child_friendly": True, "wheelchair_accessible": True, "stroller_ok": True },
      "must_include": [
        "Science and Industry Museum",
        "Manchester Art Gallery (step-free access)",
        "Heaton Park (weather permitting)"
      ],
      "must_avoid": ["loud_bars","nightclubs","overcrowded_events"],
      "hard_windows": [],
      "timeboxing": { "buffer_minutes_between_pois": 20 },
      "hotel_location": { "description": "Near St Peter's Square (tram access)", "latitude": 53.478, "longitude": -2.245 },
      "max_one_way_distance_from_hotel_km": 2,
      "max_distance_from_hotel_per_day_km": 4,
      "avoid_backtracking": True,
      "local_transport": ["walk","tram","taxi"],
      "passes_and_payments": { "use_metrolink": True }
    }
  },

  #10 Group (3-4 male) — Intercity (Manchester, Liverpool, Newcastle)
  {
    "id": "10",
    "category": "group_male_intercity",
    "role": "user",
    "content": "6-day UK trip for a group of 4 male friends in their late 20s. Visiting Manchester, Liverpool, and Newcastle by train. Medium budget, enjoys football matches, brewery tours, live music, and city nightlife. Avoids early mornings after late nights. Intercity only."
  },
  {
    "id": "10",
    "category": "group_male_intercity",
    "role": "assistant",
    "content": {
      "city_sequence": ["Manchester","Liverpool","Newcastle"],
      "city_nights": { "Manchester": 2, "Liverpool": 2, "Newcastle": 2 },
      "intercity_prefs": {
        "preferred_mode_order": ["rail"],
        "max_leg_hours": 3,
        "depart_window": ["10:00","12:00"],
        "arrive_window": ["13:00","16:00"],
        "buffer_minutes_at_station": 15,
        "seat_pref": "table",
        "luggage": { "bags": 1, "heavy": False },
        "railcard": None,
        "avoid_overnight_travel": True
      },
      "ticketing_rules": { "prefer_advance_fares": True, "avoid_multiple_changes": True, "split_tickets_ok": True },
      "station_preferences": {
        "Manchester": ["Piccadilly"],
        "Liverpool": ["Lime Street"],
        "Newcastle": ["Central"]
      },
      "rest_strategy": { "late_start_after_late_night": True },
      "sunday_service_awareness": True
    }
  },

  #11 Group (3-4 female) — Combination (London, Brighton, Bath)
  {
    "id": "11",
    "category": "group_female_combination",
    "role": "user",
    "content": "7-day UK trip for a group of 3 female friends in their late 20s. Starting in London, visiting Brighton and Bath, ending in London. Medium-high budget, prefers rail, enjoys spa days, shopping, afternoon tea, and nightlife. Combination of intercity and intracity travel."
  },
  {
    "id": "11",
    "category": "group_female_combination",
    "role": "assistant",
    "content": {
      "city_sequence": ["London","Brighton","Bath","London"],
      "city_nights": { "London": 3, "Brighton": 2, "Bath": 2 },
      "intercity_prefs": {
        "preferred_mode_order": ["rail"],
        "max_leg_hours": 2.5,
        "depart_window": ["09:30","11:00"],
        "arrive_window": ["12:00","15:00"],
        "buffer_minutes_at_station": 20,
        "seat_pref": "table",
        "luggage": { "bags": 2, "heavy": False },
        "railcard": None,
        "avoid_overnight_travel": True
      },
      "ticketing_rules": { "prefer_advance_fares": True, "avoid_multiple_changes": True, "split_tickets_ok": False },
      "station_preferences": {
        "London": ["Victoria","Paddington"],
        "Brighton": ["Brighton"],
        "Bath": ["Bath Spa"]
      },
      "rest_strategy": { "light_day_after_leg_hours_gt": 2 },
      "sunday_service_awareness": True,
      "cities": {
        "London": {
          "hotel_location": { "description": "Covent Garden", "latitude": 51.512, "longitude": -0.123 },
          "pacing": "standard",
          "max_pois_per_day": 5,
          "min_dwell_minutes_per_poi": 50,
          "meal_windows": { "breakfast": ["08:30","09:30"], "lunch": ["12:30","14:00"], "dinner": ["18:30","21:00"] },
          "quiet_hours": ["23:30","08:00"],
          "weather_fallback_policy": ["prefer_indoor_shopping","swap_days_if_rain"],
          "must_include": ["Oxford Street & Regent Street shopping","Afternoon tea at The Ritz","West End show"],
          "must_avoid": [],
          "accessibility": { "child_friendly": False, "wheelchair_accessible": False },
          "timeboxing": { "buffer_minutes_between_pois": 15 },
          "max_one_way_distance_from_hotel_km": 5,
          "max_distance_from_hotel_per_day_km": 9,
          "avoid_backtracking": True,
          "local_transport": ["tube","walk","uber"],
          "passes_and_payments": { "contactless_or_oyster": True, "zone_cap_strategy": "zones_1_2_if_possible" }
        },
        "Brighton": {
          "hotel_location": { "description": "Seafront near Brighton Pier", "latitude": 50.819, "longitude": -0.136 },
          "pacing": "light",
          "max_pois_per_day": 4,
          "min_dwell_minutes_per_poi": 60,
          "meal_windows": { "breakfast": ["09:00","10:00"], "lunch": ["13:00","14:30"], "dinner": ["19:00","21:00"] },
          "quiet_hours": ["23:00","08:00"],
          "weather_fallback_policy": ["prefer_indoor_spa","swap_days_if_rain"],
          "must_include": ["The Lanes shopping","Brighton Pier","Beachfront brunch"],
          "must_avoid": ["overcrowded_nightclubs"],
          "accessibility": { "child_friendly": False, "wheelchair_accessible": True },
          "timeboxing": { "buffer_minutes_between_pois": 15 },
          "max_one_way_distance_from_hotel_km": 3,
          "max_distance_from_hotel_per_day_km": 6,
          "avoid_backtracking": True,
          "local_transport": ["walk","bus","taxi"]
        },
        "Bath": {
          "hotel_location": { "description": "City Centre near Roman Baths", "latitude": 51.381, "longitude": -2.359 },
          "pacing": "light",
          "max_pois_per_day": 3,
          "min_dwell_minutes_per_poi": 60,
          "meal_windows": { "breakfast": ["09:00","10:00"], "lunch": ["13:00","14:30"], "dinner": ["19:00","21:00"] },
          "quiet_hours": ["22:00","08:00"],
          "weather_fallback_policy": ["prefer_indoor_spa","swap_days_if_rain"],
          "must_include": ["Thermae Bath Spa","Roman Baths","Tea room experience"],
          "must_avoid": [],
          "accessibility": { "child_friendly": False, "wheelchair_accessible": True },
          "timeboxing": { "buffer_minutes_between_pois": 20 },
          "max_one_way_distance_from_hotel_km": 3,
          "max_distance_from_hotel_per_day_km": 5,
          "avoid_backtracking": True,
          "local_transport": ["walk","taxi"]
        }
      }
    }
  },

  #12 Family group (multi-gen) — Intercity (London, York, Edinburgh)
  {
    "id": "12",
    "category": "family_multigen_intercity",
    "role": "user",
    "content": "8-day UK trip for a multigenerational family (2 grandparents ~70, 2 parents ~40, 2 kids ages 8 & 11). Visiting London, York, and Edinburgh by train. Medium budget. Interests: landmarks, museums, light walking tours, afternoon tea, kid-friendly science/animals. Intercity only."
  },
  {
    "id": "12",
    "category": "family_multigen_intercity",
    "role": "assistant",
    "content": {
      "city_sequence": ["London","York","Edinburgh"],
      "city_nights": { "London": 3, "York": 2, "Edinburgh": 3 },
      "intercity_prefs": {
        "preferred_mode_order": ["rail"],
        "max_leg_hours": 4,
        "depart_window": ["10:00","12:00"],
        "arrive_window": ["13:00","16:30"],
        "buffer_minutes_at_station": 25,
        "seat_pref": "table_near_luggage_rack",
        "luggage": { "bags": 1, "heavy": True },
        "railcard": "Family & Friends Railcard (if eligible) or Senior Railcard for grandparents",
        "mobility_support": { "book_assisted_travel": True, "step_free_access_required": True },
        "avoid_overnight_travel": True
      },
      "ticketing_rules": { "prefer_advance_fares": True, "avoid_multiple_changes": True, "split_tickets_ok": True },
      "station_preferences": {
        "London": ["Kings Cross","St Pancras"],
        "York": ["York"],
        "Edinburgh": ["Waverley"]
      },
      "rest_strategy": { "nap_window_for_kids": ["15:00","16:00"], "light_day_after_leg_hours_gt": 3 },
      "sunday_service_awareness": True
    }
  },

  #13 Elderly couple (healthy) — Intracity (York)
  {
    "id": "13",
    "category": "elderly_healthy_intracity",
    "role": "user",
    "content": "4-day city break in York for a healthy elderly couple (mid-70s). Intracity only. Slow pace, short walking distances, enjoys heritage, afternoon tea, classical music, river cruise. Medium budget."
  },
  {
    "id": "13",
    "category": "elderly_healthy_intracity",
    "role": "assistant",
    "content": {
      "cities": {
        "York": {
          "hotel_location": { "description": "Within the City Walls near Museum Gardens", "latitude": 53.961, "longitude": -1.088 },
          "pacing": "light",
          "max_pois_per_day": 3,
          "min_dwell_minutes_per_poi": 60,
          "meal_windows": { "breakfast": ["08:30","09:30"], "lunch": ["12:30","14:00"], "tea": ["15:30","17:00"], "dinner": ["18:30","20:00"] },
          "quiet_hours": ["21:30","08:00"],
          "weather_fallback_policy": ["prefer_indoor_museums","swap_days_if_rain"],
          "must_include": ["York Minster","City Walls short section (flat)","River Ouse cruise","Bettys afternoon tea","National Railway Museum (step-free routes)"],
          "must_avoid": ["steep_staircase_attractions","late_night_events"],
          "accessibility": { "child_friendly": False, "wheelchair_accessible": True, "step_free_routes_priority": True, "bench_breaks_every_mins": 20 },
          "timeboxing": { "buffer_minutes_between_pois": 20 },
          "max_one_way_distance_from_hotel_km": 2.0,
          "max_distance_from_hotel_per_day_km": 4.0,
          "avoid_backtracking": True,
          "local_transport": ["walk","taxi"],
          "passes_and_payments": { "contactless_card": True }
        }
      }
    }
  },

  #14 Elderly couple (needs assistance) — Combination (London, Bath)
  {
    "id": "14",
    "category": "elderly_assistance_combination",
    "role": "user",
    "content": "6-day UK trip for an elderly couple; one partner uses a foldable wheelchair and needs assistance. Starting London, then Bath, return to London. Medium budget. Interests: accessible heritage, river scenery, spa/thermal experiences, gentle dining. Combination of intercity rail and intracity travel."
  },
  {
    "id": "14",
    "category": "elderly_assistance_combination",
    "role": "assistant",
    "content": {
      "city_sequence": ["London","Bath","London"],
      "city_nights": { "London": 3, "Bath": 2, "London(return)": 1 },
      "intercity_prefs": {
        "preferred_mode_order": ["rail"],
        "max_leg_hours": 2,
        "depart_window": ["10:00","11:30"],
        "arrive_window": ["12:00","14:00"],
        "buffer_minutes_at_station": 30,
        "seat_pref": "near_accessible_toilet",
        "luggage": { "bags": 1, "heavy": True },
        "railcard": "Senior Railcard",
        "mobility_support": {
          "book_assisted_travel": True,
          "ramp_and_wheelchair_space": True,
          "station_buddy_required": True
        },
        "avoid_overnight_travel": True
      },
      "ticketing_rules": { "prefer_advance_fares": True, "avoid_multiple_changes": True, "split_tickets_ok": False },
      "station_preferences": {
        "London": ["Paddington"],
        "Bath": ["Bath Spa"]
      },
      "rest_strategy": { "max_total_outing_hours_per_day": 6, "midday_rest": True },
      "sunday_service_awareness": True,
      "cities": {
        "London": {
          "hotel_location": { "description": "South Bank (flat riverside paths)", "latitude": 51.506, "longitude": -0.119 },
          "pacing": "light",
          "max_pois_per_day": 2,
          "min_dwell_minutes_per_poi": 75,
          "meal_windows": { "breakfast": ["08:30","09:30"], "lunch": ["12:30","14:00"], "dinner": ["18:00","20:00"] },
          "quiet_hours": ["21:30","08:00"],
          "weather_fallback_policy": ["prefer_indoor_galleries","shift_cruise_to_clear_day"],
          "must_include": ["Thames river cruise (accessible)","Tate Modern step-free route","Accessible West End matinee (step-free seating)"],
          "must_avoid": ["crowded_peak_tube_interchanges","venues_without_lifts"],
          "accessibility": {
            "wheelchair_accessible": True,
            "step_free_routes_priority": True,
            "book_theatre_access_seating": True,
            "accessible_taxis_priority": True
          },
          "timeboxing": { "buffer_minutes_between_pois": 25 },
          "max_one_way_distance_from_hotel_km": 3,
          "max_distance_from_hotel_per_day_km": 5,
          "avoid_backtracking": True,
          "local_transport": ["accessible_taxi","step_free_tube_lines","bus (low-floor)"],
          "passes_and_payments": { "contactless_or_oyster": True, "off_peak_only": True }
        },
        "Bath": {
          "hotel_location": { "description": "Flat approach near Bath Abbey", "latitude": 51.381, "longitude": -2.359 },
          "pacing": "light",
          "max_pois_per_day": 2,
          "min_dwell_minutes_per_poi": 75,
          "meal_windows": { "breakfast": ["09:00","10:00"], "lunch": ["12:30","14:00"], "dinner": ["18:00","20:00"] },
          "quiet_hours": ["21:00","08:00"],
          "weather_fallback_policy": ["prefer_indoor_spa","swap_days_if_rain"],
          "must_include": ["Thermae Bath Spa (accessible sessions)","Roman Baths accessible route","Pulteney Bridge viewpoint"],
          "must_avoid": ["steep_cobbled_routes"],
          "accessibility": {
            "wheelchair_accessible": True,
            "verify_lifts_and_ramps": True,
            "reserve_accessible_changing_rooms": True
          },
          "timeboxing": { "buffer_minutes_between_pois": 25 },
          "max_one_way_distance_from_hotel_km": 2,
          "max_distance_from_hotel_per_day_km": 4,
          "avoid_backtracking": True,
          "local_transport": ["accessible_taxi","walk_short_flat_segments"]
        }
      }
    }
  },

  #15 Mixed group (friends + family) — Combination (London, Cardiff)
  {
    "id": "15",
    "category": "mixed_group_combination",
    "role": "user",
    "content": "7-day UK trip for a mixed group: 2 friends (early 30s), 1 couple with a teen (14), and 1 grandparent (late 60s). Start London, visit Cardiff, end London. Medium budget. Interests: landmarks, street food, football stadium tour, light hikes, castles, family-friendly nightlife. Combination of intercity rail and intracity."
  },
  {
    "id": "15",
    "category": "mixed_group_combination",
    "role": "assistant",
    "content": {
      "city_sequence": ["London","Cardiff","London"],
      "city_nights": { "London": 4, "Cardiff": 2, "London(return)": 1 },
      "intercity_prefs": {
        "preferred_mode_order": ["rail"],
        "max_leg_hours": 2.5,
        "depart_window": ["09:30","11:30"],
        "arrive_window": ["12:00","14:30"],
        "buffer_minutes_at_station": 20,
        "seat_pref": "table_mixed",
        "luggage": { "bags": 1, "heavy": False },
        "railcard": "Two Together (friends/couple) + Senior Railcard (grandparent) if applicable",
        "avoid_overnight_travel": True
      },
      "ticketing_rules": { "prefer_advance_fares": True, "avoid_multiple_changes": True, "split_tickets_ok": True },
      "station_preferences": {
        "London": ["Paddington"],
        "Cardiff": ["Cardiff Central"]
      },
      "rest_strategy": { "late_start_after_late_night": True, "light_day_after_leg_hours_gt": 2 },
      "sunday_service_awareness": True,
      "cities": {
        "London": {
          "hotel_location": { "description": "South Kensington / Victoria (good rail & buses)", "latitude": 51.494, "longitude": -0.165 },
          "pacing": "standard",
          "max_pois_per_day": 5,
          "min_dwell_minutes_per_poi": 45,
          "meal_windows": { "breakfast": ["08:30","09:30"], "lunch": ["12:30","14:00"], "dinner": ["18:30","21:00"] },
          "quiet_hours": ["23:30","08:00"],
          "weather_fallback_policy": ["prefer_indoor_museums","swap_days_if_rain"],
          "must_include": ["Buckingham Palace area","Borough Market street food","Emirates Stadium or Wembley tour","Westminster & South Bank walk"],
          "must_avoid": ["overpacked_schedules"],
          "accessibility": { "child_friendly": True, "wheelchair_accessible": True },
          "timeboxing": { "buffer_minutes_between_pois": 15 },
          "max_one_way_distance_from_hotel_km": 6,
          "max_distance_from_hotel_per_day_km": 10,
          "avoid_backtracking": True,
          "local_transport": ["tube","bus","uber"],
          "passes_and_payments": { "contactless_or_oyster": True, "zone_cap_strategy": "zones_1_2_if_possible" }
        },
        "Cardiff": {
          "hotel_location": { "description": "City Centre near Cardiff Castle", "latitude": 51.481, "longitude": -3.181 },
          "pacing": "standard",
          "max_pois_per_day": 4,
          "min_dwell_minutes_per_poi": 50,
          "meal_windows": { "breakfast": ["09:00","10:00"], "lunch": ["12:30","14:00"], "dinner": ["18:30","21:00"] },
          "quiet_hours": ["23:00","08:00"],
          "weather_fallback_policy": ["prefer_indoor_attractions","swap_days_if_rain"],
          "must_include": ["Cardiff Castle","Principality Stadium tour","Cardiff Bay & barrage walk (light)"],
          "must_avoid": ["long_out_of_town_drives"],
          "accessibility": { "child_friendly": True, "wheelchair_accessible": True },
          "timeboxing": { "buffer_minutes_between_pois": 15 },
          "max_one_way_distance_from_hotel_km": 4,
          "max_distance_from_hotel_per_day_km": 8,
          "avoid_backtracking": True,
          "local_transport": ["walk","bus","taxi"]
        }
      }
    }
  }
]
