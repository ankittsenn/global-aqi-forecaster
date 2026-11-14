"""The cities the models are trained on, with a few documented seasonal drivers."""
import re
import unicodedata

import pandas as pd

# name, country, region, latitude, longitude
_CITIES = [
    ("Delhi", "India", "South Asia", 28.6139, 77.2090),
    ("Lahore", "Pakistan", "South Asia", 31.5497, 74.3436),
    ("Dhaka", "Bangladesh", "South Asia", 23.8103, 90.4125),
    ("Kolkata", "India", "South Asia", 22.5726, 88.3639),
    ("Mumbai", "India", "South Asia", 19.0760, 72.8777),
    ("Kathmandu", "Nepal", "South Asia", 27.7172, 85.3240),
    ("Beijing", "China", "East Asia", 39.9042, 116.4074),
    ("Shanghai", "China", "East Asia", 31.2304, 121.4737),
    ("Seoul", "South Korea", "East Asia", 37.5665, 126.9780),
    ("Tokyo", "Japan", "East Asia", 35.6762, 139.6503),
    ("Ulaanbaatar", "Mongolia", "East Asia", 47.8864, 106.9057),
    ("Bangkok", "Thailand", "Southeast Asia", 13.7563, 100.5018),
    ("Chiang Mai", "Thailand", "Southeast Asia", 18.7883, 98.9853),
    ("Hanoi", "Vietnam", "Southeast Asia", 21.0285, 105.8542),
    ("Jakarta", "Indonesia", "Southeast Asia", -6.2088, 106.8456),
    ("Singapore", "Singapore", "Southeast Asia", 1.3521, 103.8198),
    ("Kuala Lumpur", "Malaysia", "Southeast Asia", 3.1390, 101.6869),
    ("Tehran", "Iran", "Middle East and Central Asia", 35.6892, 51.3890),
    ("Riyadh", "Saudi Arabia", "Middle East and Central Asia", 24.7136, 46.6753),
    ("Dubai", "United Arab Emirates", "Middle East and Central Asia", 25.2048, 55.2708),
    ("Almaty", "Kazakhstan", "Middle East and Central Asia", 43.2389, 76.8897),
    ("Cairo", "Egypt", "Africa", 30.0444, 31.2357),
    ("Lagos", "Nigeria", "Africa", 6.5244, 3.3792),
    ("Kano", "Nigeria", "Africa", 12.0022, 8.5919),
    ("Accra", "Ghana", "Africa", 5.6037, -0.1870),
    ("Nairobi", "Kenya", "Africa", -1.2921, 36.8219),
    ("Johannesburg", "South Africa", "Africa", -26.2041, 28.0473),
    ("London", "United Kingdom", "Europe", 51.5072, -0.1276),
    ("Paris", "France", "Europe", 48.8566, 2.3522),
    ("Berlin", "Germany", "Europe", 52.5200, 13.4050),
    ("Kraków", "Poland", "Europe", 50.0647, 19.9450),
    ("Istanbul", "Turkey", "Europe", 41.0082, 28.9784),
    ("Sarajevo", "Bosnia and Herzegovina", "Europe", 43.8563, 18.4131),
    ("Los Angeles", "United States", "North America", 34.0522, -118.2437),
    ("New York", "United States", "North America", 40.7128, -74.0060),
    ("Mexico City", "Mexico", "North America", 19.4326, -99.1332),
    ("Vancouver", "Canada", "North America", 49.2827, -123.1207),
    ("São Paulo", "Brazil", "South America", -23.5505, -46.6333),
    ("Santiago", "Chile", "South America", -33.4489, -70.6693),
    ("Lima", "Peru", "South America", -12.0464, -77.0428),
    ("Bogotá", "Colombia", "South America", 4.7110, -74.0721),
    ("Sydney", "Australia", "Oceania", -33.8688, 151.2093),
]

# Seasonal pollution drivers that are widely documented for these cities. They are
# assumptions from domain knowledge, not measurements: the model only gets a 0/1
# "in season" flag and the backtest shows whether it helps.
_SEASONS = {
    "Delhi": ([10, 11], "crop-residue burning season (Oct to Nov)"),
    "Lahore": ([10, 11], "crop-residue burning season (Oct to Nov)"),
    "Dhaka": ([11, 12, 1, 2], "dry-season brick-kiln and inversion period (Nov to Feb)"),
    "Chiang Mai": ([2, 3, 4], "open-burning season (Feb to Apr)"),
    "Singapore": ([8, 9, 10], "regional fire-haze season (Aug to Oct)"),
    "Kuala Lumpur": ([8, 9, 10], "regional fire-haze season (Aug to Oct)"),
    "Vancouver": ([7, 8, 9], "wildfire-smoke season (Jul to Sep)"),
    "Los Angeles": ([8, 9, 10], "wildfire season (Aug to Oct)"),
    "Ulaanbaatar": ([10, 11, 12, 1, 2, 3], "winter coal-heating season (Oct to Mar)"),
    "Beijing": ([11, 12, 1, 2, 3], "winter heating season (Nov to Mar)"),
    "Kraków": ([10, 11, 12, 1, 2, 3], "winter heating season (Oct to Mar)"),
    "Sarajevo": ([11, 12, 1, 2], "winter heating season (Nov to Feb)"),
    "Almaty": ([11, 12, 1, 2], "winter heating and inversion season (Nov to Feb)"),
    "Santiago": ([5, 6, 7, 8], "winter wood-burning and inversion season (May to Aug)"),
    "Kano": ([12, 1, 2], "Harmattan dust season (Dec to Feb)"),
    "Lagos": ([12, 1, 2], "Harmattan dust season (Dec to Feb)"),
    "Accra": ([12, 1, 2], "Harmattan dust season (Dec to Feb)"),
}


def slugify(name: str) -> str:
    """'São Paulo' -> 'sao-paulo' (used for file names)."""
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", ascii_name.lower()).strip("-")


def catalogue() -> pd.DataFrame:
    rows = []
    for name, country, region, lat, lon in _CITIES:
        months, label = _SEASONS.get(name, ([], ""))
        rows.append({"city": name, "country": country, "region": region, "lat": lat,
                     "lon": lon, "slug": slugify(name), "season_months": list(months),
                     "season_label": label})
    return pd.DataFrame(rows)


def season_months() -> dict:
    """city -> list of months flagged as the assumed high-pollution season."""
    cat = catalogue()
    return dict(zip(cat["city"], cat["season_months"]))
