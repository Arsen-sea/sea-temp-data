"""Sea Temp Montenegro: daily satellite sea surface temperature for every beach point.

Runs on GitHub Actions (see .github/workflows/sst.yml). It
  1. takes the beach list from the site (points.php), or points.json in this repo if the site is unreachable,
  2. reads the Copernicus Marine Mediterranean ~1 km SST analysis for the Montenegro coast,
  3. for each beach takes the nearest sea pixel with data (up to MAX_KM away),
  4. writes sst.json, which the site reads and blends with the Open-Meteo model.

Copernicus login comes from the repository secrets
COPERNICUSMARINE_SERVICE_USERNAME / COPERNICUSMARINE_SERVICE_PASSWORD.
"""
import datetime as dt
import json
import math
import sys
import urllib.request

import numpy as np

DATASET = "SST_MED_SST_L4_NRT_OBSERVATIONS_010_004_c_V2"   # Mediterranean L4 SST, 1/100°, daily (night)
POINTS_URL = "https://moreza.me/points.php"
BBOX = dict(minimum_longitude=18.35, maximum_longitude=19.50, minimum_latitude=41.75, maximum_latitude=42.60)
MAX_KM = 3.0          # beach further than this from any sea pixel with data gets no satellite value
DAYS_BACK = 5         # look this many days back for the latest field with data


def load_points():
    """[{slug, lat, lng}, …] from the site; falls back to (and refreshes) points.json."""
    try:
        req = urllib.request.Request(POINTS_URL, headers={"User-Agent": "sea-temp-data/1.0"})
        with urllib.request.urlopen(req, timeout=30) as r:
            pts = json.load(r)
        if isinstance(pts, list) and pts and all({"slug", "lat", "lng"} <= set(p) for p in pts):
            with open("points.json", "w", encoding="utf-8") as f:
                json.dump(pts, f, ensure_ascii=False, separators=(",", ":"))
            print(f"points: {len(pts)} from the site")
            return pts
    except Exception as e:  # site down, not deployed yet, etc.
        print(f"points.php unavailable ({e}); using points.json")
    with open("points.json", encoding="utf-8") as f:
        pts = json.load(f)
    print(f"points: {len(pts)} from points.json")
    return pts


def sample(points, lat, lon, grid_c, max_km=MAX_KM):
    """Nearest valid pixel for each point. lat (n), lon (m), grid_c (n×m, °C, NaN = land/no data)."""
    la, lo = np.meshgrid(lat, lon, indexing="ij")
    ok = np.isfinite(grid_c)
    vla, vlo, vt = la[ok], lo[ok], grid_c[ok]
    out = {}
    if vt.size == 0:
        return out
    for p in points:
        dy = (vla - p["lat"]) * 111.2
        dx = (vlo - p["lng"]) * 111.2 * math.cos(math.radians(p["lat"]))
        d2 = dx * dx + dy * dy
        j = int(d2.argmin())
        if d2[j] <= max_km * max_km:
            out[p["slug"]] = round(float(vt[j]), 1)
    return out


def latest_field():
    """(date 'YYYY-MM-DD', lat, lon, grid °C) for the newest day that has data."""
    import copernicusmarine  # installed in the workflow

    end = dt.datetime.now(dt.timezone.utc)
    start = end - dt.timedelta(days=DAYS_BACK)
    ds = copernicusmarine.open_dataset(
        dataset_id=DATASET,
        variables=["analysed_sst"],
        start_datetime=start.strftime("%Y-%m-%dT00:00:00"),
        end_datetime=end.strftime("%Y-%m-%dT23:59:59"),
        **BBOX,
    )
    latn = next(c for c in ds["analysed_sst"].dims if c.lower().startswith("lat"))
    lonn = next(c for c in ds["analysed_sst"].dims if c.lower().startswith("lon"))
    sst = ds["analysed_sst"].transpose("time", latn, lonn)
    lat, lon = ds[latn].values, ds[lonn].values
    for i in range(sst.sizes["time"] - 1, -1, -1):
        grid = np.asarray(sst.isel(time=i).values, dtype="float64")
        if np.isfinite(grid).any():
            if np.nanmedian(grid) > 100:      # Kelvin → °C
                grid = grid - 273.15
            day = str(sst["time"].values[i])[:10]
            return day, lat, lon, grid
    raise RuntimeError("no SST data in the last %d days" % DAYS_BACK)


def main():
    points = load_points()
    day, lat, lon, grid = latest_field()
    temps = sample(points, lat, lon, grid)
    if not temps:
        sys.exit("no beach got a value — not overwriting sst.json")
    lo_t, hi_t = min(temps.values()), max(temps.values())
    if not (0 < lo_t and hi_t < 40):
        sys.exit(f"suspicious values {lo_t}..{hi_t} °C — not overwriting sst.json")
    data = {
        "source": "Copernicus Marine " + DATASET,
        "date": day,
        "updated": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
        "count": len(temps),
        "t": temps,
    }
    with open("sst.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
    print(f"sst.json: {len(temps)}/{len(points)} beaches, {day}, {lo_t}..{hi_t} °C")


if __name__ == "__main__":
    main()
