"""Moreza: daily satellite sea surface temperature for every beach point.

Runs on GitHub Actions (see .github/workflows/sst.yml). It
  1. takes the beach list from the site (points.php), or points.json in this repo if the site is unreachable,
  2. reads three satellite products for the Montenegro coast:
       L4  Copernicus Marine Mediterranean ~1 km analysis (gap-free, smoothed over several days),
       L3S Copernicus Marine Mediterranean ~1 km nightly observations (last night only, gaps under clouds),
       MUR NASA JPL global ~1 km analysis (independent method, from the open NOAA ERDDAP server),
  3. for each beach takes the nearest sea pixel with data in each product and combines them
     (see combine()): last night's real observation counts most, so the value follows fast cooling
     or warming instead of lagging behind the smoothed analysis,
  4. works out for each beach the direction towards open water (from the same land/sea mask),
     used by the site for the "offshore wind" note,
  5. writes sst.json, which the site reads and blends with the Open-Meteo model.

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
# Mediterranean L3S (supercollated nightly observations). The product has an HR (1/16°) and an UHR (1/100°)
# dataset; we try these ids and keep the finest grid that opens.
L3S_DATASETS = ["SST_MED_SST_L3S_NRT_OBSERVATIONS_010_012_b", "SST_MED_SST_L3S_NRT_OBSERVATIONS_010_012_a",
                "SST_MED_SST_L3S_NRT_OBSERVATIONS_010_012_b_V2", "SST_MED_SST_L3S_NRT_OBSERVATIONS_010_012_a_V2"]
L3S_MAX_KM = 2.0      # raw observations: only a pixel really next to the beach
# NASA JPL MUR 0.01° daily analysis on public ERDDAP servers (no account needed); first one that answers wins
MUR_SERVERS = ["https://coastwatch.pfeg.noaa.gov/erddap", "https://erddap.emodnet.eu/erddap"]
# how much each product counts in the combined satellite value (renormalised over what is available)
W_L3S, W_L4, W_MUR = 0.5, 0.3, 0.2
OUTLIER = 3.0         # °C: a product further than this from the L4 value at a beach is ignored there (cloud edge etc.)
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


def seaward(points, lat, lon, grid_c, radius_km=4.0, min_strength=0.3):
    """Direction (degrees, 0 = north, 90 = east) from each beach towards open water.

    Sum of unit vectors from the beach to every sea pixel within radius_km, nearer pixels weighing more.
    On a straight coast this points straight out to sea. Where water lies on several sides (narrow straits,
    capes, inner bays) the vectors cancel out; if what is left is weaker than min_strength we give no
    direction, and the site shows no offshore note for that beach.
    """
    la, lo = np.meshgrid(lat, lon, indexing="ij")
    ok = np.isfinite(grid_c)
    vla, vlo = la[ok], lo[ok]
    out = {}
    for p in points:
        dy = (vla - p["lat"]) * 111.2
        dx = (vlo - p["lng"]) * 111.2 * math.cos(math.radians(p["lat"]))
        d = np.hypot(dx, dy)
        m = (d > 0.05) & (d <= radius_km)
        if m.sum() < 3:
            continue
        w = 1.0 / np.maximum(d[m], 0.5)
        sx = float((dx[m] / d[m] * w).sum())
        sy = float((dy[m] / d[m] * w).sum())
        if math.hypot(sx, sy) / float(w.sum()) < min_strength:
            continue
        out[p["slug"]] = int(round(math.degrees(math.atan2(sx, sy)))) % 360
    return out


def _newest(ds, var, quality=None):
    """(date, lat, lon, grid °C) for the newest time step of ds[var] that has data."""
    latn = next(c for c in ds[var].dims if c.lower().startswith("lat"))
    lonn = next(c for c in ds[var].dims if c.lower().startswith("lon"))
    sst = ds[var].transpose("time", latn, lonn)
    q = ds[quality].transpose("time", latn, lonn) if quality else None
    lat, lon = ds[latn].values, ds[lonn].values
    for i in range(sst.sizes["time"] - 1, -1, -1):
        grid = np.asarray(sst.isel(time=i).values, dtype="float64")
        if q is not None:
            grid[np.asarray(q.isel(time=i).values) < 4] = np.nan   # keep good / best quality only
        if np.isfinite(grid).any():
            if np.nanmedian(grid) > 100:      # Kelvin → °C
                grid = grid - 273.15
            return str(sst["time"].values[i])[:10], lat, lon, grid
    return None


def _window():
    end = dt.datetime.now(dt.timezone.utc)
    start = end - dt.timedelta(days=DAYS_BACK)
    return start.strftime("%Y-%m-%dT00:00:00"), end.strftime("%Y-%m-%dT23:59:59")


def latest_field():
    """L4: (date 'YYYY-MM-DD', lat, lon, grid °C) for the newest day that has data."""
    import copernicusmarine  # installed in the workflow
    start, end = _window()
    ds = copernicusmarine.open_dataset(dataset_id=DATASET, variables=["analysed_sst"],
                                       start_datetime=start, end_datetime=end, **BBOX)
    r = _newest(ds, "analysed_sst")
    if r is None:
        raise RuntimeError("no SST data in the last %d days" % DAYS_BACK)
    return r


def latest_l3s():
    """L3S nightly observations: (date, lat, lon, grid °C) or None. Never fails the run."""
    import copernicusmarine
    start, end = _window()
    best = None
    for dsid in L3S_DATASETS:
        try:
            ds = copernicusmarine.open_dataset(dataset_id=dsid, start_datetime=start, end_datetime=end, **BBOX)
        except Exception as e:
            print(f"L3S {dsid}: not available ({str(e)[:120]})")
            continue
        var = next((v for v in ("adjusted_sea_surface_temperature", "sea_surface_temperature", "analysed_sst") if v in ds), None)
        if var is None:
            print(f"L3S {dsid}: no SST variable in {list(ds.data_vars)}")
            continue
        quality = "quality_level" if "quality_level" in ds else None
        try:
            r = _newest(ds, var, quality)
        except Exception as e:
            print(f"L3S {dsid}: read failed ({str(e)[:120]})")
            continue
        if r is None:
            print(f"L3S {dsid}: no data in the last {DAYS_BACK} days")
            continue
        step = float(abs(np.diff(r[1][:2])[0])) if len(r[1]) > 1 else 1.0
        print(f"L3S {dsid}: {r[0]}, var {var}, grid {step:.4f}°, {int(np.isfinite(r[3]).sum())} sea pixels")
        if best is None or step < best[0]:
            best = (step, dsid, r)
    return best[2] if best else None


def latest_mur():
    """NASA JPL MUR 0.01° from ERDDAP: (date, lat, lon, grid °C) or None. Never fails the run."""
    q = (f"/griddap/jplMURSST41.json?analysed_sst[(last)]"
         f"[({BBOX['minimum_latitude']}):({BBOX['maximum_latitude']})]"
         f"[({BBOX['minimum_longitude']}):({BBOX['maximum_longitude']})]")
    for server in MUR_SERVERS:
        try:
            req = urllib.request.Request(server + q, headers={"User-Agent": "moreza-sst/1.0"})
            with urllib.request.urlopen(req, timeout=120) as r:
                tab = json.load(r)["table"]
            cols = tab["columnNames"]
            it, ila, ilo, iv = (cols.index(c) for c in ("time", "latitude", "longitude", "analysed_sst"))
            rows = tab["rows"]
            lat = np.array(sorted({row[ila] for row in rows}))
            lon = np.array(sorted({row[ilo] for row in rows}))
            grid = np.full((len(lat), len(lon)), np.nan)
            li = {v: i for i, v in enumerate(lat)}
            lj = {v: j for j, v in enumerate(lon)}
            for row in rows:
                if row[iv] is not None:
                    grid[li[row[ila]], lj[row[ilo]]] = row[iv]
            if not np.isfinite(grid).any():
                raise ValueError("empty field")
            if np.nanmedian(grid) > 100:
                grid = grid - 273.15
            day = str(rows[0][it])[:10]
            print(f"MUR {server}: {day}, {int(np.isfinite(grid).sum())} sea pixels")
            return day, lat, lon, grid
        except Exception as e:
            print(f"MUR {server}: failed ({str(e)[:120]})")
    return None


def combine(l4, l3, mur, l4_day, l3_day, mur_day):
    """Combined satellite value per beach (°C, 0.1).

    L3S counts only if it is from the same night as L4 or newer (it is the freshest real observation);
    MUR only if it is at most 2 days older than L4. A product more than OUTLIER °C away from L4 at a beach
    is ignored there (cloud edges and land-contaminated pixels show up like that). Where L4 has no value
    the other products are used as they are.
    """
    def days(a, b):
        try:
            return (dt.date.fromisoformat(a) - dt.date.fromisoformat(b)).days
        except Exception:
            return -99
    use_l3 = l3 is not None and days(l3_day, l4_day) >= 0
    use_mur = mur is not None and days(mur_day, l4_day) >= -2
    out = {}
    for slug in set(l4) | (set(l3 or {}) if use_l3 else set()) | (set(mur or {}) if use_mur else set()):
        base = l4.get(slug)
        parts = []
        if base is not None:
            parts.append((W_L4, base))
        for ok, src, w in ((use_l3, l3, W_L3S), (use_mur, mur, W_MUR)):
            v = src.get(slug) if ok else None
            if v is not None and (base is None or abs(v - base) <= OUTLIER):
                parts.append((w, v))
        if parts:
            out[slug] = round(sum(w * v for w, v in parts) / sum(w for w, _ in parts) + 1e-9, 1)
    return out


def main():
    points = load_points()
    day, lat, lon, grid = latest_field()
    l4 = sample(points, lat, lon, grid)
    dirs = seaward(points, lat, lon, grid)

    l3, l3_day = None, ""
    try:
        r = latest_l3s()
        if r:
            l3_day = r[0]
            l3 = sample(points, r[1], r[2], r[3], max_km=L3S_MAX_KM)
    except Exception as e:
        print(f"L3S skipped: {e}")
    mur, mur_day = None, ""
    try:
        r = latest_mur()
        if r:
            mur_day = r[0]
            mur = sample(points, r[1], r[2], r[3])
    except Exception as e:
        print(f"MUR skipped: {e}")

    temps = combine(l4, l3, mur, day, l3_day, mur_day)
    if not temps:
        sys.exit("no beach got a value — not overwriting sst.json")
    lo_t, hi_t = min(temps.values()), max(temps.values())
    if not (0 < lo_t and hi_t < 40):
        sys.exit(f"suspicious values {lo_t}..{hi_t} °C — not overwriting sst.json")
    data = {
        "source": "Copernicus Marine L4 + L3S, NASA JPL MUR",
        "date": day,
        "updated": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
        "count": len(temps),
        "t": temps,              # combined satellite value: what the site uses
        "dir": dirs,
        "l4": l4,                # the three products separately, for checking
        "l3": l3 or {}, "l3date": l3_day,
        "mur": mur or {}, "murdate": mur_day,
    }
    with open("sst.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
    print(f"sst.json: {len(temps)}/{len(points)} beaches, L4 {day} ({len(l4)}), L3S {l3_day or '-'} ({len(l3 or {})}), "
          f"MUR {mur_day or '-'} ({len(mur or {})}), {lo_t}..{hi_t} °C; sea direction for {len(dirs)}")
    if l3:
        d = [l3[s] - l4[s] for s in l3 if s in l4]
        if d:
            print(f"  L3S - L4: mean {sum(d)/len(d):+.2f} °C over {len(d)} beaches")
    if mur:
        d = [mur[s] - l4[s] for s in mur if s in l4]
        if d:
            print(f"  MUR - L4: mean {sum(d)/len(d):+.2f} °C over {len(d)} beaches")


if __name__ == "__main__":
    main()
