"""Synthetic heat + movement data, calibrated to the real data summary:
 - 10 cows (db-cow-walking), DIM window 11-110 (estrus paper), 10 Hz IMU clips
 - 441 labelled clips: Grazing 105, Walking 217, Resting 10, 9 rare behaviours (109 total)
All values here are SYNTHETIC. Estrus effect sizes are assumptions from the literature direction
(activity up, rumination down, mounting up, head temp up), not measured from the user's data.
"""
import numpy as np, pandas as pd

rng = np.random.default_rng(42)
N_COWS = 10
DIM_START, DIM_END = 11, 110
HOURS = (DIM_END - DIM_START + 1) * 24
BREEDS = ["Holstein"] * N_COWS

rows = []
for c in range(N_COWS):
    cow = f"cow_{c+1:02d}"
    parity = int(rng.integers(1, 5))
    base_act = rng.normal(1.0, 0.12)            # cow-specific baseline activity
    base_rum = rng.normal(27, 2.5)              # rumination min/h
    cycle = rng.normal(21, 1.5)                 # estrus cycle (days)
    heat_dur = float(np.clip(rng.normal(14, 4), 6, 24))  # cow-specific duration (h) -> heritable trait
    act_gain = rng.normal(1.9, 0.3)             # activity multiplier in heat
    first = rng.uniform(0, cycle)               # offset of first heat from DIM 11
    heat_starts = [first * 24 + k * cycle * 24 + rng.normal(0, 12) for k in range(6)]
    heat_starts = [h for h in heat_starts if 0 <= h < HOURS]

    heat = np.zeros(HOURS, dtype=int)
    for h in heat_starts:
        heat[int(h): int(h + heat_dur)] = 1
    # days (hours) to next heat start
    nxt = np.full(HOURS, np.nan)
    for t in range(HOURS):
        fut = [h - t for h in heat_starts if h - t >= 0]
        if fut: nxt[t] = min(fut) / 24

    for t in range(HOURS):
        hod = t % 24
        dim = DIM_START + t // 24
        diurnal = 0.55 + 0.45 * np.sin((hod - 6) / 24 * 2 * np.pi) ** 2
        h_ = heat[t]
        thi = 68 + 6 * np.sin((hod - 9) / 24 * 2 * np.pi) + rng.normal(0, 1)
        act = base_act * diurnal * (act_gain if h_ else 1) * rng.lognormal(0, 0.15)
        rum = max(0, base_rum * (1 - 0.8 * diurnal * 0.5) * (0.72 if h_ else 1) + rng.normal(0, 2))
        walk = np.clip(0.18 * act / base_act + rng.normal(0, 0.03), 0, 1)
        graze = np.clip(0.30 * diurnal + rng.normal(0, 0.04) - (0.08 if h_ else 0), 0, 1)
        lying = np.clip(1 - walk - graze - 0.12 + rng.normal(0, 0.04) - (0.15 if h_ else 0), 0, 1)
        stand = np.clip(1 - walk - graze - lying, 0, 1)
        rows.append(dict(
            cow_id=cow, breed=BREEDS[c], parity=parity, dim=dim, hour=hod, t_index=t,
            activity_index=round(act, 3), rumination_min=round(rum, 1),
            walking_frac=round(walk, 3), grazing_frac=round(graze, 3),
            lying_frac=round(lying, 3), standing_frac=round(stand, 3),
            distance_m=round(max(0, walk * 60 * rng.lognormal(0, 0.2)), 1),
            mounting_events=int(rng.poisson(1.6 if h_ else 0.03)),
            neighbour_proximity=round(float(np.clip(rng.normal(0.5 + 0.25 * h_, 0.1), 0, 1)), 3),
            head_temp_c=round(float(36.4 + 0.5 * h_ + 0.2 * (thi - 68) / 6 + rng.normal(0, 0.15)), 2),
            thi=round(float(thi), 1),
            heat=int(h_), days_to_next_heat=None if np.isnan(nxt[t]) else round(float(nxt[t]), 2),
        ))

hourly = pd.DataFrame(rows)
hourly.to_csv("synthetic_hourly_10cows.csv", index=False)

# daily aggregation with per-cow z-scores against own rolling 7-day baseline
g = hourly.groupby(["cow_id", "dim"])
daily = g.agg(activity_index=("activity_index", "mean"), rumination_min=("rumination_min", "sum"),
              walking_frac=("walking_frac", "mean"), lying_frac=("lying_frac", "mean"),
              distance_m=("distance_m", "sum"), mounting_events=("mounting_events", "sum"),
              head_temp_c=("head_temp_c", "mean"), thi=("thi", "mean"), heat=("heat", "max")).reset_index()
for col in ["activity_index", "rumination_min", "distance_m", "head_temp_c"]:
    b = daily.groupby("cow_id")[col].transform(lambda s: s.shift(1).rolling(7, min_periods=3).median())
    sd = daily.groupby("cow_id")[col].transform(lambda s: s.shift(1).rolling(7, min_periods=3).std())
    daily[col + "_z"] = ((daily[col] - b) / sd.replace(0, np.nan)).round(2)
daily.to_csv("synthetic_daily_10cows.csv", index=False)

# IMU clip summaries (10 Hz, ~1 min clips) matching real class counts
counts = {"Grazing": 105, "Walking": 217, "Resting": 10}
rare = ["Standing", "Licking", "Rising", "Lying_down", "Drinking", "Scratching", "Trotting", "Head_butting", "Defecating"]
rare_counts = [20, 18, 15, 14, 12, 10, 8, 7, 5]          # sums to 109 -> 441 total
assert sum(counts.values()) + sum(rare_counts) == 441 and len(rare) == 9
counts.update(dict(zip(rare, rare_counts)))
profile = {  # (acc_mag_mean, acc_mag_std, gyro_std, dominant_freq_hz)
    "Grazing": (1.02, .18, 18, 0.9), "Walking": (1.08, .35, 32, 1.6), "Resting": (1.00, .03, 2, 0.2),
    "Standing": (1.00, .05, 4, 0.3), "Licking": (1.01, .12, 22, 1.1), "Rising": (1.15, .45, 55, 0.5),
    "Lying_down": (1.05, .30, 40, 0.4), "Drinking": (1.02, .15, 20, 0.7), "Scratching": (1.10, .40, 48, 2.4),
    "Trotting": (1.25, .60, 60, 2.2), "Head_butting": (1.20, .55, 70, 1.8), "Defecating": (1.03, .10, 10, 0.3)}
clips = []
for lab, n in counts.items():
    m, s, gy, f = profile[lab]
    for _ in range(n):
        clips.append(dict(cow_id=f"cow_{rng.integers(1, N_COWS + 1):02d}", label=lab,
                          duration_s=round(float(rng.uniform(20, 90)), 1), sample_hz=10,
                          acc_mag_mean=round(float(rng.normal(m, .02)), 3), acc_mag_std=round(float(abs(rng.normal(s, s * .15))), 3),
                          gyro_std=round(float(abs(rng.normal(gy, gy * .2))), 2), dom_freq_hz=round(float(abs(rng.normal(f, .15))), 2)))
pd.DataFrame(clips).sample(frac=1, random_state=1).to_csv("synthetic_imu_clips_441.csv", index=False)

# sanity report
ev = hourly[hourly.heat == 1]; non = hourly[hourly.heat == 0]
print("rows hourly", len(hourly), "daily", len(daily), "clips", len(clips))
print("heat hours %", round(100 * hourly.heat.mean(), 2))
print("heat events per cow", hourly.groupby("cow_id").heat.apply(lambda s: int((s.diff() == 1).sum() + s.iloc[0])).tolist())
print("activity ratio heat/non", round(ev.activity_index.mean() / non.activity_index.mean(), 2))
print("rumination ratio heat/non", round(ev.rumination_min.mean() / non.rumination_min.mean(), 2))
print("head temp delta", round(ev.head_temp_c.mean() - non.head_temp_c.mean(), 2))
print("clip counts", pd.DataFrame(clips).label.value_counts().to_dict())
