"""Deterministic synthetic cattle data generator per PRD Section 4 (FR-S1 to FR-S7).

Generates:
- data/cows.csv (one row per cow)
- data/daily.csv (one row per cow per day, DIM 1..305)
- data/inseminations.csv (one row per insemination event)
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent


def load_config(config_path: Path | None = None) -> dict:
    path = config_path or (ROOT / "config.yaml")
    if path.exists():
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    return {}


def ar1_series(rng: np.random.Generator, n: int, rho: float, sd: float) -> np.ndarray:
    """Generate a zero-mean AR(1) series with marginal or innovation SD."""
    eps = rng.normal(0.0, sd, size=n)
    out = np.zeros(n, dtype=float)
    out[0] = eps[0]
    for t in range(1, n):
        out[t] = rho * out[t - 1] + eps[t]
    return out


def wood_curve(dim: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    return a * np.power(dim, b) * np.exp(-c * dim)


def simulate_herd(
    n_cows: int = 300,
    seed: int = 42,
    out_dir: Path | None = None,
    config_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    cfg_all = load_config(config_path)
    sim = cfg_all.get("simulator", {})

    rng = np.random.default_rng(seed)
    lactation_days = int(sim.get("lactation_days", 305))
    dims = np.arange(1, lactation_days + 1, dtype=int)

    parity_probs = sim.get("parity_probs", [0.38, 0.30, 0.32])
    holstein_mean = float(sim.get("holstein_mean", 0.88))
    holstein_sd = float(sim.get("holstein_sd", 0.10))
    holstein_clip = sim.get("holstein_clip", [0.50, 1.00])

    wood_a_base = sim.get("wood_a_base", [16.0, 20.0, 23.0])
    wood_a_milk_coef = float(sim.get("wood_a_milk_coef", 0.08))
    wood_a_noise_sd = float(sim.get("wood_a_noise_sd", 0.06))
    wood_b_base = sim.get("wood_b_base", [0.18, 0.22, 0.24])
    wood_b_noise_sd = float(sim.get("wood_b_noise_sd", 0.015))
    wood_c_base = sim.get("wood_c_base", [0.0020, 0.0030, 0.0034])
    wood_c_noise_sd = float(sim.get("wood_c_noise_sd", 0.0002))

    base_act_mean = float(sim.get("baseline_activity_mean", 100.0))
    base_act_sd = float(sim.get("baseline_activity_sd", 12.0))
    base_act_g_coef = float(sim.get("baseline_activity_g_coef", 4.0))
    base_rum_mean = float(sim.get("baseline_rumination_mean", 480.0))
    base_rum_sd = float(sim.get("baseline_rumination_sd", 35.0))
    base_rum_g_coef = float(sim.get("baseline_rumination_g_coef", 15.0))
    base_lying_mean = float(sim.get("baseline_lying_mean", 600.0))
    base_lying_sd = float(sim.get("baseline_lying_sd", 50.0))
    base_temp_mean = float(sim.get("baseline_temp_mean", 38.6))
    base_temp_sd = float(sim.get("baseline_temp_sd", 0.15))

    technicians = sim.get("technicians", ["T1", "T2", "T3"])
    semen_types = sim.get("semen_types", ["sexed_dairy", "conventional_dairy", "beef"])
    parity3_semen_probs = sim.get("parity3_semen_probs", [0.35, 0.40, 0.25])
    vwp_range = sim.get("vwp_range", [50, 74])
    first_estrus_range = sim.get("first_estrus_range", [22, 47])

    cycle_mean = float(sim.get("cycle_mean", 21.0))
    cycle_sd = float(sim.get("cycle_sd", 2.5))
    cycle_clip = sim.get("cycle_clip", [17, 26])
    silent_heat_prob = float(sim.get("silent_heat_prob", 0.15))
    silent_heat_shrink = float(sim.get("silent_heat_shrink", 0.25))
    silent_insem_prob = float(sim.get("silent_insemination_prob", 0.30))

    duration_base = float(sim.get("duration_base", 13.0))
    duration_g_coef = float(sim.get("duration_g_coef", 0.8))
    duration_sd = float(sim.get("duration_sd", 3.5))
    duration_clip = sim.get("duration_clip", [4.0, 24.0])

    s_act_median_base = float(sim.get("s_act_median_base", 1.9))
    s_act_g_coef = float(sim.get("s_act_g_coef", 0.12))
    s_act_sigma = float(sim.get("s_act_sigma", 0.25))
    s_act_clip = sim.get("s_act_clip", [1.0, 4.5])

    s_rum_base = float(sim.get("s_rum_base", 0.80))
    s_rum_g_coef = float(sim.get("s_rum_g_coef", -0.03))
    s_rum_sd = float(sim.get("s_rum_sd", 0.08))
    s_rum_clip = sim.get("s_rum_clip", [0.45, 1.0])

    ar1_act = sim.get("ar1_activity", [0.5, 5.0])
    ar1_rum = sim.get("ar1_rumination", [0.5, 18.0])
    ar1_lying = sim.get("ar1_lying", [0.5, 30.0])
    ar1_temp = sim.get("ar1_temp", [0.4, 0.08])
    milk_noise_sd = float(sim.get("milk_noise_sd", 0.04))
    pre_heat_frac = float(sim.get("pre_heat_activity_frac", 0.15))
    preg_drop_max = float(sim.get("pregnancy_milk_drop_max", 0.12))
    preg_drop_start = int(sim.get("pregnancy_drop_start_days", 120))
    preg_drop_ramp = float(sim.get("pregnancy_drop_ramp_days", 150.0))
    missing_sensor_rate = float(sim.get("missing_sensor_rate", 0.02))

    cows_rows = []
    daily_rows = []
    insem_rows = []
    cycle_lengths = []

    for cow_id in range(1, n_cows + 1):
        parity = int(rng.choice([1, 2, 3], p=parity_probs))
        pct_holstein = float(np.clip(rng.normal(holstein_mean, holstein_sd), holstein_clip[0], holstein_clip[1]))
        calving_doy = int(rng.integers(0, 365))

        g_act = float(rng.normal(0.0, 1.0))
        g_rum = float(rng.normal(0.0, 1.0))
        g_milk = float(rng.normal(0.0, 1.0))

        p_idx = parity - 1
        wood_a = float(wood_a_base[p_idx] * np.exp(wood_a_milk_coef * g_milk + rng.normal(0.0, wood_a_noise_sd)))
        wood_b = float(wood_b_base[p_idx] + rng.normal(0.0, wood_b_noise_sd))
        wood_c = float(wood_c_base[p_idx] + rng.normal(0.0, wood_c_noise_sd))

        base_act = float(rng.normal(base_act_mean, base_act_sd) + base_act_g_coef * g_act)
        base_rum = float(rng.normal(base_rum_mean, base_rum_sd) + base_rum_g_coef * g_rum)
        base_lying = float(rng.normal(base_lying_mean, base_lying_sd))
        base_temp = float(rng.normal(base_temp_mean, base_temp_sd))

        technician = str(rng.choice(technicians))
        if parity == 3:
            semen_plan = str(rng.choice(semen_types, p=parity3_semen_probs))
        else:
            semen_plan = "sexed_dairy"

        vwp = int(rng.integers(vwp_range[0], vwp_range[1] + 1))
        first_estrus = int(rng.integers(first_estrus_range[0], first_estrus_range[1] + 1))

        cows_rows.append(
            {
                "cow_id": cow_id,
                "parity": parity,
                "pct_holstein": round(pct_holstein, 4),
                "calving_doy": calving_doy,
                "technician": technician,
                "semen_plan": semen_plan,
                "g_activity": round(g_act, 4),
                "g_rumination": round(g_rum, 4),
                "g_milk": round(g_milk, 4),
                "is_synthetic": 1,
            }
        )

        # Simulate estrus loop from first_estrus until conception or day 305
        heats: dict[int, dict] = {}
        conception_dim: int | None = None
        current_dim = first_estrus

        while current_dim <= lactation_days:
            silent = int(rng.random() < silent_heat_prob)
            dur_raw = float(
                np.clip(rng.normal(duration_base + duration_g_coef * g_act, duration_sd), duration_clip[0], duration_clip[1])
            )
            med_act = max(1.05, s_act_median_base + s_act_g_coef * g_act)
            s_act_raw = float(np.clip(rng.lognormal(np.log(med_act), s_act_sigma), s_act_clip[0], s_act_clip[1]))
            s_rum_raw = float(np.clip(rng.normal(s_rum_base + s_rum_g_coef * g_rum, s_rum_sd), s_rum_clip[0], s_rum_clip[1]))

            if silent == 1:
                s_act = 1.0 + silent_heat_shrink * (s_act_raw - 1.0)
                s_rum = 1.0 - silent_heat_shrink * (1.0 - s_rum_raw)
                dur_h = float(np.clip(duration_clip[0] + silent_heat_shrink * (dur_raw - duration_clip[0]), duration_clip[0], duration_clip[1]))
            else:
                s_act = s_act_raw
                s_rum = s_rum_raw
                dur_h = dur_raw

            heats[current_dim] = {
                "silent": silent,
                "s_act": s_act,
                "s_rum": s_rum,
                "duration_h": dur_h,
            }

            if current_dim >= vwp:
                do_insem = (silent == 0) or (rng.random() < silent_insem_prob)
                if do_insem:
                    tau = 0.20 if technician == "T1" else (-0.05 if technician == "T3" else 0.0)
                    is_sexed = 1.0 if semen_plan == "sexed_dairy" else 0.0
                    is_beef = 1.0 if semen_plan == "beef" else 0.0
                    logit_p = (
                        -0.35
                        + 0.55 * np.log(max(s_act, 1e-6))
                        - 1.0 * (1.0 - s_rum)
                        - 0.175 * (parity - 1)
                        - 0.25 * is_sexed
                        + 0.15 * is_beef
                        - 0.004 * max(current_dim - 100, 0)
                        + tau
                    )
                    p_conc = float(1.0 / (1.0 + np.exp(-logit_p)))
                    conceived = int(rng.random() < p_conc)
                    insem_rows.append(
                        {
                            "cow_id": cow_id,
                            "dim": current_dim,
                            "parity": parity,
                            "semen_type": semen_plan,
                            "technician": technician,
                            "estrus_activity_strength": round(s_act, 4),
                            "estrus_rumination_strength": round(s_rum, 4),
                            "estrus_duration_h": round(dur_h, 2),
                            "silent_heat": silent,
                            "p_conception_true": round(p_conc, 4),
                            "conceived": conceived,
                            "is_synthetic": 1,
                        }
                    )
                    if conceived == 1:
                        conception_dim = current_dim
                        break

            cycle = int(np.clip(np.round(rng.normal(cycle_mean, cycle_sd)), cycle_clip[0], cycle_clip[1]))
            cycle_lengths.append(cycle)
            current_dim += cycle

        # Generate daily signals over DIM 1..305
        act_noise = ar1_series(rng, lactation_days, float(ar1_act[0]), float(ar1_act[1]))
        rum_noise = ar1_series(rng, lactation_days, float(ar1_rum[0]), float(ar1_rum[1]))
        lying_noise = ar1_series(rng, lactation_days, float(ar1_lying[0]), float(ar1_lying[1]))
        temp_noise = ar1_series(rng, lactation_days, float(ar1_temp[0]), float(ar1_temp[1]))

        act = base_act + act_noise
        rum = base_rum + rum_noise
        lying = base_lying + lying_noise
        temp = base_temp + temp_noise

        base_milk = wood_curve(dims.astype(float), wood_a, wood_b, wood_c)
        milk_mult_noise = rng.normal(1.0, milk_noise_sd, size=lactation_days)
        milk = base_milk * milk_mult_noise

        estrus_day_arr = np.zeros(lactation_days, dtype=int)
        estrus_dur_arr = np.zeros(lactation_days, dtype=float)
        pregnant_arr = np.zeros(lactation_days, dtype=int)

        if conception_dim is not None:
            pregnant_arr[conception_dim - 1 :] = 1
            for d_idx in range(conception_dim - 1, lactation_days):
                days_preg = (d_idx + 1) - conception_dim
                if days_preg >= preg_drop_start:
                    drop = preg_drop_max * min(1.0, (days_preg - float(preg_drop_start)) / preg_drop_ramp)
                    milk[d_idx] *= 1.0 - drop

        for h_dim, h_info in heats.items():
            idx = h_dim - 1
            s_act = h_info["s_act"]
            s_rum = h_info["s_rum"]
            estrus_day_arr[idx] = 1
            estrus_dur_arr[idx] = h_info["duration_h"]

            # Day before heat: activity rises by 15% of the next heat's excess
            if idx - 1 >= 0:
                act[idx - 1] *= 1.0 + pre_heat_frac * (s_act - 1.0)

            # Heat day sensor modifications
            act[idx] *= s_act
            rum[idx] *= s_rum
            lying[idx] *= 1.0 - 0.12 * (s_act - 1.0) / 1.5
            temp[idx] += 0.25 * (s_act - 1.0)
            milk[idx] *= 1.0 - 0.03 * (s_act - 1.0)

        act = np.clip(act, 10.0, 600.0)
        rum = np.clip(rum, 60.0, 900.0)
        lying = np.clip(lying, 120.0, 1100.0)
        temp = np.clip(temp, 36.5, 41.5)
        milk = np.clip(milk, 2.0, 85.0)

        # 2% of cow-days lose all four sensor readings
        missing_mask = rng.random(lactation_days) < missing_sensor_rate
        for d_idx in range(lactation_days):
            is_miss = bool(missing_mask[d_idx])
            daily_rows.append(
                {
                    "cow_id": cow_id,
                    "dim": int(d_idx + 1),
                    "parity": parity,
                    "activity": np.nan if is_miss else round(float(act[d_idx]), 2),
                    "rumination_min": np.nan if is_miss else round(float(rum[d_idx]), 2),
                    "lying_min": np.nan if is_miss else round(float(lying[d_idx]), 2),
                    "body_temp_c": np.nan if is_miss else round(float(temp[d_idx]), 3),
                    "milk_kg": round(float(milk[d_idx]), 2),
                    "estrus_day": int(estrus_day_arr[d_idx]),
                    "estrus_duration_h": round(float(estrus_dur_arr[d_idx]), 2),
                    "pregnant": int(pregnant_arr[d_idx]),
                    "is_synthetic": 1,
                }
            )

    cows_df = pd.DataFrame(cows_rows)
    daily_df = pd.DataFrame(daily_rows)
    insem_df = pd.DataFrame(insem_rows)

    stats = {
        "n_cows": int(len(cows_df)),
        "cow_days": int(len(daily_df)),
        "inseminations": int(len(insem_df)),
        "estrus_day_rate": float(daily_df["estrus_day"].mean()),
        "conception_rate": float(insem_df["conceived"].mean()) if len(insem_df) else 0.0,
        "mean_milk_kg": float(daily_df["milk_kg"].mean()),
        "mean_cycle_length": float(np.mean(cycle_lengths)) if cycle_lengths else 21.0,
        "missing_sensor_rate": float(daily_df["activity"].isna().mean()),
    }

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        cows_df.to_csv(out_dir / "cows.csv", index=False)
        daily_df.to_csv(out_dir / "daily.csv", index=False)
        insem_df.to_csv(out_dir / "inseminations.csv", index=False)

    return cows_df, daily_df, insem_df, stats


def simulate_hourly_from_daily(daily_df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Disaggregate daily rows into 24 hourly bins (FR-S7) preserving daily totals
    while concentrating activity inside the heat window (estrus_duration_h)."""
    rng = np.random.default_rng(seed)
    hours = np.arange(24, dtype=int)
    diurnal = 0.55 + 0.45 * np.sin((hours - 6.0) / 24.0 * 2.0 * np.pi) ** 2

    rows = []
    for row in daily_df.itertuples(index=False):
        cow_id = int(row.cow_id)
        dim = int(row.dim)
        parity = int(row.parity)
        is_heat = int(row.estrus_day) == 1
        dur_h = int(np.clip(np.round(float(row.estrus_duration_h)), 1, 24)) if is_heat else 0

        heat_hours = np.zeros(24, dtype=int)
        if is_heat and dur_h > 0:
            start_h = int(rng.integers(0, max(1, 24 - dur_h + 1)))
            heat_hours[start_h : start_h + dur_h] = 1

        # Activity weights concentrated in the heat window while preserving daily sum/mean
        act_weights = diurnal * np.where(heat_hours == 1, 2.4, 0.85)
        act_weights = act_weights / np.sum(act_weights)

        rum_weights = (1.15 - 0.35 * diurnal) * np.where(heat_hours == 1, 0.65, 1.05)
        rum_weights = rum_weights / np.sum(rum_weights)

        lying_weights = (1.20 - 0.40 * diurnal) * np.where(heat_hours == 1, 0.60, 1.05)
        lying_weights = lying_weights / np.sum(lying_weights)

        daily_act = float(row.activity) if pd.notna(row.activity) else np.nan
        daily_rum = float(row.rumination_min) if pd.notna(row.rumination_min) else np.nan
        daily_lying = float(row.lying_min) if pd.notna(row.lying_min) else np.nan
        daily_temp = float(row.body_temp_c) if pd.notna(row.body_temp_c) else np.nan

        for h in range(24):
            h_flag = int(heat_hours[h])
            rows.append(
                {
                    "cow_id": cow_id,
                    "dim": dim,
                    "hour": h,
                    "parity": parity,
                    "activity": np.nan if np.isnan(daily_act) else round(daily_act * 24.0 * float(act_weights[h]), 4),
                    "rumination_min": np.nan if np.isnan(daily_rum) else round(daily_rum * float(rum_weights[h]), 4),
                    "lying_min": np.nan if np.isnan(daily_lying) else round(daily_lying * float(lying_weights[h]), 4),
                    "body_temp_c": np.nan
                    if np.isnan(daily_temp)
                    else round(daily_temp + (0.18 if h_flag else -0.04), 3),
                    "heat": h_flag,
                    "is_synthetic": int(row.is_synthetic),
                }
            )
    return pd.DataFrame(rows)


def check_validation_targets(stats: dict, n_cows: int = 300) -> None:
    expected_days = n_cows * 305
    scale = max(1.0, float(np.sqrt(300.0 / max(1, n_cows))))
    if stats["cow_days"] != expected_days:
        raise ValueError(f"Expected {expected_days} cow_days, got {stats['cow_days']}")
    if not (0.016 - 0.005 * scale <= stats["estrus_day_rate"] <= 0.016 + 0.006 * scale):
        raise ValueError(f"Estrus-day rate {stats['estrus_day_rate']:.4f} outside expected ~1.6% window")
    if not (0.35 - 0.08 * scale <= stats["conception_rate"] <= 0.35 + 0.08 * scale):
        raise ValueError(f"Conception rate {stats['conception_rate']:.4f} outside expected ~35% window")
    if not (34.5 - 3.0 * scale <= stats["mean_milk_kg"] <= 34.5 + 3.0 * scale):
        raise ValueError(f"Mean milk {stats['mean_milk_kg']:.2f} kg outside expected ~34.5 kg window")
    if not (21.0 - 1.0 * scale <= stats["mean_cycle_length"] <= 21.0 + 1.0 * scale):
        raise ValueError(f"Mean cycle length {stats['mean_cycle_length']:.2f} outside [20, 22] days")
    if not (0.02 - 0.008 * scale <= stats["missing_sensor_rate"] <= 0.02 + 0.010 * scale):
        raise ValueError(f"Missing sensor rate {stats['missing_sensor_rate']:.4f} outside ~2% window")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("n_cows_pos", nargs="?", type=int, default=None, help="Number of cows (positional)")
    parser.add_argument("--n-cows", type=int, default=300, help="Number of synthetic cows (default 300)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default 42)")
    parser.add_argument("--out", type=Path, default=ROOT / "data", help="Output directory")
    args = parser.parse_args()

    n_cows = args.n_cows_pos if args.n_cows_pos is not None else args.n_cows
    _, _, _, stats = simulate_herd(n_cows=n_cows, seed=args.seed, out_dir=args.out)
    check_validation_targets(stats, n_cows=n_cows)
    print(
        f"Generated {stats['n_cows']} cows ({stats['cow_days']} cow-days, "
        f"{stats['inseminations']} inseminations) -> {args.out}"
    )
    print(
        f"Stats: estrus_rate={stats['estrus_day_rate']*100:.2f}%, "
        f"conception_rate={stats['conception_rate']*100:.1f}%, "
        f"mean_milk={stats['mean_milk_kg']:.2f} kg/d, "
        f"mean_cycle={stats['mean_cycle_length']:.2f} d, "
        f"missing_rate={stats['missing_sensor_rate']*100:.2f}%"
    )


if __name__ == "__main__":
    main()
