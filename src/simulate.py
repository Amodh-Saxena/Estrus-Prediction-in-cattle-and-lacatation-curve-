"""CLI wrapper and exports for the config-driven PRD simulator (FR-S1 to FR-S7)."""
import argparse
from pathlib import Path

from generate_data import (
    simulate_herd,
    simulate_hourly_from_daily,
    check_validation_targets,
    wood_curve,
)

ROOT = Path(__file__).resolve().parents[1]

__all__ = [
    "simulate_herd",
    "simulate_hourly_from_daily",
    "check_validation_targets",
    "wood_curve",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-cows", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=ROOT / "data")
    args = parser.parse_args()
    _, _, _, stats = simulate_herd(args.n_cows, args.seed, args.out)
    check_validation_targets(stats, args.n_cows)
    print(
        f"Generated {stats['n_cows']} cows and {stats['cow_days']} cow-days "
        f"({stats['inseminations']} inseminations) at {args.out}"
    )
    print(
        f"Estrus {stats['estrus_day_rate']:.2%}; conception {stats['conception_rate']:.1%}; "
        f"mean milk {stats['mean_milk_kg']:.2f} kg/day; cycle {stats['mean_cycle_length']:.2f} days; "
        f"sensor missingness {stats['missing_sensor_rate']:.2%}"
    )


if __name__ == "__main__":
    main()
