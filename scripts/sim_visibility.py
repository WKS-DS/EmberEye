"""
sim_visibility.py — How much of the burning fire does each simulated frame show?

Usage:
    python scripts/sim_visibility.py --scenario hard --path orbit [--duration 1200] [--interval 30]

Flies the path over the scenario and prints, per frame, how many cells are
burning, how many of them at least one pixel shows, and how long the frame
took to render. Cells go unseen when they are outside the frame, hidden
behind terrain, or too far away to cover a pixel. Nothing is written to disk.

The demo orbit sees nearly 100% of the fire in every frame, so it never
tests coverage gaps or occlusion. The hard scenario should not.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from embereye.sim.fire_sim import BURNING, PATHS, SCENARIOS, FireSim, make_path, visible_burning_fraction


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenario", choices=list(SCENARIOS), default="hard")
    parser.add_argument("--path", choices=PATHS, default="orbit")
    parser.add_argument("--duration", type=float, default=1200.0, help="seconds of flight")
    parser.add_argument("--interval", type=float, default=30.0, help="seconds between frames")
    args = parser.parse_args()

    scenario = SCENARIOS[args.scenario]()
    started = time.perf_counter()
    with FireSim(scenario, make_path(args.path, scenario.grid)) as sim:
        print(f"{args.scenario} scenario, {args.path} path: "
              f"{scenario.grid.rows}x{scenario.grid.cols} cells, built in {time.perf_counter() - started:.1f} s\n")
        print(" frame  time_s  burning  visible  fraction  render_s")
        fractions, render_s = [], []
        for index in range(int(args.duration // args.interval) + 1):
            t0 = time.perf_counter()
            obs = sim.observe(index * args.interval, index)
            render_s.append(time.perf_counter() - t0)
            fraction = visible_burning_fraction(obs)
            fractions.append(fraction)
            n_burning = int((obs.truth_state == BURNING).sum())
            print(f"{index:6d}  {obs.time_s:6.0f}  {n_burning:7d}  {round(fraction * n_burning):7d}  "
                  f"{fraction:8.2f}  {render_s[-1]:8.3f}")

    fractions = np.array(fractions)
    print(f"\nvisible fraction: median {np.median(fractions):.2f}, min {fractions.min():.2f}; "
          f"frames below 0.9: {(fractions < 0.9).sum()} of {len(fractions)}")
    print(f"render time per frame: median {np.median(render_s):.3f} s, max {max(render_s):.3f} s")


if __name__ == "__main__":
    main()
