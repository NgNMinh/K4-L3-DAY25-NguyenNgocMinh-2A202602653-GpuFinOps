"""M3 — Purchasing Strategy: break-even, tier choice, spot-checkpoint sim (deck §4).
Extension 1: GPU-aware interruption rates + 1yr vs 3yr reserved.
Extension 5: Carbon-aware scheduling for interruptible jobs.

Run: python missions/m3_purchasing.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from missions._common import load_csv, num, catalog_by_type
from finops import pricing, sustainability

DAYS = 30


def run(verbose: bool = True) -> dict:
    jobs = load_csv("workloads.csv")
    cat = catalog_by_type()
    on_demand_monthly = optimized_monthly = 0.0
    recs = []
    for j in jobs:
        gtype = j["gpu_type"]
        ngpu = int(num(j["num_gpus"]))
        hpd = num(j["hours_per_day"])
        interruptible = bool(int(num(j["interruptible"])))
        job_days = int(num(j.get("days", 30)))
        c = cat[gtype]
        gpu_hours = hpd * DAYS * ngpu
        od = num(c["on_demand_hr"])
        on_demand_cost = gpu_hours * od

        # Extension 1: pass gpu_type and job_days to the improved recommend_tier
        tier = pricing.recommend_tier(hpd, interruptible, gpu_type=gtype, job_days=job_days)
        detail = pricing.recommend_tier_with_detail(
            hpd, interruptible, gpu_type=gtype, job_days=job_days,
            on_demand_hr=od, spot_hr=num(c["spot_hr"])
        )
        if tier == "spot":
            sim = pricing.spot_checkpoint_cost(
                gpu_hours, num(c["spot_hr"]), od,
                interrupt_rate=pricing.GPU_SPOT_INTERRUPT_RATE.get(gtype, 0.05),
            )
            opt_cost = sim["spot_cost"]
        elif tier == "reserved":
                # Extension 1: use 1yr vs 3yr rate based on job duration
            # Job clearly temporary (< 30 days window) → 1yr; otherwise 3yr production rate
            if job_days < 30:
                opt_cost = gpu_hours * num(c["reserved_1yr_hr"])
            else:
                opt_cost = gpu_hours * num(c["reserved_3yr_hr"])
        else:
            opt_cost = on_demand_cost

        on_demand_monthly += on_demand_cost
        optimized_monthly += opt_cost
        recs.append({"job_id": j["job_id"], "gpu_type": gtype, "tier": tier,
                     "job_days": job_days,
                     "on_demand": round(on_demand_cost), "optimized": round(opt_cost),
                     "note": detail["note"]})

    savings = on_demand_monthly - optimized_monthly
    savings_pct = savings / on_demand_monthly * 100 if on_demand_monthly else 0.0

    if verbose:
        print("== M3 Purchasing Strategy (Extension 1: GPU-aware tier selection) ==")
        print(f"break-even @ 1yr(30% disc)={pricing.break_even_utilization(0.30):.0%}  "
              f"@ 3yr(45% disc)={pricing.break_even_utilization(0.45):.0%}")
        print(f"\nGPU spot interruption rates used:")
        for gtype, rate in pricing.GPU_SPOT_INTERRUPT_RATE.items():
            print(f"  {gtype:8} {rate:.0%}/hr  (assumption — see pricing.py docstring)")
        print(f"\n{'job':18}{'gpu':7}{'days':>6}{'tier':11}{'on-demand':>12}{'optimized':>12}  note")
        for r in recs:
            print(f"{r['job_id']:18}{r['gpu_type']:7}{r['job_days']:>6}{r['tier']:11}"
                  f"${r['on_demand']:>11,}${r['optimized']:>11,}  {r['note']}")
        print(f"\nmonthly: on-demand ${on_demand_monthly:,.0f} -> optimized ${optimized_monthly:,.0f}"
              f"  ({savings_pct:.1f}% saved)")

        # ── Extension 5: Carbon-aware scheduling ──────────────────────────────
        print(f"\n── Extension 5: Carbon-aware Scheduling (interruptible jobs only) ──")
        print(f"  Period: {DAYS} days  |  Baseline region: us-east-1")
        print(f"  Energy estimated from catalog watts × GPU-hours\n")
        print(f"  {'Region':20} {'gCO2/kWh':>10} {'$/kWh':>8}")
        for reg, gco2 in sorted(sustainability.REGION_CARBON.items(), key=lambda x: x[1]):
            pkwh = sustainability.REGION_PRICE_KWH.get(reg, 0.12)
            print(f"  {reg:20} {gco2:>10,} {pkwh:>8.3f}")

        print(f"\n  {'job':18}{'gpu':7}  {'Wh':>10}  {'us-east-1':>12}  {'europe-north1':>14}  {'saving gCO2':>12}")
        total_base_co2 = total_best_co2 = 0.0
        interruptible_jobs = [j for j in load_csv("workloads.csv") if bool(int(num(j["interruptible"])))]
        cat2 = catalog_by_type()
        for j in interruptible_jobs:
            gtype = j["gpu_type"]
            ngpu = int(num(j["num_gpus"]))
            hpd = num(j["hours_per_day"])
            gpu_hours = hpd * DAYS * ngpu
            watts = num(cat2[gtype]["watts"])
            wh = watts * gpu_hours                          # Wh for the whole job
            co2_base = sustainability.carbon_g(wh, "us-east-1")
            co2_best = sustainability.carbon_g(wh, "europe-north1")
            saving = co2_base - co2_best
            total_base_co2 += co2_base
            total_best_co2 += co2_best
            print(f"  {j['job_id']:18}{gtype:7}  {wh:>10,.0f}  {co2_base/1e6:>10.3f} kgCO2"
                  f"  {co2_best/1e6:>12.3f} kgCO2  {saving/1e6:>10.3f} kg")

        total_saving_kg = (total_base_co2 - total_best_co2) / 1e6
        total_saving_pct = (total_base_co2 - total_best_co2) / total_base_co2 * 100 if total_base_co2 else 0
        print(f"\n  Total carbon saving (interruptible jobs, {DAYS}d): "
              f"{total_saving_kg:,.1f} kgCO2e  ({total_saving_pct:.0f}% reduction)")
        print(f"  Cheapest electricity: us-east-wa (${sustainability.REGION_PRICE_KWH['us-east-wa']:.3f}/kWh)")
        print(f"  Cleanest grid      : europe-north1 ({sustainability.REGION_CARBON['europe-north1']} gCO2/kWh)")
        print(f"  Best balance (low $ + low CO2): us-west-2 or europe-north1")
        print(f"  ⚠ Trade-off note: europe-north1 (Norway) may add network latency for US-based users;")
        print(f"    interruptible batch/training jobs are latency-insensitive and are ideal candidates.")

    return {"recommendations": recs, "on_demand_monthly": round(on_demand_monthly),
            "optimized_monthly": round(optimized_monthly), "savings_pct": round(savings_pct, 1)}


if __name__ == "__main__":
    run()
