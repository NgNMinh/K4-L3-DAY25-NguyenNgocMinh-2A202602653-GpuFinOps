"""M1 — Efficiency Audit: MFU/MBU, the GPU-Util lie, and idle waste (deck §5).
Extension 2: Right-sizing recommendation based on MBU and $/GB-VRAM.

Run: python missions/m1_efficiency_audit.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from collections import defaultdict
from missions._common import load_csv, num, catalog_by_type
from finops import metrics

# ── Extension 2 helpers ────────────────────────────────────────────────────────
# MBU target for inference workloads.  We only suggest a right-size when the
# GPU is clearly under-utilising bandwidth (< MBU_LOW_THRESHOLD) AND the model
# fits comfortably in the candidate GPU's VRAM.
MBU_LOW_THRESHOLD = 0.45   # below this: bandwidth is not the bottleneck → may over-provision
VRAM_HEADROOM = 1.15       # candidate must have ≥ 115% of observed peak mem (15% safety margin)


def _rightsize_candidates(gpu_type: str, max_mem_gb: float, cat: dict) -> list[dict]:
    """Return cheaper GPU options that have enough VRAM and comparable bandwidth.

    Selection criteria (in order):
    1. candidate on_demand_hr < current on_demand_hr  (cheaper per hour)
    2. candidate hbm_gb >= max_mem_gb × VRAM_HEADROOM  (enough VRAM)
    3. sorted by on_demand_hr ascending (cheapest first)

    We do NOT require the same peak_bw_tbs because for a memory-bound workload
    that is already bandwidth-under-utilised (MBU low), a lower-BW GPU may
    still be sufficient.  We flag this limitation in the output.
    """
    cur_price = float(cat[gpu_type]["on_demand_hr"])
    candidates = []
    for gt, row in cat.items():
        if gt == gpu_type:
            continue
        cand_price = float(row["on_demand_hr"])
        cand_hbm = float(row["hbm_gb"])
        if cand_price < cur_price and cand_hbm >= max_mem_gb * VRAM_HEADROOM:
            candidates.append({
                "gpu_type": gt,
                "on_demand_hr": cand_price,
                "hbm_gb": cand_hbm,
                "peak_bw_tbs": float(row["peak_bw_tbs"]),
                "usd_per_gb_vram": round(cand_price / cand_hbm, 4),
            })
    candidates.sort(key=lambda x: x["on_demand_hr"])
    return candidates


def run(verbose: bool = True) -> dict:
    tel = load_csv("gpu_telemetry.csv")
    cat = catalog_by_type()

    # per-row MFU/MBU, then aggregate per GPU
    agg = defaultdict(lambda: {"util": [], "mfu": [], "mbu": [], "type": None,
                                "idle_hours": 0, "max_mem_gb": 0.0})
    for r in tel:
        gtype = r["gpu_type"]
        peak_fp16 = num(cat[gtype]["peak_tflops_fp16"])
        peak_bw = num(cat[gtype]["peak_bw_tbs"])
        mfu = metrics.compute_mfu(num(r["achieved_tflops"]), peak_fp16)
        mbu = metrics.compute_mbu(num(r["achieved_bw_tbs"]), peak_bw)
        a = agg[r["gpu_id"]]
        a["type"] = gtype
        a["util"].append(num(r["gpu_util_pct"]))
        a["mfu"].append(mfu)
        a["mbu"].append(mbu)
        mem_gb = num(r.get("mem_used_gb", 0))
        if mem_gb > a["max_mem_gb"]:
            a["max_mem_gb"] = mem_gb
        if num(r["gpu_util_pct"]) < 10:  # effectively idle this interval (1h)
            a["idle_hours"] += 1

    summary = []
    for gid, a in agg.items():
        summary.append({
            "gpu_id": gid, "gpu_type": a["type"],
            "gpu_util_pct": round(sum(a["util"]) / len(a["util"]), 1),
            "mfu": round(sum(a["mfu"]) / len(a["mfu"]), 3),
            "mbu": round(sum(a["mbu"]) / len(a["mbu"]), 3),
            "idle_hours": a["idle_hours"],
            "max_mem_gb": round(a["max_mem_gb"], 1),
        })

    lies = metrics.flag_util_lies(summary)
    idle_waste = 0.0
    for s in summary:
        on_demand = num(catalog_by_type()[s["gpu_type"]]["on_demand_hr"])
        idle_waste += metrics.idle_waste_usd(s["idle_hours"], on_demand)

    # ── Extension 2: Right-sizing recommendations ──────────────────────────────
    rightsize_recs = []
    DAYS = 30
    for s in summary:
        gtype = s["gpu_type"]
        mbu = s["mbu"]
        max_mem = s["max_mem_gb"]
        on_demand_hr = num(cat[gtype]["on_demand_hr"])
        usd_per_gb = round(on_demand_hr / float(cat[gtype]["hbm_gb"]), 4)

        if mbu < MBU_LOW_THRESHOLD:
            candidates = _rightsize_candidates(gtype, max_mem, cat)
            if candidates:
                best = candidates[0]
                monthly_saving = (on_demand_hr - best["on_demand_hr"]) * 24 * DAYS
                rightsize_recs.append({
                    "gpu_id": s["gpu_id"],
                    "cur_type": gtype,
                    "cur_hr": on_demand_hr,
                    "cur_usd_per_gb": usd_per_gb,
                    "cur_bw_tbs": float(cat[gtype]["peak_bw_tbs"]),
                    "mbu": mbu,
                    "max_mem_gb": max_mem,
                    "rec_type": best["gpu_type"],
                    "rec_hr": best["on_demand_hr"],
                    "rec_usd_per_gb": best["usd_per_gb_vram"],
                    "rec_hbm_gb": best["hbm_gb"],
                    "rec_bw_tbs": best["peak_bw_tbs"],
                    "monthly_saving_usd": round(monthly_saving, 0),
                    "limitation": "throughput parity unverified — benchmark required before migration",
                })

    if verbose:
        print("== M1 Efficiency Audit ==")
        print(f"{'GPU':14}{'type':7}{'util%':>7}{'MFU':>7}{'MBU':>7}{'idle_h':>8}")
        for s in sorted(summary, key=lambda x: x["mfu"]):
            print(f"{s['gpu_id']:14}{s['gpu_type']:7}{s['gpu_util_pct']:>7}{s['mfu']:>7}{s['mbu']:>7}{s['idle_hours']:>8}")
        print(f"\nGPU-Util LIES (util>=90% but MFU<30%): {[l['gpu_id'] for l in lies]}")
        print(f"Idle waste (1 day): ${idle_waste:,.2f}  ->  ${idle_waste*30:,.0f}/month")

        # Extension 2 output
        print(f"\n── Extension 2: Right-sizing recommendations (MBU < {MBU_LOW_THRESHOLD:.0%}) ──")
        print(f"{'GPU':14}{'cur':7}{'$/hr':>6}{'$/GB':>7}{'MBU':>6}  →  "
              f"{'rec':7}{'$/hr':>6}{'$/GB':>7}{'hbm':>5}  saving/mo  note")
        if rightsize_recs:
            for r in rightsize_recs:
                print(f"{r['gpu_id']:14}{r['cur_type']:7}{r['cur_hr']:>6.2f}"
                      f"{r['cur_usd_per_gb']:>7.4f}{r['mbu']:>6.3f}  →  "
                      f"{r['rec_type']:7}{r['rec_hr']:>6.2f}{r['rec_usd_per_gb']:>7.4f}"
                      f"{r['rec_hbm_gb']:>5.0f}  ${r['monthly_saving_usd']:>8,.0f}"
                      f"  ⚠ {r['limitation']}")
        else:
            print("  No right-size candidates found (all GPUs either high MBU or no cheaper VRAM-adequate option).")
        total_rs_saving = sum(r["monthly_saving_usd"] for r in rightsize_recs)
        print(f"  Total potential monthly saving if all right-sized: ${total_rs_saving:,.0f}")
        print(f"  (Assumption: {VRAM_HEADROOM:.0%} VRAM headroom required; "
              f"benchmark to confirm throughput parity before migrating)")

    return {
        "summary": summary,
        "lies": lies,
        "idle_waste_daily": round(idle_waste, 2),
        "rightsize_recs": rightsize_recs,
    }


if __name__ == "__main__":
    run()
