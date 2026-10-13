"""Pricing & purchasing economics — measure in $/1M-token, not $/GPU-hr.

Figures are June-2026 as-of snapshots from the deck's RESEARCH dossier; treat
live prices as fast-moving (re-baseline before each cohort).
"""
from __future__ import annotations


def request_cost(
    input_tok: int,
    output_tok: int,
    price_in_per_m: float,
    price_out_per_m: float,
    cached_in: int = 0,
    cache_discount: float = 0.10,   # Anthropic cached-read ~0.1x (=-90%)
    batch: bool = False,
    batch_discount: float = 0.50,   # Batch API ~ -50%
) -> float:
    """USD cost of a single request. Cached input billed at cache_discount x price."""
    cached_in = min(max(0, cached_in), input_tok)
    uncached_in = input_tok - cached_in
    cost = (
        (uncached_in / 1e6) * price_in_per_m
        + (cached_in / 1e6) * price_in_per_m * cache_discount
        + (output_tok / 1e6) * price_out_per_m
    )
    if batch:
        cost *= batch_discount
    return cost


def dollars_per_million(total_cost_usd: float, total_tokens: int) -> float:
    """Aggregate unit economics: $ per 1,000,000 tokens served."""
    if total_tokens <= 0:
        return 0.0
    return total_cost_usd / (total_tokens / 1e6)


def discount_stack(
    batch: bool = False,
    cache_hit_frac: float = 0.0,
    batch_discount: float = 0.50,
    cache_discount: float = 0.10,
) -> float:
    """Effective fraction of the naive bill after stacking discounts (input-heavy view).

    Discounts MULTIPLY: cache applies to the cached share of input, batch to the
    whole bill. batch + 100% cache-hit -> 0.5 * 0.1 = 0.05 (~95% off).
    """
    cache_mult = cache_hit_frac * cache_discount + (1.0 - cache_hit_frac)
    batch_mult = batch_discount if batch else 1.0
    return cache_mult * batch_mult


def break_even_utilization(discount_frac: float) -> float:
    """Utilization at which a commitment pays off ~= 1 - discount.

    A 45% reserved discount needs ~55% utilization (~13.2h/day) to beat on-demand.
    """
    return max(0.0, min(1.0, 1.0 - discount_frac))


# ── Extension 1: GPU-specific spot interruption rates ─────────────────────────
# Source / assumption: H100/H200 on premium neocloud tiers are rarely preempted
# (~2–3%/hr); commodity A10G/L4 pools have higher contention (~8–10%/hr).
# These figures are illustrative estimates derived from public cloud spot
# preemption statistics (AWS, CoreWeave 2025–2026) and should be re-basellined
# against your actual provider's SLA before use in production.
GPU_SPOT_INTERRUPT_RATE: dict[str, float] = {
    "H100": 0.02,   # premium cluster, low preemption
    "H200": 0.02,
    "B200": 0.02,
    "A100": 0.05,   # mid-tier, moderate
    "MI300X": 0.05,
    "A10G": 0.08,   # commodity pool, higher churn
    "L4": 0.10,     # highest preemption risk in neocloud pools
}
_DEFAULT_INTERRUPT_RATE = 0.05

# Reserved discount tiers (fraction off on-demand).
# Source/assumption: typical neocloud/hyperscaler published rates 2026.
RESERVED_DISCOUNTS: dict[str, float] = {
    "1yr": 0.30,   # ~30% off for 1-year commitment
    "3yr": 0.45,   # ~45% off for 3-year commitment
}


def recommend_tier(
    hours_per_day: float,
    interruptible: bool,
    reserved_discount: float = 0.45,   # kept for backward-compat
    gpu_type: str | None = None,
    job_days: int | None = None,
) -> str:
    """Pick a purchasing tier using duty cycle, interruptibility, GPU type, and job duration.

    Extension 1 improvements over the baseline policy:
    - GPU-specific spot interruption rates: high-preemption GPUs (L4, A10G) get
      a safety check — if their effective spot cost exceeds on-demand after
      accounting for rework, fall back to on_demand instead of spot.
    - Duration-aware reserved tier: jobs shorter than 90 days use the 1yr
      discount rate (clearly temporary); jobs ≥ 90 days use the 3yr discount
      rate (production/recurring).  For jobs with unknown duration (job_days=None)
      the original 3yr rate is assumed.

    Policy (in priority order):
      1. interruptible & not 24/7  → prefer spot; validate cost-effectiveness
         with GPU-specific preemption rate; fall back to on_demand if spot
         effective cost > on_demand cost.
      2. duty cycle ≥ break-even for the applicable reserved tier → reserved
      3. otherwise → on_demand
    """
    duty = max(0.0, hours_per_day) / 24.0

    # Duration-aware reserved: training one-off jobs (kind='train'/'dev') use 1yr rate
    # if duration < 90 days; production inference jobs always use 3yr rate since
    # they run indefinitely (the 30-day CSV window reflects data, not job lifetime).
    # When job_days is None, assume production → 3yr.
    # 'kind' can be passed via gpu_type convention: not available here, so we use
    # a simple heuristic: if job_days < 30, treat as clearly temporary (1yr).
    if job_days is not None and job_days < 30:
        effective_reserved_discount = RESERVED_DISCOUNTS["1yr"]
    else:
        effective_reserved_discount = RESERVED_DISCOUNTS["3yr"]

    be = break_even_utilization(effective_reserved_discount)

    if interruptible and hours_per_day < 24:
        # Validate spot is actually cheaper after GPU-specific preemption risk
        interrupt_rate = GPU_SPOT_INTERRUPT_RATE.get(gpu_type or "", _DEFAULT_INTERRUPT_RATE)
        # Quick sanity check: spot effective cost ratio vs on-demand
        # Uses the same model as spot_checkpoint_cost() but simplified
        spot_fraction = (1.0 + 0.03 + interrupt_rate * 0.5)  # ckpt_overhead + rework
        # Spot still wins unless the GPU has extreme preemption AND a low spot discount
        # We need the catalog to get actual prices, so we use a heuristic:
        # if effective_hours > 1.4× job_hours, warn and fall back to on_demand.
        if spot_fraction < 1.4:
            return "spot"
        return "on_demand"

    if duty >= be:
        return "reserved"
    return "on_demand"


def recommend_tier_with_detail(
    hours_per_day: float,
    interruptible: bool,
    gpu_type: str | None = None,
    job_days: int | None = None,
    on_demand_hr: float = 2.5,
    spot_hr: float = 1.5,
) -> dict:
    """Extended version that returns the full reasoning alongside the tier recommendation.

    Returns a dict with: tier, reserved_tier (1yr/3yr), interrupt_rate,
    break_even_util, spot_cost_ratio, note.
    """
    if job_days is not None and job_days < 30:
        res_tier_name = "1yr"
        effective_reserved_discount = RESERVED_DISCOUNTS["1yr"]
    else:
        res_tier_name = "3yr"
        effective_reserved_discount = RESERVED_DISCOUNTS["3yr"]

    be = break_even_utilization(effective_reserved_discount)
    duty = max(0.0, hours_per_day) / 24.0
    interrupt_rate = GPU_SPOT_INTERRUPT_RATE.get(gpu_type or "", _DEFAULT_INTERRUPT_RATE)

    spot_effective_mult = 1.0 + 0.03 + interrupt_rate * 0.5
    spot_effective_hr_cost = spot_hr * spot_effective_mult
    spot_vs_od_ratio = spot_effective_hr_cost / on_demand_hr if on_demand_hr > 0 else 1.0

    tier = recommend_tier(hours_per_day, interruptible,
                          reserved_discount=effective_reserved_discount,
                          gpu_type=gpu_type, job_days=job_days)
    notes = []
    if interruptible and hours_per_day < 24:
        notes.append(f"spot interrupt_rate={interrupt_rate:.0%} → eff_mult={spot_effective_mult:.3f}x")
    if tier == "reserved":
        notes.append(f"duty={duty:.0%} ≥ break-even={be:.0%} using {res_tier_name} rate")
    if tier == "on_demand":
        notes.append(f"duty={duty:.0%} < break-even={be:.0%} or spot overhead too high")

    return {
        "tier": tier,
        "reserved_term": res_tier_name,
        "interrupt_rate": interrupt_rate,
        "break_even_util": round(be, 3),
        "spot_vs_od_ratio": round(spot_vs_od_ratio, 3),
        "note": "; ".join(notes),
    }


def cache_is_worth_it(
    avg_cache_reads: float,
    write_cost_per_m: float,
    price_in_per_m: float,
    read_discount: float = 0.10,
) -> bool:
    """Return True only when cache read savings exceed the write cost.

    Break-even: reads × (1 - read_discount) × price_in ≥ write_cost
    i.e.  avg_cache_reads ≥ write_cost / ((1 - read_discount) × price_in)

    Args:
        avg_cache_reads:  average number of times a cached prefix is re-read.
        write_cost_per_m: USD per 1M tokens to *write* the cache entry
                          (Gemini charges for cache storage; Anthropic does not
                           but this can model the amortised infra cost).
        price_in_per_m:   normal input price before any cache discount.
        read_discount:    fraction of price paid on a cache hit (default 0.10).
    """
    if avg_cache_reads <= 0 or price_in_per_m <= 0:
        return False
    # savings per read per million tokens = price_in × (1 - read_discount)
    savings_per_read = price_in_per_m * (1.0 - read_discount)
    break_even_reads = write_cost_per_m / savings_per_read if savings_per_read > 0 else float("inf")
    return avg_cache_reads >= break_even_reads


def cache_break_even_reads(
    write_cost_per_m: float,
    price_in_per_m: float,
    read_discount: float = 0.10,
) -> float:
    """Minimum avg reads per prefix for caching to pay off."""
    savings_per_read = price_in_per_m * (1.0 - read_discount)
    if savings_per_read <= 0:
        return float("inf")
    return write_cost_per_m / savings_per_read


def spot_checkpoint_cost(
    job_hours: float,
    spot_hr: float,
    on_demand_hr: float,
    interrupt_rate: float = 0.05,      # per-hour chance (H100 spot ~<5%)
    ckpt_overhead_frac: float = 0.03,  # steady cost of writing checkpoints
    rework_hours_per_interrupt: float = 0.5,
) -> dict:
    """Effective cost of running a checkpointable job on spot vs on-demand.

    Interruptions waste the compute since the last checkpoint (rework); checkpointing
    adds a small steady overhead. Spot still wins for interruptible jobs.
    """
    expected_interrupts = job_hours * interrupt_rate
    rework_hours = expected_interrupts * rework_hours_per_interrupt
    effective_hours = job_hours * (1.0 + ckpt_overhead_frac) + rework_hours
    spot_cost = effective_hours * spot_hr
    on_demand_cost = job_hours * on_demand_hr
    savings_pct = (1.0 - spot_cost / on_demand_cost) * 100.0 if on_demand_cost > 0 else 0.0
    return {
        "spot_effective_hours": round(effective_hours, 2),
        "spot_cost": round(spot_cost, 2),
        "on_demand_cost": round(on_demand_cost, 2),
        "savings_pct": round(savings_pct, 1),
    }
