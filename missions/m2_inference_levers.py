"""M2 — Inference Cost Levers: $/1M-token, batch x cache x cascade (deck §7).
Extension 3: cache_is_worth_it() — only count cache savings when reads ≥ break-even.
Extension 4: Reasoning budget — split is_reasoning=1 traffic, propose 10% cap.

Run: python missions/m2_inference_levers.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from missions._common import load_csv, num
from finops import pricing, sustainability

# $/1M tokens (input, output) — illustrative 2026.
MODEL_PRICES = {"small": (0.20, 0.40), "large": (3.00, 15.00)}

# ── Extension 3 assumptions ────────────────────────────────────────────────────
# Anthropic/Gemini charge a "cache write" cost to store a prefix the first time.
# Anthropic: cache write ~1.25× the normal input price; Gemini charges separately
# for storage.  We use 1.25× as a conservative estimate since token_usage.csv
# doesn't distinguish providers.  This is a stated assumption.
CACHE_WRITE_MULTIPLIER = 1.25   # write cost = 1.25 × price_in_per_m  (assumption)
# Average reads per cached prefix — not in CSV, so we model two scenarios:
AVG_READS_LOW = 1.5    # pessimistic: many one-off prompts
AVG_READS_HIGH = 8.0   # optimistic: shared system prompt read many times

# ── Extension 4 assumptions ────────────────────────────────────────────────────
REASONING_CAP_PCT = 0.10   # proposed cap: at most 10% of requests use reasoning


def run(verbose: bool = True) -> dict:
    rows = load_csv("token_usage.csv")
    base_cost = opt_cost = 0.0
    total_tokens = 0

    # Extension 3: break-even reads per model tier
    be_reads = {}
    for tier, (pin, pout) in MODEL_PRICES.items():
        write_cost_pm = pin * CACHE_WRITE_MULTIPLIER
        be_reads[tier] = pricing.cache_break_even_reads(write_cost_pm, pin)

    # Extension 4 accumulators
    reason_cost = noreason_cost = 0.0
    reason_tok = noreason_tok = 0
    reason_wh = noreason_wh = 0.0
    reason_count = noreason_count = 0

    for r in rows:
        inp, out = int(num(r["input_tokens"])), int(num(r["output_tokens"]))
        cached = int(num(r["cached_input_tokens"]))
        is_batch = bool(int(num(r["is_batch"])))
        is_reasoning = bool(int(num(r.get("is_reasoning", 0))))
        total_tokens += inp + out

        # BASELINE: naive — large model, no cache, no batch
        lin, lout = MODEL_PRICES["large"]
        base_cost += pricing.request_cost(inp, out, lin, lout)

        # OPTIMIZED: cascade + caching (only if worth it) + batch
        tier = r["route_tier"]
        pin, pout = MODEL_PRICES[tier]
        write_cost_pm = pin * CACHE_WRITE_MULTIPLIER

        # Extension 3: only apply cached_in discount if cache is worth it
        # We use AVG_READS_HIGH as the expected-case scenario (shared system prompts).
        # The analysis section below shows both scenarios.
        use_cache = cached > 0 and pricing.cache_is_worth_it(
            AVG_READS_HIGH, write_cost_pm, pin
        )
        eff_cached = cached if use_cache else 0

        opt_cost += pricing.request_cost(inp, out, pin, pout,
                                         cached_in=eff_cached, batch=is_batch)

        # Extension 4: tách reasoning vs non-reasoning
        tok_total = inp + out
        wh = sustainability.wh_per_query(tok_total, is_reasoning=is_reasoning)
        req_cost_val = pricing.request_cost(inp, out, pin, pout,
                                            cached_in=eff_cached, batch=is_batch)
        if is_reasoning:
            reason_cost += req_cost_val
            reason_tok += tok_total
            reason_wh += wh
            reason_count += 1
        else:
            noreason_cost += req_cost_val
            noreason_tok += tok_total
            noreason_wh += wh
            noreason_count += 1

    base_pm = pricing.dollars_per_million(base_cost, total_tokens)
    opt_pm = pricing.dollars_per_million(opt_cost, total_tokens)
    savings_pct = (1 - opt_cost / base_cost) * 100 if base_cost else 0.0

    # Extension 4: reasoning cap simulation
    total_count = len(rows)
    current_reason_pct = reason_count / total_count if total_count else 0.0
    if current_reason_pct > REASONING_CAP_PCT:
        # requests to shift from reasoning → non-reasoning
        excess_count = reason_count - int(total_count * REASONING_CAP_PCT)
        # cost/Wh per reasoning request on average
        avg_reason_cost = reason_cost / reason_count if reason_count else 0.0
        avg_reason_wh = reason_wh / reason_count if reason_count else 0.0
        # assume shifted requests become normal (non-reasoning) at same token count
        avg_noreason_cost = noreason_cost / noreason_count if noreason_count else 0.0
        avg_noreason_wh = noreason_wh / noreason_count if noreason_count else 0.0
        cap_cost_saving = excess_count * (avg_reason_cost - avg_noreason_cost)
        cap_wh_saving = excess_count * (avg_reason_wh - avg_noreason_wh)
    else:
        excess_count = 0
        cap_cost_saving = 0.0
        cap_wh_saving = 0.0

    if verbose:
        print("== M2 Inference Cost Levers (Ext 3: cache worth-it gate; Ext 4: reasoning budget) ==")
        print(f"requests={len(rows)}  tokens={total_tokens:,}")
        print(f"baseline  : ${base_cost:,.2f}/day   ${base_pm:.3f}/1M-token")
        print(f"optimized : ${opt_cost:,.2f}/day   ${opt_pm:.3f}/1M-token")
        print(f"savings   : {savings_pct:.1f}%  (cascade + caching + batch)")
        print(f"discount stack (batch + 100% cache): {pricing.discount_stack(batch=True, cache_hit_frac=1.0):.3f} of naive")

        print(f"\n── Extension 3: cache_is_worth_it() analysis ──")
        print(f"  Cache write cost assumed = {CACHE_WRITE_MULTIPLIER}× input price (stated assumption)")
        for tier, (pin, _) in MODEL_PRICES.items():
            write_pm = pin * CACHE_WRITE_MULTIPLIER
            be = be_reads[tier]
            print(f"  [{tier:6}] price_in=${pin:.2f}/1M  write=${write_pm:.3f}/1M  "
                  f"break-even={be:.2f} reads/prefix")
        print(f"  Scenario LOW  ({AVG_READS_LOW} reads/prefix):")
        for tier, (pin, _) in MODEL_PRICES.items():
            worth = pricing.cache_is_worth_it(AVG_READS_LOW, pin * CACHE_WRITE_MULTIPLIER, pin)
            print(f"    [{tier:6}] cache_is_worth_it → {worth}")
        print(f"  Scenario HIGH ({AVG_READS_HIGH} reads/prefix):")
        for tier, (pin, _) in MODEL_PRICES.items():
            worth = pricing.cache_is_worth_it(AVG_READS_HIGH, pin * CACHE_WRITE_MULTIPLIER, pin)
            print(f"    [{tier:6}] cache_is_worth_it → {worth}")
        print(f"  (Using HIGH scenario in M2 optimized cost above)")

        print(f"\n── Extension 4: Reasoning budget ──")
        print(f"  {'group':12} {'requests':>9} {'%traffic':>9} {'$/day':>10} {'%cost':>7} {'Wh/day':>10}")
        total_cost = reason_cost + noreason_cost
        total_wh = reason_wh + noreason_wh
        print(f"  {'reasoning':12} {reason_count:>9,} {current_reason_pct:>9.1%} "
              f"{reason_cost:>10.2f} {reason_cost/total_cost*100 if total_cost else 0:>6.1f}% "
              f"{reason_wh:>10.2f}")
        print(f"  {'normal':12} {noreason_count:>9,} {1-current_reason_pct:>9.1%} "
              f"{noreason_cost:>10.2f} {noreason_cost/total_cost*100 if total_cost else 0:>6.1f}% "
              f"{noreason_wh:>10.2f}")
        print(f"  Why reasoning costs ~{sustainability.REASONING_ENERGY_MULTIPLIER:.0f}× energy: "
              f"reasoning models run many internal chain-of-thought steps before")
        print(f"  producing output — typically 10-20× more tokens than a standard query,")
        print(f"  each token requiring the same memory bandwidth × compute.")
        if current_reason_pct > REASONING_CAP_PCT:
            print(f"\n  Current reasoning traffic ({current_reason_pct:.1%}) > cap ({REASONING_CAP_PCT:.0%})")
            print(f"  Shifting {excess_count} requests/day from reasoning → normal:")
            print(f"    Cost saving : ${cap_cost_saving:,.2f}/day  (${cap_cost_saving*30:,.0f}/month)")
            print(f"    Energy saving: {cap_wh_saving:,.1f} Wh/day  ({cap_wh_saving*30/1000:.1f} kWh/month)")
        else:
            print(f"\n  Traffic already at/below {REASONING_CAP_PCT:.0%} cap → savings = $0 / 0 Wh")

    return {
        "baseline_daily": round(base_cost, 2), "optimized_daily": round(opt_cost, 2),
        "baseline_per_m": round(base_pm, 3), "optimized_per_m": round(opt_pm, 3),
        "savings_pct": round(savings_pct, 1), "total_tokens": total_tokens,
        # Extension 3
        "cache_break_even_reads": {k: round(v, 2) for k, v in be_reads.items()},
        # Extension 4
        "reasoning": {
            "count": reason_count, "pct_traffic": round(current_reason_pct, 3),
            "cost_daily": round(reason_cost, 2), "wh_daily": round(reason_wh, 2),
        },
        "cap_cost_saving_daily": round(cap_cost_saving, 2),
        "cap_wh_saving_daily": round(cap_wh_saving, 2),
    }


if __name__ == "__main__":
    run()
