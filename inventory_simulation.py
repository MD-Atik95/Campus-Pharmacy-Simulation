"""
Simulation and Modeling Lab Project
Campus Pharmacy Inventory Simulation: finding the best (s, S) reorder policy
Monte Carlo / discrete-time simulation with random demand and random lead time.

Modules: RNG, Variate Generators, Inventory Simulator, Replication/Statistics,
         Policy Search (cost analysis), Dataset export, Visualization, Main
"""
import csv
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = ""   # files are saved in the folder where you run the program

# ---------------- Model parameters ----------------
DEMAND_MEAN = 20.0          # strips/day, Poisson
LEAD_TIMES = (1, 2, 3)      # days, equally likely
ORDER_COST = 300.0          # BDT per order (fixed)
HOLD_COST = 0.5             # BDT per strip per day
SHORT_COST = 8.0            # BDT per strip of lost sale
HORIZON = 365               # days simulated per run
WARMUP = 30                 # days discarded (start-up effect)
REPS = 100                  # replications per policy


# ---------------- Module 1: RNG (Linear Congruential Generator) ----------------
class LCG:
    def __init__(self, seed=12345, a=1664525, c=1013904223, m=2**32):
        self.x, self.a, self.c, self.m = seed, a, c, m

    def next(self):
        self.x = (self.a * self.x + self.c) % self.m
        return self.x / self.m


def chi_square_uniformity(rng, n=10000, k=10):
    """Critical value (df = 9, alpha = 0.05) = 16.919"""
    counts = [0] * k
    for _ in range(n):
        counts[min(int(rng.next() * k), k - 1)] += 1
    e = n / k
    chi2 = sum((o - e) ** 2 / e for o in counts)
    return chi2, chi2 < 16.919


# ---------------- Module 2: Variate generators ----------------
def poisson(rng, mean):
    """Inverse transform for Poisson: smallest d with F(d) >= U."""
    u = rng.next()
    p = math.exp(-mean)
    cdf = p
    d = 0
    while u > cdf:
        d += 1
        p *= mean / d
        cdf += p
    return d


def lead_time(rng):
    return LEAD_TIMES[min(int(rng.next() * len(LEAD_TIMES)), len(LEAD_TIMES) - 1)]


# ---------------- Module 3: Inventory simulator ----------------
def simulate(s, S, seed, horizon=HORIZON, warmup=WARMUP, log=None):
    """One run of an (s, S) policy with daily review and lost sales.
    Two separate random streams (demand, lead time) allow common random numbers
    when different policies are compared."""
    dem_rng, lt_rng = LCG(seed), LCG(seed + 7919)
    on_hand = S
    pipeline = []                      # outstanding orders: [arrival_day, qty]
    tot = dict(order=0.0, hold=0.0, short=0.0, orders=0, demand=0, lost=0, inv=0.0)

    for day in range(1, horizon + warmup + 1):
        # 1) receive orders arriving today
        for o in [o for o in pipeline if o[0] == day]:
            on_hand += o[1]
        pipeline = [o for o in pipeline if o[0] > day]
        start_inv = on_hand
        # 2) random demand, lost sales if stock is short
        d = poisson(dem_rng, DEMAND_MEAN)
        sold = min(on_hand, d)
        lost = d - sold
        on_hand -= sold
        # 3) review: order up to S when inventory position <= s
        position = on_hand + sum(o[1] for o in pipeline)
        qty = 0
        if position <= s:
            qty = S - position
            pipeline.append([day + lead_time(lt_rng), qty])
        # 4) costs (only after warm-up)
        if day > warmup:
            tot["order"] += ORDER_COST if qty else 0
            tot["hold"] += HOLD_COST * on_hand
            tot["short"] += SHORT_COST * lost
            tot["orders"] += 1 if qty else 0
            tot["demand"] += d
            tot["lost"] += lost
            tot["inv"] += on_hand
            if log is not None:
                log.append((day - warmup, start_inv, d, sold, lost, on_hand, qty))

    total = tot["order"] + tot["hold"] + tot["short"]
    return {
        "cost_day": total / horizon,
        "order_day": tot["order"] / horizon,
        "hold_day": tot["hold"] / horizon,
        "short_day": tot["short"] / horizon,
        "service": 1 - tot["lost"] / tot["demand"],
        "avg_inv": tot["inv"] / horizon,
        "orders": tot["orders"],
        "mean_demand": tot["demand"] / horizon,
    }


# ---------------- Module 4: Replication statistics ----------------
def replicate(s, S, reps=REPS):
    runs = [simulate(s, S, seed=1000 + r) for r in range(reps)]   # same seeds for all policies (CRN)
    out = {}
    for k in runs[0]:
        vals = [r[k] for r in runs]
        m = sum(vals) / reps
        sd = math.sqrt(sum((v - m) ** 2 for v in vals) / (reps - 1))
        out[k] = m
        out[k + "_ci"] = 1.96 * sd / math.sqrt(reps)
    return out


# ---------------- Module 5: Policy search (cost analysis) ----------------
def search(s_values, S_values):
    results = {}
    for s in s_values:
        for S in S_values:
            if S > s:
                results[(s, S)] = replicate(s, S)
    best = min(results, key=lambda k: results[k]["cost_day"])
    return results, best


# ---------------- Module 6: Dataset export ----------------
def export_dataset(s, S, path):
    log = []
    simulate(s, S, seed=42, log=log)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["day", "start_inventory", "demand", "sales", "lost_sales",
                    "end_inventory", "order_qty"])
        w.writerows(log)
    return log


# ---------------- Module 7: Visualization ----------------
def plot_results(results, best, s_values, S_values, log):
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
    # (a) cost heatmap
    grid = [[results[(s, S)]["cost_day"] if (s, S) in results else float("nan")
             for S in S_values] for s in s_values]
    im = ax[0].imshow(grid, cmap="viridis_r", aspect="auto", origin="lower")
    ax[0].set_xticks(range(len(S_values)), S_values)
    ax[0].set_yticks(range(len(s_values)), s_values)
    ax[0].set(title="Average cost per day (BDT)", xlabel="S (order-up-to level)", ylabel="s (reorder point)")
    for i, s in enumerate(s_values):
        for j, S in enumerate(S_values):
            if (s, S) in results:
                ax[0].text(j, i, f"{results[(s, S)]['cost_day']:.0f}", ha="center", va="center",
                           color="white", fontsize=8)
    ax[0].plot(S_values.index(best[1]), s_values.index(best[0]), "r*", markersize=16)
    fig.colorbar(im, ax=ax[0])
    # (b) service level vs s for best S
    Sb = best[1]
    ss = [s for s in s_values if (s, Sb) in results]
    ax[1].errorbar(ss, [results[(s, Sb)]["service"] * 100 for s in ss],
                   yerr=[results[(s, Sb)]["service_ci"] * 100 for s in ss], fmt="o-", capsize=3)
    ax[1].set(title=f"Service level vs s (S = {Sb})", xlabel="Reorder point s", ylabel="% of demand met")
    ax[1].grid(alpha=0.3)
    # (c) inventory trajectory
    days = [r[0] for r in log][:90]
    ax[2].plot(days, [r[5] for r in log][:90], label="End-of-day stock")
    ax[2].axhline(best[0], color="orange", ls="--", label=f"s = {best[0]}")
    ax[2].axhline(best[1], color="red", ls="--", label=f"S = {best[1]}")
    ax[2].set(title="Stock level, first 90 days (best policy)", xlabel="Day", ylabel="Strips")
    ax[2].legend(); ax[2].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT + "inventory_results.png", dpi=150)


# ---------------- Module 8: Main / Controller ----------------
def main():
    chi2, ok = chi_square_uniformity(LCG(2024))
    print(f"RNG Chi-square = {chi2:.2f} (critical 16.919) -> {'PASS' if ok else 'FAIL'}")

    # Validation of the demand generator
    r = LCG(99)
    d = [poisson(r, DEMAND_MEAN) for _ in range(100000)]
    mu = sum(d) / len(d)
    var = sum((x - mu) ** 2 for x in d) / (len(d) - 1)
    print(f"Poisson check: mean = {mu:.3f} (target {DEMAND_MEAN}), variance = {var:.3f} (target {DEMAND_MEAN})")

    eoq = math.sqrt(2 * ORDER_COST * DEMAND_MEAN / HOLD_COST)
    print(f"EOQ order quantity (rough guide) = {eoq:.0f} strips\n")

    s_values = [20, 30, 40, 50, 60, 70, 80]
    S_values = [100, 150, 175, 200, 225, 250, 300]
    results, best = search(s_values, S_values)

    print(f"{'(s,S)':>10} | {'Cost/day':>9} {'+-95%CI':>8} | {'Order':>6} {'Hold':>6} {'Short':>6} | {'Service%':>8} | {'AvgInv':>6}")
    for k in sorted(results, key=lambda k: results[k]["cost_day"])[:8]:
        v = results[k]
        print(f"{str(k):>10} | {v['cost_day']:9.1f} {v['cost_day_ci']:8.1f} | {v['order_day']:6.1f} "
              f"{v['hold_day']:6.1f} {v['short_day']:6.1f} | {v['service']*100:8.2f} | {v['avg_inv']:6.1f}")
    worst = max(results, key=lambda k: results[k]["cost_day"])
    print(f"\nBest policy  (s,S) = {best}: {results[best]['cost_day']:.1f} BDT/day")
    print(f"Worst policy (s,S) = {worst}: {results[worst]['cost_day']:.1f} BDT/day")
    saving = (1 - results[best]['cost_day'] / results[worst]['cost_day']) * 100
    print(f"Saving of best vs worst = {saving:.1f}% ; yearly = {(results[worst]['cost_day']-results[best]['cost_day'])*365:.0f} BDT")

    log = export_dataset(best[0], best[1], OUT + "inventory_daily_data.csv")
    plot_results(results, best, s_values, S_values, log)

    # Descriptive statistics of the dataset run
    dem = [r_[2] for r_ in log]
    lost = [r_[4] for r_ in log]
    print(f"\nDataset run (seed 42, best policy): days={len(log)}, mean demand={sum(dem)/len(dem):.2f}, "
          f"max demand={max(dem)}, stock-out days={sum(1 for x in lost if x>0)}, "
          f"orders placed={sum(1 for r_ in log if r_[6]>0)}")


if __name__ == "__main__":
    main()
