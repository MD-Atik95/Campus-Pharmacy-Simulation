# Pharmacy Inventory Simulation: Best (s, S) Reorder Policy

**Course:** CSE 413 - Simulation & Modeling Lab

## Problem
A pharmacy sells paracetamol strips. Daily demand is random (Poisson, mean 20) and an order takes
1 to 3 days to arrive (equally likely). Under an **(s, S) policy** the shop checks its inventory
position (stock on hand + stock on order) every evening. If the position is **at or below s**, it
orders enough to bring the position **up to S**. Unmet demand is lost.

## Model parameters
| Parameter | Value |
|---|---|
| Daily demand | Poisson(20) |
| Lead time | 1, 2 or 3 days |
| Order cost | 300 BDT per order |
| Holding cost | 0.5 BDT per strip per day |
| Shortage cost | 8 BDT per lost strip |
| Run length | 365 days (+30 days warm-up) |
| Replications | 100 per policy |

## Method
1. **LCG** random number generator (a = 1664525, c = 1013904223, m = 2^32).
2. **Chi-square test** to check that the LCG numbers are uniform.
3. **Inverse transform** method for Poisson demand; discrete uniform for lead time.
4. **Simulator** with daily steps: receive orders, sell, review position, order, add costs.
5. **Policy search** over 49 (s, S) pairs (s = 20..80, S = 100..300) with common random numbers.
6. **Heatmap** of average cost and service level plot.
7. **t-tests**: Welch two-sample, paired, and one-sample (benchmark 85 BDT/day).
8. **Daily dataset** (CSV) of one year under the best policy.

## Files
- `inventory_simulation.py` - full simulation code
- `inventory_daily_data.csv` - 365 days of the best policy (seed 42)
- `inventory_results.png` - heatmap, service level, cost comparison, stock level
- `Inventory_Simulation_Colab.ipynb` - the original Colab notebook (includes the animation)

## How to run
```
pip install numpy scipy matplotlib seaborn
python inventory_simulation.py
```
It takes about 1 to 2 minutes and creates the CSV and PNG in the same folder.

## Results
- **Best policy: (s, S) = (30, 175)**, about **80.7 BDT/day**, service level about 95.4%.
- Worst policy: (80, 100), about 231.4 BDT/day. The best policy saves about 65% (around 55,000 BDT per year).
- The best policy is significantly cheaper than the cautious policy (60, 200): mean difference
  about -7.3 BDT/day (Welch p < 0.001).
- Its mean cost is significantly below the 85 BDT/day benchmark.
- Very small S (100) is costly because of frequent stock-outs. A very high s with large S gives high holding cost.
  Several nearby policies, for example (30, 200) and (20, 175), cost almost the same.

## CSV columns
`day, start_inventory, demand, sales, lost_sales, end_inventory, order_qty`
