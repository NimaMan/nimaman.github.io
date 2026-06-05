---
author: Nima Manaf, PhD
title: "One Recipe for Ten Inventory Problems: Evolution Strategies for Inventory Control"
date: 2026-06-05
description: "A single gradient-free recipe — CMA-ES optimizing small, interpretable policies — learns competitive inventory-control policies across ten classical problems."
math: true
---

# Summary

- One generic recipe — **Covariance Matrix Adaptation Evolution Strategies (CMA-ES)** optimizing small, interpretable policies — learns competitive inventory-control policies across **ten classical problems**, with little tuning.
- The learned policies are tiny: **tens to a few hundred parameters**, orders of magnitude smaller than the deep networks usually reported, and small enough to read.
- The recurring trick is that the **action parameterization is part of the policy**: the decoder is shaped like the relevant heuristic's coordinate system, which is what lets such a small policy reach the good operating region.

This post started life as a tutorial on the lost-sales problem alone. Since then the study has broadened: the same loop now runs, essentially unchanged, across ten structurally different inventory problems. I lead with lost sales — still the most developed example — and then widen out.

# The setup: a sequential decision under uncertainty

Inventory management is a textbook sequential decision problem. Each period a controller decides how much to order, balancing the cost of holding stock against the cost of shortages. These problems are naturally Markov decision processes (MDPs), but exact dynamic programming is only tractable for small instances — the state space grows with the lead time and the number of stocking locations (the curse of dimensionality). So the field leans on two things:

- **Problem-specific heuristics** — base-stock, $(s,S)$, $(s,nQ)$, dual-index, capped dual-index. These are strong, but each one takes specialized analysis and simplifying assumptions that limit where it applies.
- **Deep reinforcement learning (DRL)** — represent the policy as a neural network and tune it from simulated experience. This is generic, but it demands a lot of compute and hyperparameter tuning, and the networks (thousands to millions of parameters) are hard to interpret.

The question I want to ask here is different from "what is the best method for problem X". It is: *can one lightweight, portable recipe stand in for bespoke heuristics across many different inventory problems, while staying small enough to interpret and cheap enough to retrain?*

The recipe is to optimize the parameters of a compact policy with **CMA-ES**, a gradient-free, population-based optimizer. Three things make it a good fit, and they hold across all ten problems:

1. It optimizes the **realized objective** — the simulated long-run average cost — directly. No value-function bootstrapping, no reward shaping, no credit assignment to individual actions. The same loop applies whether the action is a scalar order, a regular/expedite pair, or a pair of echelon order-up-to levels.
2. Its built-in **covariance adaptation supplies exploration automatically**, so there is no exploration strategy to design.
3. Because it never differentiates the policy, it happily handles the **non-smooth, discretized, clipped action maps** that are awkward for backpropagation but completely natural for inventory controls.

> A note on continuity from the original post: the earlier version pitched ES against RL specifically for lost sales. The thesis here is the same in spirit but bigger in scope — it is the *portability* of one recipe across ten problems that is the point, not a single head-to-head on one problem.

## The training loop

CMA-ES is treated as a black-box optimizer. Each generation does three things: it samples a **population** of policy-parameter vectors from a Gaussian search distribution; it **evaluates** each one by short rollouts in the simulator (the only step that touches the environment), giving a simulated long-run cost; and it **ranks and updates** the search distribution from the best members. We use a single fixed configuration throughout (population 64, a short 2000-period training horizon) and do *not* tune it per problem.

```python
es = CMAES(num_params=num_params, popsize=64)
for generation in range(num_generations):
    candidates = es.ask()                      # sample a population of policies
    costs = [rollout_cost(decode(c)) for c in candidates]  # evaluate each (parallelizable)
    es.tell(costs)                             # update mean, step size, covariance
```

The evaluation step is embarrassingly parallel — one of ES's structural advantages over RL — so it scales cleanly across cores or a cluster.

# The key idea: the action parameterization *is* the policy

This is the one idea worth taking away, and it runs through every problem.

A learner that can only emit a raw scalar order is implicitly restricted to whatever that scalar can express. A learner whose **decoder is shaped like the relevant structured heuristic** searches a far more useful action geometry. So I treat each policy as a single map from the problem's **state** to a **valid action**, and everything in between — a divide-by-scale normalization, a small backbone (a linear map, a one-hidden-layer network, or a soft decision tree), and crucially a **decoder / action geometry** — is owned by the policy and learned by CMA-ES.

Where a structured heuristic exists, the decoder lives in *that heuristic's coordinate system*:

- an **ordinal "one more unit" quantity** for lost sales,
- **capped-dual-index coordinates** for dual sourcing,
- **direct echelon order-up-to levels** for multi-echelon.

The payoff is that the decoder, not the network width, drives several of the results. The clearest example is multi-echelon (below): with the same tree, optimizer and horizon, changing only the decoder's reachable action set swings the policy from ~14% *better* than the best base-stock to more than 200% *worse*.

# Lost sales (the original problem)

The lost-sales problem is one of the fundamental problems in inventory theory. Single item, periodic review, integer demand and orders. Each period: an order placed $L$ periods ago arrives and joins on-hand stock; the controller places a new integer order $q_t$ that will arrive after lead time $L$; demand $D_t$ is realized and served from stock, and any **unmet demand is lost** rather than backordered. A holding cost $h$ is charged per leftover unit and a penalty $p$ per lost unit:

$$
c_t = h\,(I_t - D_t)^{+} + p\,(D_t - I_t)^{+}.
$$

The state is the on-hand inventory plus the pipeline of outstanding orders, an $L$-dimensional vector — and the state space grows exponentially in the lead time, which is exactly why exact solution is intractable and the problem is famously hard.

The action here is a single integer order, so this is where the decoder family lives. We use three policy-owned decoders: a **soft-gated direct quantity** (a logistic gate decides *whether* to order, a softplus head decides *how much*), a **soft-gated ordinal quantity** (a sum of soft "one more unit" indicators), and **soft decision trees** with linear leaves.

We benchmark on a surface, not a single instance: holding cost 1, mean demand 5, lead times $L\in\{4,6,8,10\}$, lost-sales penalties $p\in\{4,19\}$, and three mean-preserving demand families (Poisson, Geometric, and a positively-correlated Markov-modulated Poisson). That is **24 vanilla instances**. The comparators are the strong classical heuristics: Myopic-1, Myopic-2, and standard vector base-stock.

**Result.** Across the 24-instance vanilla surface, the learned policies are **instance-best in 22 of 24 cases**. The soft tree is the most frequent winner. The two cases still won by a classical baseline are the high-penalty, autocorrelated MMPP instances at the longest lead times ($L=8,10$) — a regime-switching, deep-pipeline combination that is the hardest setting for a single stationary compact policy.

<center>
<img class="special-img-class" style="width:60%" src="/static/images/Lost_sales_p_4_l_4.jpg" label="lead_time_4"/>
</center>

## Fixed-cost lost sales

Add a fixed setup cost $K$ charged whenever a strictly positive order is placed, and the *order/no-order* decision becomes an explicit part of the problem — it is no longer optimal to order every period. The comparators become the $(s,S)$, $(s,nQ)$, and modified $(s,S,q)$ policies; the surface grows to **48 instances** with $K\in\{5,25\}$.

**Result.** Learned policies are **instance-best in 47 of 48 instances**. And the architecture story sharpens: once a setup charge makes the order/no-order boundary dominant, the **gated** decoders — which factor the decision into an explicit gate (*whether*) and a quantity head (*how much*) — win most instances, inverting the vanilla preference. There is an honest failure mode visible right in the table: at high setup cost a couple of the decoders collapse onto a degenerate "never order" policy; the gated ordinal decoder is the one that consistently escapes it. The decoder, not the optimizer, governs that boundary.

# Broadening out: the other eight problems

The same loop, the same optimizer configuration, the same idea — only the decoder and the action geometry change.

## Dual sourcing — *matches* the proven optimum

Two supply modes (a cheap slow regular source and an expensive fast expedited one). On the six small benchmark instances of Gijsbrechts et al. (2022), the strongest structured policy is the **capped dual-index (CDI)** heuristic, whose published optimality gap is $\le 0.11\%$ — so it is effectively an optimal proxy.

By putting the learned soft tree in **capped-dual-index coordinates** and warm-starting CMA-ES at the CDI solution, the learned policy **matches CDI on all six instances** (four at the discrete-grid rounding floor, two negligibly below by 0.009% and 0.041%, well inside CDI's own optimality band). I report these as **matches, not improvements** — you cannot meaningfully beat a near-optimal proxy by hundredths of a percent. As an *indicative* reference, the published A3C learner sits at a 0.51–1.85% gap, i.e. outside the band the learned policy lands in. The lever here is the action geometry: a raw direct-order decoder cannot express CDI and does not reach this level.

## Divergent multi-echelon with special delivery — ~14.4% over the best base-stock

One warehouse, $R$ retailers, with a special-delivery option. The action is a warehouse order plus a shared retailer order-up-to level. This is the sharpest test of the action-geometry principle, because the wrong geometry is fatal: the cost-minimizing warehouse base-stock is roughly 300–525, while the reduced action grid used in prior work caps the warehouse level at 100 — so the grid *physically cannot reach the operating region*.

**Result.** A **direct-level** soft tree (leaves estimate the order-up-to levels directly, bounded only by physical caps) improves on the **best in-environment constant base-stock by ≈14.4%** on both reported settings. The grid-action policy — same tree, same optimizer, only the reachable level set changed — stays ~230% *above* the benchmark. That contrast is the whole point.

One honesty caveat I keep explicit: the ~14.4% figure is measured against the **best in-environment constant base-stock** under one cost convention, whereas the published A3C improvements (8.95%, 12.09%) are against a different baseline under a different cost convention. So the A3C comparison is **indicative of the direct design's strength, not a strictly like-for-like ranking**.

## Perishable inventory — beats the best base-stock gate

Stock ages and expires; a waste cost is charged on outdated units. The benchmark instances issue either FIFO or LIFO. A 21-parameter age-dependent soft tree **beats the best base-stock gate** under a shared common-random-number estimator: **+1.16%** under FIFO and **+0.82%** under LIFO, both several times their paired standard error. The tree exploits the age structure a single base-stock cannot — ordering less when older stock is already on hand. (Both learned returns happen to sit essentially *on* the analytic value-iteration optimum, but that comparison mixes two estimators, so I treat it as corroborating context, not a second win.)

## General-network backorder — beats the published benchmark by over 20%

A four-supplier, four-warehouse, five-retailer network with backordered (not lost) demand. Put the policy in the **node-base-stock-targets** coordinate system and let a state-dependent tree modulate the targets, and it reduces long-run average cost by **22.4%** (and 26.7% on a second seed) over the **reproduced constant node-base-stock benchmark**, on the same environment under a paired comparison.

Two honest caveats. First, despite the family name, this verified environment charges holding and backorder cost only — no fixed ordering cost. Second, the paper's PPO best (8,714) is a **cross-protocol** figure produced by a different learner under its own protocol. Our learned policy lands below it, but that is **not a head-to-head PPO beat and I do not claim one** — the defensible claim is the paired, same-environment improvement over the published constant base-stock benchmark.

## Serial multi-echelon (Clark–Scarf) — *matches* the proven optimum

A 3-stage serial system whose optimal policy is known exactly (Clark–Scarf echelon base-stock). Because the decoder lives in the optimum's own coordinate system, the warm-started direct-level soft tree **reproduces the proven optimum** to within +0.011% — inside the environment's own ~0.06% reproduction band, statistically indistinguishable from optimal. This is a **match, not a win**: one cannot improve on a true optimum, and I do not claim to. What it shows is that the same generic recipe *recovers the optimal policy* on a serial system.

## One-warehouse multi-retailer — beats the tuned gate, but below published PPO

Asymmetric, high-variability OWMR instances from Kaynov et al. (2024). The like-for-like comparator is a strong in-repo tuned base-stock-plus-allocation gate (itself already stronger than the published Kaynov base-stock). The learned per-retailer soft tree **beats the tuned gate beyond sampling error on two of the three instances** (+1.33% and +6.44%) and ties on the hardest one (a search-limited tie, not a representation limit).

To be clear about the ceiling: the learned policy does **not** beat the published PPO, which remains the strongest *learned* reference on every row (we sit 3.14–17.77% below it). The win is over the tuned heuristic gate, not over published deep RL.

## Ameliorating inventory — beats the order-up-to gate; the LP value is an upper bound

Here stock *improves* with age (think spirits, port wine) and the objective is long-run average **profit** with a stochastic purchase price. A price-reactive linear-leaf soft tree — buy more when the realized price is low, which a fixed order-up-to level cannot do — **beats the best tuned order-up-to gate by a wide margin** (+450% and +278% of the gate's profit, with overwhelming statistical significance).

The other reference point is a **perfect-information LP upper bound**, and I treat it as exactly that. The remaining gap to it is large (94.2% and 79.3%) and *structural*: the bound assumes hindsight and a full three-part decision (purchase, production, per-age issuance solved jointly), while our policy controls only the scalar purchase volume. So the bound is **reported as a gap, never "beaten"**, and it is not comparable to the ~3.5% gap a full-action-space deep-RL agent reaches.

## Production / assembly / distribution network — a research result on a faithful environment

A 3-node serial production chain. The honesty status matters: this environment faithfully reproduces the one *published* single-node quantity (certifying its dynamics), but there is **no published optimum for the multi-node MDP**, so the comparator is the **environment's own best heuristic** (a grid-searched pairwise base-stock), not a literature number. Against that gate, the learned linear-leaf soft tree improves per-period cost by **~4–9%** across two seeds and two depths, robustly outside the standard error.

I frame this honestly as a **research result on a faithful-but-not-literature-anchored environment** — the policy beats the environment's own best heuristic, not any published cost. It is evidence for the same thesis as everywhere else: action design (here, the leaf class) governs whether a black-box search recovers structured-control performance.

# What ties it together

The same gradient-free loop, run with one fixed configuration and no per-problem tuning, produces:

| Problem | Comparator | Honest verdict |
|---|---|---|
| Lost sales | classical heuristics | instance-best in **22/24** |
| Fixed-cost lost sales | $(s,S)$, $(s,nQ)$, $(s,S,q)$ | instance-best in **47/48** |
| Dual sourcing | capped dual-index (optimal proxy) | **matches** the proven optimum |
| Divergent multi-echelon | best in-env. base-stock | **≈14.4%** better (A3C comparison indicative) |
| Perishable | best base-stock gate | **beats** the gate (+1.16% / +0.82%) |
| General-network backorder | published constant base-stock | **beats** by >20% (below PPO, not a PPO beat) |
| Serial (Clark–Scarf) | proven optimum | **matches** the proven optimum |
| One-warehouse multi-retailer | tuned base-stock gate | **beats** the gate on 2/3 (below PPO) |
| Ameliorating | order-up-to gate; LP bound | **beats** the gate; LP value is an upper bound |
| Production network | env.'s own best heuristic | **beats** it by ~4–9% (research result) |

The policies that produce this carry **tens to a few hundred parameters** — orders of magnitude fewer than published DRL networks — and the reported single-state actions are validated against an independent rollout, so they are not just compact but checkable.

Two limitations are worth stating plainly. The high-penalty, autocorrelated MMPP lost-sales instances at the longest lead times stay won by classical heuristics — a real gap for stationary compact policies under regime-switching demand. And CMA-ES, while far more sample-efficient than A3C here, is still data-hungry relative to limited-data settings, and its population evaluation cost grows with the covariance dimension, so very large parameterizations would need restricted or separable covariance structures.

# Takeaways

- A single, generic recipe — **CMA-ES over small, interpretable policies** — is competitive across ten classical inventory problems, with little tuning.
- The consequential design choice is the **decoder and its action geometry**, not the network size: shape the decoder like the relevant heuristic's coordinate system and a tiny policy reaches the good operating region.
- Read the verdicts honestly: **match** where the comparator is a proven optimum (dual sourcing, serial Clark–Scarf), **beat** where the comparator is a heuristic, and **below** published deep RL where one exists. Understatement is the right default.

# Citation

If you find this work useful, please cite it as:

```
@article{ManafInventoryES,
  title   = "Learning Inventory Control Policies with Evolution Strategies across Lost-Sales, Dual-Sourcing, and Multi-Echelon Problems",
  author  = "Manaf, Nima",
  journal = "nimamanaf.com",
  year    = "2026",
  url     = "https://nimamanaf.com/posts/evolution-strategies-for-inventory-control"
}
```
