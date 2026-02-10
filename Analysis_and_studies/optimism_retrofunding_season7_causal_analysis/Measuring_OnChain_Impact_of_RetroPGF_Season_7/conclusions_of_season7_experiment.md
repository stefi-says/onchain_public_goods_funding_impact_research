# Measuring RetroPGF Season 7 impact: what worked, what didn’t, and why

In 2025, I ran an exploratory causal analysis to test a simple question: **can we detect measurable, on-chain changes after RetroPGF Season 7 funding?** I published the full write-up (method, plots, diagnostics, and caveats) here:

- [Measuring On-Chain Impact of RetroPGF Season 7: An Exploratory Causal Analysis (GitHub)](https://github.com/stefi-says/onchain_public_goods_funding_impact_research/blob/main/Analysis_and_studies/optimism_retrofunding_season7_causal_analysis/Measuring_OnChain_Impact_of_RetroPGF_Season_7/Measuring_OnChain_Impact_of_RetroPGF_Season_7.md)

After sharing that post, I gathered feedback from round operators, analysts, and ecosystem stakeholders. This follow-up is a personal “what I learned” note meant for **round operators** and **data nerds**.

**What you’ll get from this post:**
- The biggest **technical failure modes** I ran into (and why they happen in crypto time series)
- The most important **assumption mismatches** between common causal models and how grant programs actually work
- The **missing data + infrastructure** that would make this kind of evaluation meaningfully more rigorous in future rounds

## Points of attention (technical)

This analysis was conducted independently. I used TensorFlow Probability’s `tfp-causalimpact` (a Bayesian Structural Time Series / BSTS causal impact model) together with on-chain time series from OSO ([Open Source Observer](https://www.oso.xyz/)). Context and assumptions came from public forum discussions and stakeholder interviews.

Because I was looking for a measurable (and as conclusive as possible) outcome—and because the model itself requires a concrete outcome metric—I also want to clarify the “return” framing I used. When I talked about “positive return,” I used a narrow proxy—**higher activity on a funded protocol → more transactions → more transaction/sequencer fees**, which (in theory) could eventually break even against the funding amount. In this experiment, the best available proxy for “activity” was **daily transaction count per protocol**. This framing **does not** capture broader ecosystem value (positive externalities / network effects), where increased activity in one protocol can create spillovers that benefit other projects and the ecosystem as a whole.

The next sections are implementation-focused “points of attention.” I’m highlighting them both to help future analysts reproduce (or improve on) this type of causal analysis, and because these technical constraints often point directly to **missing ecosystem infrastructure** that would make impact evaluation more reliable.

### 1) Noisy data (and why transformations were necessary)

If you look at the [notebook](https://github.com/stefi-says/onchain_public_goods_funding_impact_research/blob/fa002d1c6af278a3d636f7993d4b74b3289cbfc6/Analysis_and_studies/optimism_retrofunding_season7_causal_analysis/notebooks/analysis.ipynb) and at the [analysis article](https://github.com/stefi-says/onchain_public_goods_funding_impact_research/blob/fa002d1c6af278a3d636f7993d4b74b3289cbfc6/Analysis_and_studies/optimism_retrofunding_season7_causal_analysis/Measuring_OnChain_Impact_of_RetroPGF_Season_7/Measuring_OnChain_Impact_of_RetroPGF_Season_7.md) , you’ll notice I ended up using **two transformations** on the transaction time series to reduce volatility and make modeling feasible:

- **Log transform**: helps when activity grows roughly exponentially over long periods (it compresses large values).
- **Box–Cox transform**: a power transform \(y^{(\lambda)}\) used to **stabilize variance** and make errors closer to **Gaussian**. Useful because BSTS/CausalImpact is typically fit with an observation model that assumes **approximately normal, homoskedastic noise**.

In plain language: without variance stabilization, a few “spike” days can dominate the fit. These transforms make the relationship between the **outcome series** and the **counterfactual predictors** more stable (often improving fit and reducing residual issues like autocorrelation).

Even with these transforms, the Box–Cox–transformed series were still not close to normal (skew / heavy tails), which suggests the model’s observation-noise assumptions are still a mismatch. In practice, that can inflate the estimated “noise” and widen credible intervals—so a real but modest treatment effect can be hard to detect. For any future application of this technique, this assumption mismatch should be revisited and validated to ensure trustworthy inference.

![Base transactions after log and boxcox transformation ](analysis_ntbk_media/eg_base_boxcox_transf_dist.png)

*Figure: Base transactions after log and boxcox transformation used to compose the counterfactual series* 

### 2) Model mismatch: “single intervention” vs how grants actually work
For this experiment I used TensorFlow Probability’s [`tfp-causalimpact`](https://pypi.org/project/tfp-causalimpact/) (a Bayesian Structural Time Series / BSTS approach). In practice, that workflow is built around **one intervention date**: you define a pre-period, define a post-period, and ask “did something change after this point?”

That assumption is often a mismatch for grant programs:

- **Multiple payments**: funding is frequently distributed in tranches (sometimes tied to milestones), not as one clean “on/off” event.
- **Varying dosage**: different payment sizes can plausibly produce different effect sizes (a dose–response relationship).
- **Lagged + gradual effects**: even if funding helps, it may show up slowly (hiring, shipping features, running incentives, integrations).
In this analysis, I simplified all of that into a single “before vs after” cut. That makes the analysis easier to run, but it can also **hide real effects** (or make them look noisy) when the true impact is gradual, delayed, or spread across multiple funding moments.
For future applications, it’s essential to consider alternative causal designs and model families that better match grant dynamics (staggered payments, lagged effects, and varying “dosage”).
    
### 3) Counterfactual selection is the hardest part (and it’s mostly an ecosystem data problem)

Every causal impact setup needs a **counterfactual**: a reasonable estimate of “what would have happened without the grant.” In BSTS/CausalImpact, that usually means building a control signal from **similar projects** and **related ecosystem time series** that predict the target project in the pre-period.

In practice, this is fragile in grant ecosystems:

- **Similarity requires context**: to pick valid control projects, you need domain knowledge (are they actually comparable? do they serve the same users? are they exposed to the same market cycles?).
- **Controls must be untreated**: if a “control” project received *any* meaningful funding or incentives during the analysis window (from any ecosystem), it’s no longer a clean control—it’s partially treated.

The blocker I ran into is simple: **we don’t have a shared, reliable registry of funding and incentive events across ecosystems.** Grants are scattered across forums, spreadsheets, multisigs, and announcements. Without that registry, analysts can unintentionally include treated projects as controls, which weakens identification and makes results harder to trust.

## The infrastructure we’re missing

This experiment made a gap painfully clear: as an analyst trying to evaluate grants rigorously, I couldn’t find a consistent “source of truth” for key inputs like **who got funded, when funds arrived (by tranche), what the funds were intended for, and which other incentives were running in parallel**.

The goal of this section is not to prescribe a single evaluation method—it’s to highlight practical ecosystem infrastructure that would reduce blind spots and make future impact analyses more credible, comparable, and actionable for round operators.

Based on that experience, a few concrete data practices can make downstream analysis much more informative:

- **A grants + incentives registry**: who received what, from which program/round, for what purpose (e.g., infrastructure, growth/incentives, research).
- **Payment schedule / tranche-level data**: when funds actually arrived (and how much each time).
- **Deployment / usage tags (when available)**: simple labels + dates for the “real interventions” that happen after funding. For example: **hire start dates**, **feature/integration launch dates**, and **incentive campaign start/end dates** (or a coarse spend tag like “mostly audits” vs “mostly incentives”). These don’t need to be perfect—just enough to align expected impact timing with on-chain outcomes and interpret delays.

Separately, **basic project metadata** (category, chain coverage, maturity, business model, size of the team) makes it easier to group similar projects into cohorts. That improves **benchmarking** (“what’s normal for this cohort?”) and makes it easier to build more credible **control groups** for causal analysis.


Collecting this kind of information tends to unlock the questions grant managers and round operators usually want answered:

- **Which grant types** (e.g., infrastructure vs growth vs research) tend to correlate with stronger outcomes?
- **Which projects** respond best to which kind of support (and at what approximate “dosage”)?
- **How long** does it typically take for different interventions to show measurable effects (weeks vs months)?
- **What does “good performance” look like for a given cohort?** (e.g., early-stage infra vs mature apps) and what’s a reasonable benchmark to compare against?
- **Which projects should we include in future rounds given what we learned from early grants?** For example: do projects that show strong follow-through and leading indicators after an initial small grant tend to outperform later?
- **Are effects sustained or temporary?** (e.g., does activity persist after an incentive campaign ends?)

---

Thanks for reading. I’m sharing this work to contribute to the broader ecosystem’s ability to measure impact more rigorously, learn faster from funding experiments, and make better allocation decisions over time. If you’re working on grant evaluation, causal inference, or RetroPGF impact measurement, I’d love to compare notes—reach out on Telegram: `@hi_stefi`.

---

## References

1. Google. “TensorFlow Probability CausalImpact.” `https://github.com/google/tfp-causalimpact`
2. Open Source Observer (OSO). `https://www.opensource.observer/`
3. Optimism RetroPGF. `https://gov.optimism.io/t/season-7-retro-funding-early-evidence-on-onchain-builders-impact/10163`
4. Toward Recurrent and Concurrent Grants Rounds in Web3. `https://mirror.xyz/stefipereira.eth/SNXPcTKTO88BGgctU_eJw5_N_q6Tw23q4ed1zGBdCHo`

## License

This work is shared under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) — feel free to build upon it with attribution.





