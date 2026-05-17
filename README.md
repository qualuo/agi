# agi

An agent runtime engine. Not AGI — that remains an unsolved scientific
problem. This is what you can credibly build today: a capable agent on top
of Claude Opus 4.7 with tools to act on the world, persistent memory and
skills across sessions, sandboxed self-extension via tool synthesis,
subagent delegation, and an event-driven runtime surface that a higher-level
coordination engine can drive.

## What's in here

```
agi/                # runtime + agent + reference coordinator
  runtime.py        # Runtime, Session, SessionConfig — the engine surface
  events.py         # EventBus + typed Event kinds (the coordination signal)
  server.py         # HTTP+SSE server exposing Runtime (stdlib only)
  protocol.py       # JSON-RPC 2.0 over stdio — drive the Runtime as a subprocess
  agent.py          # streaming agent loop — adaptive thinking + tool dispatch
  coordinator.py    # reference Coordinator + Goal/Plan/PlanStep abstractions
  goalc.py          # Goal compiler: heuristic + LLM-based default decomposers
  autoloop.py       # AutonomousLoop — retry-with-lessons until goal accepted
  fork.py           # SessionFork — race N variants, pick winner by critic
  pool.py           # RuntimePool — federation: many runtimes, one dispatch surface
  capabilities.py   # observed-performance routing — learn which roles win where
  policy.py         # PolicyRouter — Thompson-sampled bandit on top of capabilities
  selfeval.py       # SelfEvalBank — agent-mined regression suite + promotion gate
  autonomy.py       # AutonomyEngine — continuous closed-loop self-improvement
  knowledge.py      # KnowledgeGraph — typed nodes + relations + facts
  governance.py     # multi-tenant budgets, quotas, rate limits, fair-share
  preflight.py      # cost/duration/p_success forecast + admission advisor
  mcp.py            # Model Context Protocol server: drive from Claude Desktop/Code
  evolve.py         # EvolutionEngine — closed-loop self-improvement over strategies
  contract.py       # TicketSLO + SLOCompiler + hedged execution + ComplianceLedger
  driver.py         # RuntimeDriver — single entry point with portfolio + SLO surfaces
  oracle.py         # TicketOracle — counterfactual receipt replay + admission auto-tune
  experiments.py    # ExperimentRunner — A/B experiments with guardrails (Bayesian decisions)
  portfolio.py      # PortfolioOptimizer — fixed-budget allocation across many tickets
  attest.py         # AttestationLedger — tamper-evident, HMAC-signed receipt chain
  calibration.py    # CalibrationEngine — isotonic/Platt recalibration of p_success
  policy_lab.py     # PolicyLab — off-policy evaluation (IPS/SNIPS/DM/DR/SWITCH-DR)
  policy_improver.py # PolicyImprover — safe off-policy *optimization* (CRM/POEM) with finite-sample HCPI (Bernstein-LCB) safety gate; the policy-shipping dual of PolicyLab
  conformal.py      # ConformalPredictor — distribution-free, finite-sample-valid prediction intervals (CQR / Mondrian / Jackknife+ / ACI / RAPS)
  causal.py         # CausalLab — heterogeneous treatment effects (T/S/X/DR-learners, Qini, BLP, permutation test)
  strategist.py     # Strategist — top-level meta-decision API; fuses calibration + conformal + causal + OPE into one risk-adjusted recommendation
  experiment_design.py # ExperimentDesigner — Bayesian Optimal Experiment Design (EIG / BALD / KG / Thompson Top-K / Fedorov D-optimal); picks the next batch of experiments that maximally sharpens the policy per dollar
  deliberator.py    # Deliberator — adaptive sequential sampling kernel; anytime-valid stopping (WSR confidence sequences) so the runtime spends 1-3 samples on easy queries and escalates on ambiguous ones, with one quality dial α
  drift.py          # DriftSentinel — anytime-valid sequential drift detection (Page-Hinkley CUSUM + BOCPD changepoint posterior + Shin-Ramdas-Rinaldo e-process); the trust-gate that tells the coordinator when calibration/conformal/policy estimates have gone stale
  arbiter.py        # Arbiter — fixed-confidence Best-Arm Identification (Track-and-Stop / KL-LUCB / Sequential Halving); identifies the winning model/prompt/policy at (ε, δ)-PAC with asymptotically optimal sample complexity, the cross-strategy dual of Deliberator
  cartographer.py   # Cartographer — zone-of-proximal-development curriculum kernel; Beta-Binomial competence with Wilson CIs + Oudeyer learning-progress + prereq-DAG + Sviridenko submodular knapsack; the *what-should-I-learn-next* primitive — upstream of Arbiter
  coalition.py      # Coalition — Shapley credit-assignment as a runtime primitive (exact + Monte-Carlo + stratified + Owen group + Banzhaf); anytime PAC bounds via Hoeffding/Bernstein; the *who-deserves-the-credit* primitive for retraining-budget allocation, multi-tenant cost split, and skill deprecation
  robustifier.py    # Robustifier — Distributionally Robust Optimization (Wasserstein-1 / KL / χ² / CVaR / Empirical Likelihood); worst-case-mean evaluation over a finite-sample-valid uncertainty ball; the *how-to-plan-against-the-drift-you-haven't-yet-seen* primitive — downstream of DriftSentinel, upstream of Strategist
  auditor.py        # Auditor — multiple-hypothesis testing with FDR / FWER control (BH / BY / Holm / Hochberg / Bonferroni / Šidák / Storey adaptive / e-BH / LORD / SAFFRON / ADDIS / α-investing) + Fisher/Stouffer/Simes/HMP combiners; the *which-of-our-thousand-simultaneous-drift-or-experiment-or-arbiter-tests-are-actually-real* primitive — composes with DriftSentinel, ExperimentRunner, Arbiter, CausalDiscoverer
  negotiator.py     # Negotiator — multi-party allocation as a runtime primitive (utilitarian / egalitarian / leximin / Nash bargaining / Kalai-Smorodinsky / proportional-fair / VCG); KKT-based water-filling on concave-utility, sealed-bid externality on indivisible items; the *fair-and-truthful-split-when-N-tenants-compete-for-the-same-finite-resource* primitive — composes with TicketMarket, TicketEconomist, PortfolioOptimizer, Coalition
  equilibrator.py   # Equilibrator — non-cooperative game-theoretic equilibria as a runtime primitive (Nash / pure Nash / correlated / coarse correlated / minimax / ESS) via support enumeration / multiplicative weights / fictitious play / replicator dynamics / Big-M simplex; exact 2-player zero-sum LP and exploitability (NashConv) on every profile; the *strategically-stable-when-N-agents-don't-cooperate* primitive — composes with Negotiator (threat-points), Coalition (worst-case characteristic function), Robustifier (adversarial games), TicketMarket (incentive-compatibility verification), Strategist (exploitability-penalised meta-decisions)
  transporter.py    # Transporter — optimal transport as a runtime primitive (Hungarian assignment / log-stabilised Sinkhorn / Sinkhorn-divergence / sliced Wasserstein / closed-form 1-D EMD / unbalanced Sinkhorn / entropic Gromov-Wasserstein / 1-D Kantorovich-Rubinstein dual / 1-D W_2 barycenter); cyclic-monotonicity optimality witnesses, Weed-Bach finite-sample bias bound, marginal-violation feasibility certificate; the *how-far-apart-are-these-two-distributions-and-which-plan-realises-it* primitive — composes with Robustifier (W_1/W_2 DRO balls materialised with explicit plan + dual potential), DriftSentinel (W_1 drift score with distribution-free finite-sample bound), CausalDiscoverer (Hungarian or Sinkhorn counterfactual matching of treated/control), Conformal (CDF-distance goodness-of-fit), Coalition (Wasserstein-Shapley), PolicyImprover/PolicyLab (state-visitation W_1 between policies), Strategist (risk-adjusted decisions on the (EV, W_1-shift) plane)
  forecaster.py     # Forecaster — anytime-valid probabilistic forecasting as a runtime primitive (Bernoulli / Categorical / Gaussian / Empirical / Interval forecasts); strictly proper scoring rules (Brier / log / spherical / quadratic / CRPS / pinball / linex); anytime-valid calibration test via an aGRAPA-betting e-process on the PIT (Ville's inequality — rejects calibration breach under *any* stopping rule); Massart-DKW finite-sample KS thresholds, Anderson-Darling A² with Stephens (1986) p-value; isotonic / histogram / PIT / Platt recalibration; Hedge (Cesa-Bianchi & Lugosi) and polynomial-weights aggregators with provable O(√(T log K)) cumulative regret; online conformal prediction intervals; the *give-me-a-calibrated-probability-and-prove-it* primitive — composes with Arbiter (anytime-valid stopping on forecast scores), Auditor (FDR-controlled batched calibration tests across streams), Equilibrator (game-theoretic interpretation of forecasts as bets), Strategist (decision-theoretic loss minimisation under proper scoring), DriftSentinel (e-process detects forecast-calibration drift)
  truthserum.py     # TruthSerum — incentive-compatible peer-prediction as a runtime primitive (Output Agreement / Bayesian Truth Serum (Prelec 2004) / Robust BTS (Witkowski-Parkes 2012) / Correlated Agreement (Dasgupta-Ghosh 2013) / Determinant-based MI (Kong 2020) / f-MI (Kong-Schoenebeck 2018) / Surrogate Scoring Rules (Liu-Wang-Chen 2020)); Hoeffding / empirical-Bernstein anytime-valid confidence intervals on every reporter's expected payment; Bonferroni-joint α across reporters; empirical strict-Nash equilibrium check (the worst-deviation gap across constant-report strategies); Dawes-Skene EM (per-reporter confusion matrices + posterior-truth aggregation, no ground truth required); collusion-clique detection at joint level α; the *elicit-honest-reports-from-N-agents-when-you-have-no-ground-truth* primitive — composes with Auditor (FDR-controlled per-reporter truthfulness tests across streams), Coalition (Shapley credit over reporters), Forecaster (calibrated belief reports become SSR inputs), Equilibrator (truthful-Nash verification on the empirical payoff bimatrix), Negotiator (truthful allocation of reward conditional on elicited reports), Strategist (decision under noisy aggregated truth)
  persuader.py      # Persuader — Bayesian persuasion / information design as a runtime primitive (binary-state exact concavification (Kamenica-Gentzkow 2011) + general |Ω| × |A| LP via Bergemann-Morris (2016) BCE obedience constraints, solved with a stdlib revised-simplex + Bland's-rule no-cycling pivoting; online Bayesian persuasion via Hedge / IPS over an ε-net of schemes with O(√(T log K)) regret bound (Castiglioni-Marchesi-Romano-Gatti 2020); robust maxmin persuasion over a finite prior set (Dworczak-Pavan 2022); independent private multi-receiver (Babichenko-Barman 2017) and public-signal multi-receiver via joint-profile BCE (Mathevet-Perego-Taneva 2020); Bayes-plausibility + obedience verifier with deviation-gain certificate; Hoeffding / empirical-Bernstein anytime PAC certificate on simulated sender payoff); the *what-information-should-I-reveal-to-make-N-strategic-agents-take-the-right-action* primitive — the transfer-free dual of MechanismDesigner; composes with TruthSerum (elicit receiver utility), Equilibrator (verify BCE on the obedience polytope), Negotiator (refine recommendations into a fair-and-truthful allocation), ActiveInferencer (drop-in solver for receiver best response when receiver is a generative model), Strategist (risk-adjusted choice between info-design and payment-design), AttestationLedger (commitment via tamper-evident receipt)
  submodular.py     # Submodular — discrete subset selection as a runtime primitive (lazy greedy (Minoux 1978) / naive greedy (Nemhauser-Wolsey-Fisher 1978) / CELF (Leskovec et al. 2007) / stochastic greedy (Mirzasoleiman et al. 2015) / cost-benefit greedy (Khuller-Moss-Naor 1999) / Sviridenko 2004 partial-enumeration knapsack / randomised + deterministic double greedy (Buchbinder-Feldman-Naor-Schwartz 2015) / distorted greedy for γ-weakly-submodular (Harshaw-Feldman-Ward-Karbasi 2019) / Sieve-Streaming one-pass (Badanidiyuru et al. 2014) / Wolsey 1982 submodular cover / threshold greedy (Badanidiyuru-Vondrák 2014)); built-in objectives (facility location / weighted coverage / log-determinant DPP / Gaussian entropy / max-cut / concave-over-modular / feature-based summarisation); Conforti-Cornuéjols 1984 curvature-aware bound; anytime PAC certificate of submodularity via Hoeffding / empirical-Bernstein (Maurer-Pontil 2009) on random DR-violation samples; the *which-K-of-these-N-options-do-I-pick* primitive — composes with Cartographer (next-K frontier tasks under the prereq-DAG), ExperimentDesigner (batch BOED via greedy on mutual information), Coalition (Shapley-baseline credit on submodular utility), Negotiator (subset allocation of indivisible items), Auditor (top-K significant findings post-BH), PolicyLab / PolicyImprover (diverse counterfactual policy bank via log-det), Strategist (cost-benefit portfolio under fixed budget), Forecaster (diverse-yet-skillful ensemble pick), Skills (top-K retrieval with redundancy penalty), AttestationLedger (replayable subset-decision receipts)
  diplomat.py       # Diplomat — counterfactual regret minimization for extensive-form games with imperfect information (vanilla CFR / CFR+ (Tammelin 2014, the *Science*-paper algorithm for heads-up limit Texas hold'em) / Linear CFR / Discounted CFR (Brown-Sandholm 2019, the algorithm behind Libratus) / Predictive CFR+ (Farina-Kroer-Sandholm 2021, O(1/T) last-iterate) / external-sampling MCCFR / chance-sampling CFR / outcome-sampling MCCFR / exact sequence-form LP (von Stengel 1996, solved with a stdlib mixed-constraint Big-M revised-simplex); exact O(|tree|) best-response and NashConv exploitability; Kuhn 1953 poker + matching pennies + RPS + multi-round bargaining + private-signal coin-match builders; perfect-recall verification at build time; tamper-evident certificate fingerprint on every SolveReport); the *sequentially-stable-when-N-agents-play-an-imperfect-information-protocol* primitive — the extensive-form / sequential dual of Equilibrator; composes with Equilibrator (one-shot specialisation), Negotiator (multi-round bargaining protocols), Persuader (multi-stage information design), TruthSerum (truthful Nash on the entire EFG, not just per round), MechanismDesigner (dominant-strategy IC verified on the full game tree), Strategist (exploitability-adjusted policy shipping), AttestationLedger (commitment to the equilibrium before any agent acts)
  synthesizer.py    # Synthesizer — program synthesis as a runtime primitive (typed AST + DSL declaration / version-space enumeration by depth (Mitchell 1982) and by MDL-cost-priority-queue Dijkstra over the program graph / counterexample-guided inductive synthesis (Solar-Lezama 2008) with arbitrary executable verifier / programming-by-example over built-in string (Gulwani 2011 FlashFill primitives — concat, substring, split, replace, case-fold, strip, index-of, length), integer (+, -, *, //, %, min, max, abs, ifzero), and list (head, last, length, sum, max, min) DSLs / anti-unification / least general generalisation (Plotkin 1970) / L\\* DFA learning from membership + equivalence oracles (Angluin 1987)); Blumer-Ehrenfeucht-Haussler-Warmuth (1987) Occam's-razor PAC bound on generalisation error from AST complexity, sample-complexity inversion ``m ≥ (|h| ln 2 + ln(1/δ)) / ε``; tamper-evident SHA-256 fingerprint over (DSL, examples, AST); top-K alternative-program enumeration; ``max_visited`` search cap; pure stdlib — no Z3, no SMT-LIB, no PEG parser; the *write-me-a-verifiable-program-from-examples-and-give-me-a-PAC-bound-on-its-generalisation* primitive — the **self-extension** mechanism the runtime needs to author its own tools — composes with Toolsynth (sandboxed execution of synthesised programs), SkillMine (promote programs to Skill library when Occam bound clears threshold), AttestationLedger (tamper-evident receipt-replayable synthesis), Auditor (BH on held-out e-values across top-K candidates), Sampler (ADVI fit of continuous constants in a symbolic skeleton), AutonomousLoop (synthesise missing skills from successful traces), EvolutionEngine (AST-level mutation / crossover over strategies)
  sampler.py        # Sampler — Bayesian probabilistic inference as a runtime primitive (Random-walk Metropolis with Haario-Saksman-Tamminen 2001 adaptive proposal / Metropolis-adjusted Langevin (Roberts-Tweedie 1996, optimal acceptance 0.574) / Unadjusted Langevin (Durmus-Moulines 2017) / Hamiltonian Monte Carlo (Neal 2011) with diagonal mass and dual-averaging step-size adaptation / No-U-Turn Sampler (Hoffman-Gelman 2014, recursive doubling + slice variable + U-turn termination + Nesterov 2009 primal-dual log-step averaging — the algorithm behind Stan) / Slice sampling (Neal 2003, zero-tuning stepping-out + shrinkage) / Parallel tempering with geometric ladder and replica exchange (Earl-Deem 2005) / Sequential Monte Carlo with adaptive geometric tempering and unbiased log-evidence estimator (Del Moral-Doucet-Jasra 2006) / self-normalised importance sampling with PWM-fitted Pareto-k tail diagnostic (Vehtari-Simpson-Gelman-Yao-Gabry 2024) / mean-field and full-rank ADVI on the reparameterised ELBO (Kucukelbir-Tran-Ranganath-Gelman-Blei 2017) with AdaGrad); rank-normalised split-R̂ (Vehtari et al. 2021), bulk-ESS, tail-ESS, Geyer 1992 initial-monotone-sequence integrated autocorrelation time, Geweke 1992 stationarity z-score, divergence and max-tree-depth counters; anytime-valid credible sets via Massart-DKW finite-sample CDF band and Howard-Ramdas-McAuliffe-Sekhon (2021) bounded-mean confidence sequence; tamper-evident reproducibility fingerprint on every SampleReport; the *give-me-the-posterior-and-prove-it* primitive — the foundational Bayesian-inference engine for Forecaster (posterior predictive), Causal (non-conjugate ATE posteriors), ActiveInferencer (particle belief), PolicyImprover (amortised safe-policy posterior via ADVI), Strategist (Thompson sampling on diagnosed-converged posterior), Persuader (uncertain-prior robust persuasion), CausalDiscoverer (parallel-tempered DAG posterior), AttestationLedger (replayable sample-receipt)
  refuter.py        # Refuter — automated falsification as a runtime primitive (Popperian conjecture-and-refutation in code; portfolio search over typed search spaces — boundary corner enumeration (Goldberg 1991 IEEE-754 corners), Halton 1960 low-discrepancy quasi-random, uniform random, (1+λ) evolution strategies (Rechenberg 1973) with 1/5-success-rule step adaptation, optional Nelder-Mead 1965 simplex; metamorphic relations (Chen-Cheung-Yiu 1998) for oracle-free testing; bound mode for tightness audits; sequential anytime-valid e-process (Vovk-Wang 2021) accumulating Bernoulli evidence under Ville's inequality — reject at any stopping time; QuickCheck-style structural shrinking (Hughes-Claessen 2000) reduces witnesses to minimal form; exact Clopper-Pearson 1934 finite-sample UCB on failure rate (1 − α^{1/n} at k=0; closed-form via beta inversion otherwise); CEGIS scaffold (Solar-Lezama 2008) for refute-then-resynthesise loops with Synthesizer; tamper-evident SHA-256 fingerprint over (predicate signature, space, seed, witnesses, strategy counts) on every RefutationReport; pure stdlib — no SMT solver, no autograd); the *try-to-break-every-claim-this-runtime-makes* primitive — the **epistemic** mechanism that turns any other primitive's output into a falsifiable, statistically-bounded claim — composes with Synthesizer (CEGIS over any DSL), Forecaster (metamorphic PIT-uniformity refutation), Sampler (posterior-predictive stress test), ConformalPredictor (coverage refutation on adversarial points), CausalDiscoverer (refute CI claims that justified an edge orientation), Submodular (refute diminishing-returns / submodularity), Skills (refute pre/post-conditions before action), AttestationLedger (replay-verifiable refutation receipts), Auditor (BH on per-claim e-values for multiple-refutation control), AutonomousLoop (refute every plan's preconditions before commit)
  privacy.py        # PrivacyAccountant — differential privacy as a runtime primitive (Laplace (Dwork-McSherry-Nissim-Smith 2006) / Gaussian with Balle-Wang 2018 analytic σ-calibration (tight (ε,δ) for any ε; 20-30% noise savings vs. classical √(2 ln(1.25/δ))Δ/ε at ε≤1) / classical Dwork-Roth Gaussian / Mironov 2012 snapping mechanism with floating-point side-channel resistance and 2λ/b ε overhead / McSherry-Talwar 2007 exponential mechanism for private 'best of N' selection / Lyu-Su-Li 2017 corrected Sparse Vector Technique (ε₁ on threshold, ε₂·c on c positive answers) / Chan-Shi-Song 2010 binary-tree mechanism for continual release with O(log T) privacy loss per prefix sum); RenyiAccountant (Mironov 2017) tracks α-Rényi divergence over a configurable α-grid with additive composition and tight α-optimal conversion ε_{ε,δ} = inf_α ε(α) + log(1/δ)/(α−1); tight subsampled-Gaussian RDP (Mironov-Talwar-Zhang 2019; Wang-Balle-Kasiviswanathan 2019) for DP-SGD-style minibatch updates; basic / advanced (Dwork-Rothblum-Vadhan 2010 ε √(2k ln(1/δ')) + kε(e^ε−1)) / zCDP (Bun-Steinke 2016) composition theorems; per-release immutable Release receipts with SHA-256 fingerprint over (mechanism, sensitivity, ε, δ, seed, value_in, value_out); a per-session ledger_hash chaining all release fingerprints; hard-fail privacy odometer that raises BudgetExhausted when a request would exceed the target (ε,δ); pure stdlib — Beasley-Springer 1977 / Moro 1995 standard-normal inverse CDF, math.erf for Φ); the *prove-the-(ε,δ)-DP-bound-on-everything-this-runtime-touches* primitive — the **regulatory** mechanism every enterprise / regulated-industry deployment needs — composes with AttestationLedger (each Release is replay-verifiable), Auditor (refuses ingestion when odometer trips), Sampler (DP-SGD via the moments accountant), Forecaster (DP score release on held-out labels), Cartographer (per-task counters), Coordinator (per-user budget on Session boundaries), Refuter (refute the (ε,δ) claim itself via metamorphic invariance on neighbouring datasets)
  bandit.py         # Bandit — sequential decision under uncertainty as a runtime primitive (UCB1 (Auer-Cesa-Bianchi-Fischer 2002 finite-time analysis, R_T ≤ 8 Σ_{Δ>0} log T / Δ + (1 + π²/3) Σ Δ) / KL-UCB (Garivier-Cappé 2011 — asymptotically Lai-Robbins-optimal for Bernoulli, matches lim R_T / log T = Σ_a Δ_a / d(μ_a, μ*)) / MOSS (Audibert-Bubeck 2009/2010 — minimax-optimal Õ(√(KT))) / UCB-V (Audibert-Munos-Szepesvári 2009 — empirical-Bernstein bonus on low-variance arms) / Thompson Sampling Beta-Bernoulli (Thompson 1933; Agrawal-Goyal 2012 — Bayesian, matches Lai-Robbins) / Thompson Sampling Gaussian (Russo-Van Roy-Kazerouni-Osband-Wen 2018 tutorial, conjugate Normal-Normal posterior) / Successive Elimination (Even-Dar-Mannor-Mansour 2006 — anytime PAC kicks out KL-dominated arms) / ε-greedy with 1/√t Cesa-Bianchi-Fischer 1998 schedule / EXP3 (Auer-Cesa-Bianchi-Freund-Schapire 2002 — adversarial, R_T ≤ 2√(e-1) √(TK log K)) / EXP3-IX (Neu 2015 — implicit-eXploration high-probability Õ(√(KT))) / Tsallis-INF (Zimmert-Seldin 2019/2021 — best-of-both-worlds: simultaneously minimax-optimal adversarial AND Lai-Robbins stochastic, no parameter tuning, OMD with negative Tsallis-1/2 entropy regulariser, dual variable solved by bisection) / LinUCB (Li-Chu-Langford-Schapire 2010 — per-arm ridge θ̂ = A^{-1}b with α√(xᵀA^{-1}x) bonus, used in Yahoo! Front Page) / OFUL (Abbasi-Yadkori-Pál-Szepesvári 2011 — self-normalised log-det confidence ellipsoid, optimal Õ(d√T)) / Linear Thompson Sampling (Agrawal-Goyal 2013 — sample θ̃ ~ N(θ̂, β² A^{-1}) via Cholesky) / Information-Directed Sampling (Russo-Van Roy 2014/2018 — pulls argmin Ψ²/g info-ratio, provably tighter than Thompson on informative-but-suboptimal arms)); sliding-window forget (Garivier-Moulines 2008 SW-UCB) for non-stationary environments; anytime regret upper bounds via Howard-Ramdas-McAuliffe-Sekhon 2021 confidence sequences (Ville's-inequality time-uniform Õ(√(log log n) / n)) + Maurer-Pontil 2009 empirical-Bernstein on data-driven gaps; tamper-evident SHA-256 fingerprint over (arms, algorithm, rewards, seed) on every BanditReport; pure stdlib — Beasley-Springer 1977 / Moro 1995 inverse-Φ, Marsaglia-Tsang 2000 Gamma → Beta, Box-Muller Gaussian, list-of-lists Cholesky for LinTS); the *earn-while-you-learn* primitive — the cumulative-regret dual of Arbiter (which is fixed-confidence PAC best-arm identification); composes with Arbiter (cumulative-regret vs. PAC-commit duals on the same arms), Strategist (the explore-for-reward policy when STRAT_EXPLORE is returned), PolicyRouter (a special case: Thompson on roles), Forecaster (per-arm posterior is a forecast; PIT calibration applies), Auditor (BH/FDR on joint dominance e-values across concurrent campaigns), Refuter (falsify the bandit itself via adversarial reward streams that violate i.i.d.), PrivacyAccountant (DP-bandit: noisy mean estimates with widened regret bound; Mishra-Thakurta 2015 / Tossou-Dimitrakakis 2017), DriftSentinel (forget(arm, halflife) when CUSUM trips), AttestationLedger (every committed pull is hashed), Cartographer (which-task-next via learning-progress arm), Coalition (Shapley credit on top-K selected arms)
  ranker.py         # Ranker — paired-comparison and partial-ranking inference as a runtime primitive (Bradley-Terry 1952 MM and MAP (Hunter 2004) / Plackett-Luce 1975 MM / Thurstone-Mosteller Case-V MM / Elo 1978 / Glicko 1995 / Glicko-2 2012 with Illinois bracketing on the volatility / TrueSkill 2007 with rectified-Gaussian moment matching and draw-margin support); Tarjan 1972 strongly-connected-component identifiability diagnostic; Hajek-Oh-Xu 2014 ℓ∞ top-K sample-complexity bound; anytime-valid confidence intervals via Hoeffding 1963, Maurer-Pontil 2009 empirical-Bernstein, and Howard-Ramdas-McAuliffe-Sekhon 2021 time-uniform CS on every pairwise win-rate; Fisher-information / Gaussian-prior Hessian standard errors; McFadden pseudo-R²; PAC-certified top-K decision; replay-deterministic state(); tamper-evident SHA-256 fingerprint chaining every observation; Kendall-τ and Spearman-ρ rank correlations; pure stdlib — Beasley-Springer-Moro inverse-Φ, list-of-lists Cholesky on the Fisher information, iterative Tarjan SCC; the *rank-N-candidates-with-confidence-from-pairwise-judgments* primitive — the relative-information dual of Bandit (cumulative regret) and Arbiter (PAC best-arm) — composes with Arbiter (full-ranking dual of single-best-arm PAC), Bandit (dueling bandits — Yue-Joachims 2009; Komiyama et al. 2015), Strategist (pairwise lift CI as a strategy-comparison primitive), Diplomat (rank the players in the extensive-form game), TruthSerum (use judge trust-scores as Ranker observations), Auditor (BH/FDR over many pairwise tests), DriftSentinel (forget(item, halflife) when a player's CUSUM trips), Refuter (falsify Tversky 1969 stochastic-transitivity claims), PrivacyAccountant (DP-private W_ab releases, Hay-Rastogi-Miklau-Suciu 2009), AttestationLedger (every top-K decision hashes into the chain), Forecaster (PIT-calibrate the predicted win-probability stream)
  bayesopt.py       # BayesOpt — Bayesian optimisation as a runtime primitive (Gaussian-process surrogate with stationary kernels — squared-exponential / Matérn-5/2 / Matérn-3/2 with per-dimension ARD lengthscales (MacKay 1994), Cholesky-solved posterior mean and variance (Rasmussen-Williams 2006), analytic input-gradients for gradient-ascent acquisition refinement; GP-UCB (Srinivas-Krause-Kakade-Seeger 2010 — anytime cumulative regret R_T ≤ √(C₁ T β_T γ_T), C₁ = 8 / log(1 + σ_f²/σ²)) / Expected Improvement (Močkus 1974; Jones-Schonlau-Welch 1998 — closed-form EI(x) = (μ−f*)Φ(z) + σφ(z) with Bull 2011 ``O(n^{-ν/d} log^α n)`` simple-regret on Matérn) / Probability of Improvement (Kushner 1964) / Thompson sampling on the GP posterior via Halton candidate set (Kandasamy et al. 2018 — Õ(√(T γ_T β_T)) frequentist regret) / Knowledge Gradient (Frazier-Powell-Dayanik 2009 — quasi-Monte-Carlo one-step-lookahead on the posterior maximiser); batch / parallel suggestions via constant-liar fantasy (Ginsbourger-Le Riche-Carraro 2010); mixed continuous + categorical domains via encoding-based GP; golden-section log-marginal-likelihood lengthscale learning every K observations (Rasmussen-Williams §5.4); anytime instantaneous regret upper bound 2 √β_t · max_x σ_{t-1}(x); information-gain accumulator γ̂_t = ½ Σ log(1 + σ²_t/σ_n²); tamper-evident SHA-256 fingerprint over (config, observation history) on every BayesOptReport; replay-deterministic given config.seed; pure stdlib — inline Cholesky / triangular solve / Beasley-Springer-Moro inverse-Φ / Halton 1960 low-discrepancy quasi-random); the *pick-the-next-expensive-query-and-prove-its-regret* primitive — the continuous-arm dual of Bandit (which is K-armed cumulative regret) and Arbiter (which is finite-arm PAC best-arm); composes with Bandit (warm-start GP with a categorical Bandit's posterior means; bandit-of-acquisitions meta-loop), Arbiter (PAC certification on the BayesOpt incumbent), Sampler (full-posterior Thompson via MCMC over GP hyperparameters), ExperimentDesigner (BayesOpt as the inner loop of any cost-aware design), Refuter (refute that the posterior covers the truth via held-out coverage), Forecaster (treat μ_n(x) ± σ_n(x) as a calibrated forecast; PIT applies), Auditor (BH on per-candidate improvement e-values), PrivacyAccountant (DP-BayesOpt via noisy y_t with widened regret bound), AttestationLedger (replay-verifiable suggest→observe receipts), Strategist (BayesOpt as the strategy when STRAT_EXPLOIT_CONTINUOUS), Coordinator (one expensive black-box per PlanStep)
  reasoner.py       # Reasoner — symbolic logical reasoning as a runtime primitive (DPLL (Davis-Putnam-Logemann-Loveland 1962) with unit propagation + pure-literal elimination / CDCL (Marques-Silva-Sakallah 1996 GRASP, Moskewicz et al. 2001 Chaff, Eén-Sörensson 2003 MiniSat) with two-watched literals (Zhang-Stickel 1996), VSIDS branching (Moskewicz et al. 2001), 1-UIP clause learning (Zhang-Madigan-Moskewicz-Malik 2001), and Luby restarts (Luby-Sinclair-Zuckerman 1993) / Walk-SAT (Selman-Kautz-Cohen 1994) with Schöning 1999 noise / resolution refutation (Robinson 1965) reconstructing an UNSAT proof chain ending in the empty clause / semi-naïve Datalog forward chaining (Bancilhon-Maier-Sagiv-Ullman 1986; Ullman 1989) with Robinson 1965 unification on uppercase-Prolog-convention Datalog variables / SLD-resolution backward chaining (Kowalski 1974) with full backtracking + subsumption tabling (Tamaki-Sato 1986) so left-recursive Horn rules terminate / Answer Set Programming stable-model semantics (Gelfond-Lifschitz 1988) via guess-and-check on NaF atoms with reduct evaluation and stratified-negation fast path (Apt-Blair-Walker 1988) + automatic Herbrand-universe grounding for rules with Datalog variables); Clopper-Pearson 1934 anytime-valid finite-sample upper bound on randomised-solver failure rate (closed form 1−α^{1/n} at k=0, regularised incomplete beta inversion via stdlib continued fraction otherwise); Hoeffding 1963 / Maurer-Pontil 2009 empirical-Bernstein half-widths for model-count importance sampling; Tarjan 1972 iterative SCC on the rule dependency graph for negation-stratification detection; tamper-evident SHA-256 fingerprint chaining every clause, fact, rule, and decision (replay-deterministic given seed); pure stdlib — no Z3, no SMT-LIB, no PEG parser; the *give-me-a-proof-or-give-me-a-counterexample* primitive — the **deterministic-logic** dual of Refuter (which falsifies probabilistic claims with PAC bounds) and the **discrete-logic** complement of Synthesizer (which fills in programs from examples) — composes with Refuter (Reasoner certifies what Refuter cannot refute after a CS-large budget), Synthesizer (Reasoner as the CEGIS verifier; Solar-Lezama 2008 — encode correctness predicate, Reasoner finds counter-example or certifies UNSAT), Negotiator / MechanismDesigner / PortfolioOptimizer (feasibility oracle for hard constraints — integrality, conflicting-resources, capacity), Equilibrator / Diplomat (Boolean side-conditions on equilibria solved before LP / CFR), CausalDiscoverer (Horn-program encoding of v-structure orientation rules), Auditor (BH/FDR control of the false-proof rate across simultaneous reasoning tasks), Cartographer (prereq-DAG forward chain → ready/1 for next-task pick), AttestationLedger (proof tree from backward_chain + resolution proof from UNSAT close hash directly into the ledger), PrivacyAccountant (odometer advance on each add_fact when facts came from sensitive data), Strategist (entailment query "does policy A satisfy invariant I in every model of these rules" before the risk-adjusted score is quoted)
  predictor.py      # Predictor — universal sequence prediction via Context Tree Weighting (Willems-Shtarkov-Tjalkens 1995 *The Context-Tree Weighting Method: Basic Properties*; Willems 1998 *Extensions*) — exact Bayesian mixture over the exponential class of variable-order Markov models of depth ≤ D in O(D) per-symbol time, with redundancy bound `-log₂ P_CTW(x₁ⁿ) ≤ -log₂ P_S(x₁ⁿ) + (|S|·(A-1)/2)·log₂(n/|S|) + 2|S| - 1` against every tree source S of leaves |S| (Krichevsky-Trofimov 1981 parameter redundancy minimax-optimal; 2|S|-1 model redundancy bits from the CTW prior); KT-Dirichlet `Dir(½, …, ½)` per-context posterior predictive `(c_x + ½) / (n + A/2)` — minimax-optimal under log-loss on memoryless sources (Xie-Barron 1997); log-sum-exp stable internal-node mixing `log P_w = log(½ exp(log P_KT) + ½ exp(log P_w(0s) + log P_w(1s)))` with lazy node expansion (O(min(Aᴰ, n·D)) memory); Volf-Willems 1998 switching-CTW with per-symbol switch rate α tracking non-stationary sources (piecewise-stationary tree mixtures); Willems-Shtarkov-Tjalkens 1993 Context-Tree Maximisation (CTM) MAP-tree via `max` recursion under the same prior — returns the interpretable variable-order Markov model with highest posterior weight; plug-in entropy-rate estimator `Ĥ_n = -log₂ P_w(x₁ⁿ)/n` consistent for any stationary ergodic source (Cover-Thomas 1991 §13); anytime-valid e-process `e_T = Aᵀ · P_w(x₁ᵀ)` for `H₀: x_t iid Uniform({0,…,A-1})` (Ville's inequality; Vovk-Wang 2021 *E-values*) — reject at level α whenever e ≥ 1/α under any data-dependent stopping rule; greedy / argmax `k`-step continuation via non-destructive snapshot-restore rollout; tamper-evident SHA-256 fingerprint chain over every observe / predict / select / report event for AttestationLedger replay; thread-safe re-entrant lock; pure stdlib — math.log / math.exp / hashlib, no NumPy / SciPy / dependencies); the *give-me-the-universal-predictor-over-an-exponential-model-class-with-an-O(log-n)-redundancy-certificate* primitive — the **non-parametric prediction** companion to Forecaster (calibrated parametric forecasts), Hedger (universal aggregation over a finite expert pool), and Compressor (MDL model-class selection from a finite catalog) — implements the *universal-predictor half* of MC-AIXI-CTW (Veness-Ng-Hutter-Uther-Silver 2011 *A Monte-Carlo AIXI Approximation*), the most credible AGI approximation; composes with Forecaster (CTW predictive distribution IS a calibrated forecast — PIT-uniformity tests apply to ranks; log-loss per symbol IS the proper-scoring-rule score), Compressor (CTW code length IS prequential MDL for the variable-order-Markov class — the universal codelength against which Compressor's finite catalog is benchmarked), Hedger (register multiple Predictors of different depth as experts; AdaHedge picks the best D online — universal predictor of universal predictors), Abductor (each hypothesis's CTW codelength IS the marginal log-likelihood used in the Bayes-factor ratio — no closed-form needed), DriftSentinel (running CTW log-loss is a martingale under correct model spec — CUSUM detects regime change; switching-CTW α-rate IS the prior changepoint frequency), Refuter (CTW e-process refutes uniformity / i.i.d. / fixed-depth-d Markov nulls with anytime-valid p-values), Filterer (CTW on discretised innovations is a model-free residual test for state-space mis-specification), Reasoner (CTW MAP tree = variable-order Markov rules to be encoded into a Horn program for symbolic queries on regularities), Sampler (the predictive distribution IS the proposal in Sequential Monte Carlo on discrete symbol streams), ActiveInferencer (CTW IS a learned generative observation model whose log-likelihood plugs into expected free energy — closes a planner-predictor loop), AttestationLedger (every observe / predict / report hash chains into the ledger so an auditor can replay the prediction trace byte-for-byte), Strategist (Strategist picks among Predictor configurations whose universal redundancy bounds set the *worst-case* regret in the risk-adjusted score), PrivacyAccountant (DP-noisy symbols additively widen the KT log-loss by `log A · ε` per release — odometer advances accordingly), Coordinator (every Goal whose execution is sequential and observed as a symbol stream — tool-call success/fail, anomaly streams, log streams, plan-step outcomes — routes through Predictor.observe / .predict so the coordination engine has a calibrated belief over what comes next, with anytime-valid certificates the compliance officer can sign before action)
  compressor.py     # Compressor — Minimum Description Length hypothesis selection as a runtime primitive (refined-MDL Normalized Maximum Likelihood (Shtarkov 1987; Rissanen 1996) with exact Shtarkov-sum log C_n for Bernoulli (closed form), Multinomial-of-k (Mononen-Myllymäki 2008 O(k) recurrence), Geometric / Poisson / Gaussian-known-σ / Gaussian-unknown-σ / Histogram / Markov-of-order-r — luckiness-NML with coordinator-supplied bounded parameter ranges for non-compact parameter spaces (Grünwald 2007 Ch. 7); universal codes for the positive integers — Elias-γ / Elias-δ (Elias 1975) and Rissanen 1983 log* with universal constant c₀ = 2.865064; classical two-part Rissanen 1978 MDL with the (k/2) log n optimal-precision parameter code as a sanity-check against NML; prequential / sequential Dawid 1984 plug-in codes — Krichevsky-Trofimov 1981 (½, ½)-Dirichlet for binary and multinomial sequences (minimax regret matches NML to O(1)), Laplace 1814 rule (Dirichlet 1) as a textbook baseline, normal-inverse-gamma Bayesian Student-t mixture for the Gaussian families; Schwarz 1978 BIC and Akaike 1974 AIC for cross-method consistency checks; Vovk 1990 strong-aggregating-algorithm per-symbol regret bound on the runner-up (free-of-distribution); pairwise Bayes-factor comparison plus Stone 1974 / Geisser 1975 leave-one-out cross-check; anytime-valid online observation — every per-symbol call returns the prequential codelength increment, accumulates into a running total that matches the batch KT bit-exactly, and stays valid at every stopping time; tamper-evident SHA-256 fingerprint chain (genesis ``compressor.v1.genesis``) hashing every register / fit / score / select / observe / compare / report event so an auditor can replay the model-selection trace byte-for-byte; thread-safe re-entrant lock; pure stdlib — math.lgamma / math.log / math.exp / hashlib; the *which-model-class-itself-is-best-supported-by-the-data* primitive — the **meta-decision** that no other primitive in the runtime supplies: Solomonoff 1964's compression-equals-induction thesis, Rissanen 1978's MDL, Hutter 2005's universal AGI, behind one API — composes with Sampler (Compressor picks the model class, Sampler simulates the posterior in it), Forecaster (Compressor monitors prequential codelength → triggers re-fit on misspec), DriftSentinel (Compressor's rolling-window codelength IS the drift statistic), Refuter (codelength gap → Bayes factor → reject/accept "model M is best"), Reasoner (Compressor scores competing boolean encodings of a structured constraint), Composer (Compressor ranks candidate plan structures by joint MDL of formula + outcomes), PrivacyAccountant (codelength releases under DP, additive composition over streams), AttestationLedger (every codelength event canonicalised SHA-256), Strategist (which model class to pivot to → MDL-best registered candidate)
  hedger.py         # Hedger — universal prediction with experts / online learning with provable regret as a runtime primitive (Hedge / EWA / Multiplicative Weights (Vovk 1990 *Aggregating strategies*; Littlestone-Warmuth 1994; Freund-Schapire 1997) with Vovk-1990 minimax-optimal η = √(8 log N / T) — R_T ≤ √(T log N / 2); AdaHedge (de Rooij-van Erven-Grünwald-Koolen 2014 *Follow the Leader if you can, hedge if you must*) parameter-free adaptive learning rate η_t = log N / Δ_{t-1} with cumulative mixability gap Δ_t — R_T ≤ 2√(V_T log N) + O(log N) second-order; NormalHedge (Chaudhuri-Freund-Hsu 2009) anytime parameter-free with per-rank regret R_T(d) ≤ √(2 T (log(d+1) + log N)); Squint (Koolen-van Erven 2015 *Second-order quantile methods for experts*) improper-prior aggregation with second-order K-quantile regret, closed-form integral over η ∈ [0, 1/2] evaluated via 64-point Simpson + log-max stabilisation; ML-Prod / Prod (Cesa-Bianchi-Mansour-Stoltz 2007 *Improved second-order bounds*) polynomial-weighted with R_T ≤ √(8 V_T log N) + 5 log N; FTRL-Entropy (= Hedge) and FTRL-L2 (= projected OGD) with Wang-Carreira-Perpinan 2013 linear-time exact simplex projection; FTPL (Hannan 1957 *Approximation to Bayes risk in repeated play*; Kalai-Vempala 2005) with IID exponential perturbations for combinatorial action spaces, Monte-Carlo-estimated, replay-deterministic given seed; Online Mirror Descent with entropic mirror map (Beck-Teboulle 2003); BOA (Wintenberger 2017 *Optimal learning with Bernstein online aggregation*) per-expert Bernstein-tilted second-order η_i = 1 / (2 (1 + log(1+V_i))) with bound R_T ≤ √(2 V_T (1 + log N)) + 2(1 + log N); specialists / sleeping experts (Freund-Schapire-Singer-Warmuth 1997) via observe_partial; PAC-Bayes regret bound (McAllester 1999; Catoni 2007) against arbitrary reference distribution; Howard-Ramdas-McAuliffe-Sekhon 2021 anytime-valid confidence sequences on every per-expert mean loss; Maurer-Pontil 2009 empirical-Bernstein LCB / UCB; Hoeffding 1963 distribution-free LCB / UCB; realised KL(w_t ‖ π_0) in nats; tamper-evident SHA-256 fingerprint chain (genesis ``hedger.v1.genesis``) over every predict / select / observe so AttestationLedger replays the trace byte-for-byte; thread-safe re-entrant lock; pure stdlib — math.log / math.exp / math.erf / hashlib); the *combine-any-K-expert-recommendations-and-prove-the-vanishing-regret-against-the-best-one-in-hindsight* primitive — the universal **online-learning** aggregator that turns the runtime's pool of decision primitives (Bandit / BayesOpt / Arbiter / Strategist / Forecaster / Quantilizer) into a single meta-decision whose cumulative loss tracks the best-fixed-primitive-in-hindsight up to O(√(T log N)) without any distributional assumption; composes with Bandit / BayesOpt / Arbiter / Strategist (register each as an expert; Hedger.select() picks the right primitive at runtime, with bounded regret), Forecaster (log-loss aggregation gives constant regret R_T ≤ log N — universal predictor — applicable to ensembles of probabilistic forecasters under any proper scoring rule), PolicyImprover (PAC-Bayes regret bound becomes HCPI-style safety gate), Quantilizer (q-quantilize over Hedger weights to bound KL from a safe-expert baseline; cost amplification 1/q caps the regret/safety trade-off), DriftSentinel (AdaHedge mixability gap δ_t is a martingale drift signal — CUSUM on δ_t detects regime change), Refuter (per-expert anytime confidence sequence refutes dominance claims at any stopping time), Composer (a Plan-level Hedger lets the coordinator hedge over candidate Plans with composed reliability bounds; Hedger's KL bound sets the safety constant in Composer's PAC certificate), AttestationLedger (every predict / select / observe chain-hashes), Coordinator (every Goal whose execution picks among candidate primitives / model versions / prompts / tools is routed through Hedger.select() — the coordination engine learns at runtime which primitive to call, with anytime-valid regret certificates the compliance officer can sign before action)
  quantilizer.py    # Quantilizer — safety-bounded optimisation as a runtime primitive (hard exact discrete q-quantilizer (Taylor 2016 *Quantilizers: A Safer Alternative to Maximizers for Limited Optimization*), top-K quantilizer with deterministic SHA-256 tie-break, soft Boltzmann / Gibbs quantilizer with KL budget solved by bisection so KL(π_β ‖ b) = B exactly, sample-based empirical quantilizer with Massart-DKW 1990 finite-sample band on the (1-q)-quantile, Hoeffding 1963 / Maurer-Pontil 2009 empirical-Bernstein / Howard-Ramdas-McAuliffe-Sekhon 2021 anytime-valid LCB / UCB on the expected utility under the quantilizer, Taylor 2016 hidden-cost amplification UCB (E_q[c] ≤ E_b[c] / q), exact KL bound log(1/q), TV bound 1 − q, Pinsker / Bretagnolle-Huber 1979 / Le Cam derived divergence bounds, deterministic JSON-canonical SHA-256 fingerprint chain (genesis ``quantilizer.v1.genesis``) so every selection / quantilization / observation hashes into a replay-verifiable receipt for AttestationLedger, thread-safe re-entrant lock, pure stdlib — Beasley-Springer-Moro 1995 inverse-Φ, math.log / math.exp / hashlib); the *give-me-a-Goodhart-resistant-optimiser-and-prove-the-KL-bound-on-the-policy-deviation* primitive — the **safety** companion to Bandit / BayesOpt / Arbiter (cumulative-regret + PAC best-arm) and PolicyImprover (CRM-optimised policy) that bounds how far an optimiser may drift from a safe base under reward misspecification (Manheim-Garrabrant 2018 Goodhart variants); composes with Bandit (wrap select_arm in Quantilizer.select for safe exploration with KL-budget log(1/q) above the bandit's own policy; cost amplification 1/q sets the worst-case regret/safety trade-off), BayesOpt (q-quantilize EI selections to bound KL from a safe-prior Gaussian-process acquisition policy), Arbiter (the safety wrapper that converts an asymptotic best-arm-identification answer into a KL-bounded one), PolicyImprover (KL-bounded safe-improvement step: soft_quantilize(deployed_policy, CRM_score, kl_budget) lands exactly on the budgeted frontier — log(1/q) becomes the safety constant in the HCPI Bernstein-LCB gate), Persuader (q-quantilize over signal schemes bounds information design's KL from a truthful disclosure baseline), Strategist (risk-adjusted, KL-bounded meta-decision over recommendations), Refuter (adversarial search becomes a quantilizer when the falsification budget needs cost-amplification bound), Sampler (consume MCMC draws into quantilize_samples; Sampler's PSRF/ESS diagnostics gate the chain), Forecaster (PIT-calibrated quantilizer over calibrated predictions; Brier loss + cost amplification = decision-theoretic risk), DriftSentinel (a sudden change in the realised (1-q)-quantile threshold IS a drift signal on the base distribution), AttestationLedger (every Selection chain-hashes including the cryptographic commit to the base distribution, proxy utility, q, and seed), PrivacyAccountant (quantilization is post-processing of b — DP guarantee on b transfers verbatim to the quantilizer with no additional ε spent), Coordinator (every Goal whose execution chooses among candidate plans / prompts / models / tools is safety-budgeted by routing the candidate distribution through Quantilizer before action)
  filterer.py       # Filterer — Bayesian state-space filtering as a runtime primitive (Kalman 1960 linear-Gaussian Kalman filter with Joseph-form Bucy-Joseph 1968 stabilised covariance update / Information Filter (Maybeck 1979 §7.3) inverse-covariance dual / Potter 1963 / Bierman 1977 square-root Kalman; Extended Kalman Filter (Smith-Schmidt-McGee 1962; Anderson-Moore 1979 §8) with analytical Jacobian linearisation; Unscented Kalman Filter (Julier-Uhlmann 1997 *A new extension of the Kalman filter to nonlinear systems*) with scaled symmetric (2n+1) sigma-points, third-order Taylor-exact under affine transforms, Jacobian-free; Sequential Importance Resampling particle filter (Gordon-Salmond-Smith 1993 *Novel approach to nonlinear/non-Gaussian Bayesian state estimation*), Auxiliary Particle Filter (Pitt-Shephard 1999 *Filtering via simulation*) with one-step-lookahead pre-resampling, Bootstrap filter; Rauch-Tung-Striebel 1965 backward-sweep linear-Gaussian smoother; Carpenter-Clifford-Fearnhead 1999 systematic / Kitagawa 1996 stratified / Liu-Chen 1998 residual / multinomial resampling — all single-tier optimal); exact log-marginal likelihood via the Gaussian innovation decomposition for the Kalman family, importance-weight log-sum-exp for SMC (prequential MDL ⇒ composes with Compressor for state-space model selection); Bar-Shalom-Li-Kirubarajan 2001 normalised-innovation-squared (NIS) anytime χ² model-misspecification test; Wilson-Hilferty 1931 χ²(m) approximation for the NIS threshold; Box-Pierce 1970 innovation whiteness statistic; Crisan-Doucet 2002 *A survey of convergence results on particle filtering methods for practitioners* O(1/N) finite-sample MSE bound on bounded test functions; Kong-Liu-Wong 1994 effective-sample-size degeneracy diagnostic with auto-resample at ESS < N/2; Massart-DKW 1990 finite-sample distribution-free CDF band on the filtered marginal; tamper-evident SHA-256 fingerprint chain (genesis ``filterer.v1.genesis``) over every predict / update / resample / smooth event; thread-safe re-entrant lock; pure stdlib — list-of-lists matrix ops, Cholesky with adaptive jitter on numerical PD-failure, Joseph-form covariance update for numerical stability, Beasley-Springer-Moro 1995 inverse-Φ, no NumPy / SciPy; the *give-me-the-Bayesian-belief-over-the-latent-state-given-everything-I-have-observed-so-far* primitive — the **belief-update** foundation onto which every other decision primitive composes when the underlying world is sequential and partially observable — composes with ActiveInferencer (filtered posterior IS the belief over POMDP state — drop-in for the expected-free-energy planning step), Forecaster (the one-step predictive ``Filterer.predict()`` IS a calibrated forecast; PIT-uniformity tests compose with Forecaster's calibration e-process), Sampler (particle filter IS sequential importance sampling; Sampler's PWM Pareto-k tail diagnostic applies directly to the weight distribution; ADVI fits state-space hyperparameters on top), Compressor (the prequential log-marginal ``log p(y_t | y_{1:t-1})`` IS the MDL codelength of the observation stream under the supplied state-space model — Compressor selects the model class), Hedger (register competing state-space models as experts; negative log predictive is the per-step loss; AdaHedge learns the regime), DriftSentinel (standardised innovations are a martingale-difference under correct specification; CUSUM on NIS detects breaks; BOCPD localises change points), CausalDiscoverer (Filterer is the E-step in dynamic structural causal models — Murphy 2002), Refuter (refute the white-noise innovation assumption via QuickCheck-style metamorphic sample-path stress on autocorrelation bias), AttestationLedger (every predict / update / resample hash chains into the ledger so an auditor can replay the filtering trace byte-for-byte), Strategist (risk-adjusted decisions consume the filtered (mean, covariance) or particle approximation as the canonical belief input), PrivacyAccountant (DP-noisy observations widen R by 2σ²_DP; the (ε, δ) odometer advances on every update — composes with the runtime's regulatory mechanism), Coordinator (every Goal whose execution is sequential and partially-observable routes through Filterer.update() — the coordination engine maintains a calibrated belief over latent state, with anytime-valid receipts the compliance officer can sign before action)
  composer.py       # Composer — typed, certified compositional planning as a runtime primitive (classical STRIPS / ADL with conjunctive preconditions and add/delete-list effects (Fikes-Nilsson 1971; Pednault 1989) over a typed registry of operators; A* (Hart-Nilsson-Raphael 1968) with consistent admissible heuristics — h_zero (Dijkstra), h_goal_count, and h_landmark (HSP-style cheapest-add-list achiever, Helmert-Domshlak 2009) — over a state-space graph whose g-function is operator cost plus negative-log mean reliability; IDA* (Korf 1985) iterative-deepening for memory-bounded deep search; STRIPS goal regression (Fikes-Nilsson 1971; Bonet-Geffner 2001) for dense-operator / small-goal regimes; Tarjan 1972 SCC + Kahn 1962 topological sort on the predicate-flow graph to diagnose cyclic operator registrations; monomorphic Hindley-Milner unification (Robinson 1965; Milner 1978) on parameter and dataflow types with first-order substitution and the standard occurs-check; per-operator Beta-Bernoulli reliability posterior (Bayes 1763 / Laplace 1814) updated by ``observe()`` with a configurable prior (mean × strength or raw α/β); end-to-end PAC certificate composing per-step Clopper-Pearson 1934 lower bounds (closed form α^{1/n} at k=n, regularised incomplete beta inversion via stdlib continued fraction otherwise), Garivier-Cappé 2011 KL-inverse upper / lower confidence bounds, Maurer-Pontil 2009 empirical-Bernstein, and Hoeffding 1963 — Bonferroni-corrected across plan length — under both INDEPENDENT (product) and WORST_CASE (union-bound) regimes; Catoni 2007 PAC-Bayes lower bound on the average reliability of a posterior over operator choices; tamper-evident SHA-256 fingerprint chain (genesis ``composer.v1.genesis``) hashing every register / axiom / plan / verify / observe / execute event so an auditor can replay the planning trace byte-for-byte; pure stdlib — heapq priority queue, recursive descent type parser, JSON-canonical event payloads; the *plan-and-prove-the-bound-on-the-plan* primitive — the **planning** companion to Reasoner (deterministic SAT/Horn/ASP), Synthesizer (PBE program search), and Refuter (PAC falsification) — composes with Reasoner (register Reasoner.solve as a feasibility-gate operator), Refuter (register Refuter.refute as a PAC-gate operator before any downstream consumer), Bandit / BayesOpt / Arbiter (decision-theoretic operators whose own per-pull outcomes feed Composer.observe), Synthesizer (Composer plans over Synthesizer's DSL operators; Synthesizer fills in any unknown leaf), PrivacyAccountant (advanced composition over per-operator ε contributions reported alongside the reliability bound), AttestationLedger (every certificate and observation hash chains into the ledger), Coordinator (the natural target — every Goal compiles to a Plan, every PlanStep is a primitive call), Cartographer (curriculum step → ``operator`` registration so Cartographer's ready/1 predicate gates plan emission)
  intender.py       # Intender — inverse reinforcement learning / preference-based reward inference as a runtime primitive (MaxEnt IRL (Ziebart-Maas-Bagnell-Dey 2008 *Maximum entropy inverse reinforcement learning*; Ziebart 2010 thesis) with closed-form feature-matching gradient ``∇L(θ) = μ̂_E − E_{π_soft(θ)}[φ]`` and concave L2-penalised log-likelihood — converges to the unique global optimum; Bayesian IRL (Ramachandran-Amir 2007 *Bayesian inverse reinforcement learning*) random-walk Metropolis-Hastings on Boltzmann-rationality likelihood under Gaussian prior, Roberts-Rosenthal 2009 adaptive proposal scale targeting the 0.234 acceptance optimum, Geweke 1992 two-window z-score stationarity diagnostic before reporting credible regions; preference-based reward learning (Christiano-Leike-Brown-Martic-Legg-Amodei 2017 *Deep reinforcement learning from human preferences*; Bradley-Terry 1952) convex negative-log-likelihood ``−Σ log σ(β θᵀ (Φ(τ_w) − Φ(τ_l)))`` with L2 regularisation; max-margin apprenticeship learning (Abbeel-Ng 2004 *Apprenticeship learning via inverse reinforcement learning*) unit-L2 projection step; soft Q-iteration (Haarnoja-Tang-Abbeel-Levine 2017) inner solver returning the soft Q-function, soft value, and stochastic policy ``π_soft(a | s) ∝ exp(Q(s, a))``; behavioural cloning (Pomerleau 1989 *ALVINN*) α-Laplace-smoothed empirical state-conditional action policy as baseline; identifiability bound (Cao-Cohen-Szepesvári 2021 *Identifiability in inverse reinforcement learning*) rank/nullity/conditioning of the feature matrix — the dimensions of reward space the data cannot distinguish; KL(π_soft ‖ π_BC) as Quantilizer's safe-deployment KL budget; Howard-Ramdas-McAuliffe-Sekhon 2021 anytime-valid confidence sequence on held-out preference agreement; Maurer-Pontil 2009 empirical-Bernstein / Hoeffding 1963 finite-sample LCB / UCB on every aggregate statistic; PAC-Bayes regret bound (McAllester 1999) on the preference-learning loss against any reference prior on θ; tamper-evident SHA-256 fingerprint chain (genesis ``intender.v1.genesis``) over every observe / fit / preference / report event so AttestationLedger replays the inference trace byte-for-byte; thread-safe re-entrant lock; pure stdlib — list-of-lists matrices, log-sum-exp with explicit max-subtraction, ``random.Random(seed)`` for full reproducibility, no NumPy / SciPy; the *learn-what-the-user-actually-values-from-observed-behaviour-with-an-identifiability-bound* primitive — the **preference-elicitation** kernel onto which the rest of the runtime composes when reward is not given — composes with ActiveInferencer (learned ``θᵀφ`` becomes the log-preference ``log P(o | C)`` in the active-inference generative model — closes the loop where the coordination engine learns user preferences before planning under them), Strategist (risk-adjusted action selection consumes ``E[r(s, a)]`` from Intender's posterior with credible region as the uncertainty input), Quantilizer (Intender's ``KL(π_soft ‖ π_BC)`` IS the safe-deployment KL budget; quantilize on the learned soft policy → certified not-too-different-from-expert), Bandit / BayesOpt (pointwise reward queries on novel (s, a) consume ``θ̂ᵀφ(s, a)`` from MAP or BIRL posterior mean; acquisition functions read posterior variance for Thompson sampling and UCB), Composer (plans whose terminal value is ``θᵀφ`` get parameterised by the posterior — Composer's PAC certificate carries Intender's identifiability bound forward), Ranker (Ranker fits a ranking, Intender fits a reward — they compose; Ranker's pairwise comparisons feed Intender as preference observations, Intender's reward feeds Ranker as item utility), Mechanism / Persuader (both require a model of the receiver's utility; Intender supplies a learned one from observed behaviour), PolicyImprover (Intender supplies the reward, PolicyImprover deploys safely under HCPI — end-to-end RLHF pipeline), Refuter (refute candidate rewards via QuickCheck-style stress on the feature-matching residual), DriftSentinel (per-trajectory log-likelihood under the fitted reward is a martingale-difference under the null "no preference drift" — CUSUM detects user-preference shifts), AttestationLedger (every observe / fit / preference / report event hash chains into the ledger so an auditor can replay the inference trace byte-for-byte), Coordinator (every Goal that requires aligning to user behaviour routes through Intender — the coordination engine learns what users want from demonstrations and preferences, with anytime-valid certificates the compliance officer can sign before action)
  speculator.py     # Speculator — speculative execution as a runtime primitive (Speculative Sampling (Chen-Borgeaud-Irving-Lespiau-Sifre-Jumper 2023 *Accelerating Large Language Model Decoding with Speculative Sampling* — per-position accept-with-prob ``min(1, p_target(x)/q_draft(x))``, rejection-resample from residual ``(p−q)₊ / ∑(p−q)₊`` to recover target marginal exactly; provably equivalent to sampling i.i.d. from p_target) / Leviathan-Kalman-Matias 2023 *Fast Inference from Transformers via Speculative Decoding* — the originating algorithm, equivalent to spec-sampling for stochastic targets; greedy variant accepts only when draft argmax matches target argmax / Greedy verification (argmax target; strongest when target distribution is peaky) / Medusa-style tree decoding (Cai-Li-Geng-Peng-Lee-Chen-Dao 2024 *Medusa: Simple LLM Inference Acceleration with Multiple Decoding Heads* — draft proposes a token tree, target verifies in one shot, accept longest sampling-equivalent prefix) / Self-speculative early-exit decoding (Zhang-Yang-Sun et al. 2024 *Draft and Verify: Lossless Large Language Model Acceleration via Self-Speculative Decoding* — same model as draft and target, skipping ``skip_fraction`` of internal layers during draft) / EAGLE (Li-Wei-Zhang-Zhang 2024 *EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty* — extrapolation-tuned draft, identical acceptance test) / Lookahead decoding (Fu-Bansal-Beltagy et al. 2024 *Lookahead Decoding* — n-gram cache populated from past acceptances skips the draft-model call entirely on cache hits)); statistical certificates per report (acceptance-rate LCB via Hoeffding 1963 / Maurer-Pontil 2009 empirical-Bernstein / Howard-Ramdas-McAuliffe-Sekhon 2021 anytime-valid confidence sequences; speedup LCB on ``E[accepted_tokens + 1 per target call] ∈ [1, K+1]``; equivalence martingale ``log p_target(emitted)`` mean with Maurer-Pontil LCB — under correct implementation a martingale-difference with mean 0, a significant drift indicates implementation bug or rejection-sampling violation); tamper-evident SHA-256 fingerprint chain (genesis ``speculator.v1.genesis``) with optional HMAC-SHA-256 over every step / report / reset event so AttestationLedger replays the speculative trace byte-for-byte; thread-safe re-entrant lock; transport-agnostic — operates on caller-supplied ``draft(state) -> [(token, dist), …]`` and ``target(state, tokens) -> [(token, dist), …]`` callables so it accelerates LLM token decoding, plan-step execution (Distiller-fit draft + Searcher target), tool dispatch (cached draft + canonical target), and retrieval pipelines identically; pure stdlib — math.log / math.exp / random / hashlib, no NumPy / PyTorch / model runtime; the *run-cheap-then-verify-and-prove-the-speedup-while-preserving-output-equivalence* primitive — the **runtime-acceleration** companion to Distiller (which produces the draft callable for any decision primitive) and Searcher (which can be the expensive target verifier) — composes with Distiller (Distiller-fit policy is the draft; Speculator turns its amortised forward pass into runtime-level inference acceleration with target-equivalent output), Searcher (Searcher as target verifier; Distiller as draft; Speculator brackets both in a provably equivalent acceleration), Bandit / BayesOpt (read acceptance-rate UCB to decide when to refresh the draft), DriftSentinel (running acceptance rate CUSUM rolls back the draft on drift), AttestationLedger (every speculative step chain-hashes), PrivacyAccountant (advances on each verified call when the input is sensitive), Coordinator (every Goal whose execution emits a stream of atomic decisions routes through Speculator.step for runtime-level acceleration with provable output equivalence)
  aligner.py        # Aligner — direct preference optimisation as a runtime primitive (DPO (Rafailov-Sharma-Mitchell-Ermon-Manning-Finn 2023 *Direct Preference Optimization: Your Language Model is Secretly a Reward Model* — closed-form ``-log σ(β (Δ_θ(w) - Δ_θ(l)))`` with ``Δ_θ(x) = s_θ(x) - log π_ref(x)``, the no-reward-model bridge between RLHF and a single log-likelihood) / IPO (Azar-Rowland-Piot-Guo-Calandriello-Valko-Munos 2023 *A General Theoretical Paradigm to Understand Learning from Human Preferences* — squared-loss alternative ``(Δ_θ(w) - Δ_θ(l) - 1/(2β))²`` that avoids DPO's sigmoid saturation) / KTO (Ethayarajh-Xu-Muennighoff-Jurafsky-Kiela 2024 *KTO: Model Alignment as Prospect Theoretic Optimization* — asymmetric Kahneman-Tversky loss on *unary* desirability signals, no pairs required) / SLiC-HF (Zhao-Joshi-Liu-Khalman-Saleh-Liu 2023 *SLiC-HF: Sequence Likelihood Calibration with Human Feedback* — hinge loss + SFT regulariser ``max(0, δ - β (Δ_θ(w) - Δ_θ(l))) + λ (-log π_θ(w))``) / SimPO (Meng-Xia-Chen 2024 *SimPO: Simple Preference Optimization with a Reference-Free Reward* — reference-free length-normalised ``-log σ(β (s_θ(w)/|w| - s_θ(l)/|l|) - γ)``) / ORPO (Hong-Lee-Thorne 2024 *ORPO: Monolithic Preference Optimization without Reference Model* — single-stage SFT + odds-ratio penalty) / cDPO (Mitchell 2023 conservative DPO — label-smoothed ``(1-ε) L(m) + ε L(-m)``) / rDPO (Chowdhury-Kini-Natarajan 2024 *Provably Robust DPO* — closed-form unbiased noise correction under symmetric flip rate ε < 0.5 recovering noise-free MLE); scoring models (linear hash-feature scorer with Weinberger et al. 2009 feature hashing / low-rank bilinear u(p)ᵀ V x with learned U, V matrices / pass-through identity for caller-supplied LLM log-probabilities); optimisers (AdamW with Loshchilov-Hutter 2019 decoupled weight decay + bias-corrected moments / SGD with Nesterov 2013 momentum / Crammer et al. 2006 passive-aggressive PA-II for streaming); Vitter 1985 reservoir buffer with capped capacity; Brent-style line-search temperature scaling (Guo et al. 2017) for post-hoc calibration; pool-adjacent-violators isotonic regression (Brunk et al. 1972) for non-parametric calibration; eval-gated AlphaZero-style deployment ladder (a fit promotes only if held-out preference-accuracy LCB beats deployed UCB); statistical certificates (Howard-Ramdas-McAuliffe-Sekhon 2021 anytime-valid HRMS confidence sequence + Maurer-Pontil 2009 empirical-Bernstein + Hoeffding 1963 LCB / UCB on preference accuracy + Bernstein CI half-width on KL(π_θ ‖ π_ref) + Tolstikhin-Seldin 2013 PAC-Bayes-Bernstein generalisation bound + Vovk-Wang 2021 anytime e-process on agreement-with-judge test under H₀ uniform); tamper-evident SHA-256 fingerprint chain (genesis ``aligner.v1.genesis``) with optional HMAC-SHA-256 over every observe / fit / calibrate / deploy / reject event so AttestationLedger replays the alignment trace byte-for-byte; thread-safe re-entrant lock; pure stdlib — list-of-lists matrices, log-sum-exp numerically-stable sigmoid / softplus, hashlib SHA-256, no NumPy / SciPy / PyTorch / Hugging Face; the *learn-a-preference-aligned-policy-directly-from-pairs-or-thumbs-and-bound-its-KL-drift* primitive — the **policy-shipping** dual of Intender (which learns a *reward* from preferences) and the canonical RLHF stage-2 replacement that ships everywhere because it needs no PPO loop and no reward-model side-train — composes with Intender (Intender fits the reward; Aligner skips it and ships the policy from the same preference stream), Ranker (Bradley-Terry inference dual: Ranker ranks fixed items, Aligner parameterises a policy that generates new ones), Quantilizer (Aligner.kl_divergence_to_reference IS the KL budget; quantilize on Aligner's softmax policy → certified not-too-different-from-reference at deployment), Bandit / BayesOpt (register Aligner.preference_probability as a cheap proxy reward oracle for hyperparameter search; UCB / Thompson on the score function), Forecaster (PIT-calibrate the implied preference probabilities; Brier / log-loss machinery applies verbatim), Auditor (BH-control false-positive promotion across many simultaneous Aligner deployments), DriftSentinel (running preference-accuracy CUSUM rolls back the deployed model on drift), AttestationLedger (every observe / fit / deploy / reject event chains into the ledger), PrivacyAccountant (linear scorer is amenable to DP-SGD via the Gaussian mechanism — odometer advances per observation), Coordinator (every Goal whose execution selects among candidate completions routes through Aligner.best_of_n / .softmax_sample — the coordination engine learns from preferences and ships KL-bounded policies with anytime-valid receipts the compliance officer can sign before action)
  scheduler.py      # ParallelScheduler — DAG-aware parallel plan execution
  skillmine.py      # mine reusable skills from successful trace patterns
  skills.py         # markdown skill library with retrieval (procedural memory)
  reflection.py     # per-task lessons-to-memory loop (medium-timescale learning)
  world_model.py    # observed-entity tracker (file/url/command + outcomes)
  toolsynth.py      # sandboxed Python tool synthesis (subprocess isolated)
  tasks.py          # Task / TaskQueue / TaskRunner — scheduled work
  persistence.py    # checkpoint sessions to disk and rehydrate
  memory.py         # persistent JSONL memory store + namespacing (multi-tenant)
  reconciler.py     # Reconciler — Aumann agreement as a runtime primitive (Aumann 1976 *Agreements on Agreed* — two Bayesians with common-knowledge posteriors *must* agree; Geanakoplos-Polemarchakis 1982 *We Can't Disagree Forever* — finite-time convergence of the Aumann iteration on finite state spaces; Stone 1961 linear opinion pool ``q(·) = Σ_i w_i p_i(·)`` — externally-Bayesian when weights are belief-independent (Genest-McConway 1990); Bordley 1982 logarithmic opinion pool ``q(·) ∝ Π_i p_i(·)^{w_i}`` — log-linear aggregation, the maximum-entropy combination subject to matching each expert's KL-projection of the consensus (Genest-Zidek 1986); Bregman 1967 KL-barycenter / Cuturi-Doucet 2014 — iterative fixed-point minimum-divergence consensus; four aggregation methods — linear / logarithmic / aumann (iterative cognitive-economy approximation with round-cap returning closest-to-consensus KL-barycenter when cap fires) / kl_barycenter; per-source KL gap ``KL(p_i ‖ q)`` quantifies how surprising each expert's belief looks under the consensus — the largest gap names the *outlier* the coordinator should investigate; Massey 1951 one-sample Kolmogorov-Smirnov test on probability-integral-transform of realised outcomes per source with Stephens 1970 asymptotic correction; closed-form average log-loss per source for binary-outcome calibration where the PIT test is uninformative; Howard-Ramdas-McAuliffe-Sekhon 2021 anytime-valid confidence sequence on the per-outcome consensus mass; Maurer-Pontil 2009 empirical-Bernstein on consensus stability; inverse-Herfindahl-Hirschman effective number of experts ``1 / Σ w_i²`` — equals K for K equal-weight experts, falls to 1 when one source dominates; identifiability_report flags topics where every source assigns zero mass to some outcome (consensus cannot distinguish that outcome from a zero-mass alternative) and reports the effective number of independent contributors; tamper-evident SHA-256 fingerprint chain (genesis ``agi.reconciler.v1\x00 + secret_key``) with optional HMAC-SHA-256 over every register / contribute / consensus / calibration event so AttestationLedger replays the consensus byte-for-byte; export_state() / import_state() round-trip byte-identical chain head; thread-safe re-entrant lock; pure stdlib — list-of-lists arithmetic, log-sum-exp numerically-stable softmax, hashlib SHA-256, no NumPy / SciPy / PyTorch; the *aggregate-K-conflicting-posteriors-from-K-primitives-into-one-coherent-consensus-belief-with-a-replay-verifiable-receipt* primitive — the **consensus-belief kernel** onto which every primitive that emits a posterior composes when the runtime must reason from more than one source at once — composes with Bandit / BayesOpt / Imaginator / Forecaster / Predictor (each contributes its posterior to a Reconciler topic and the coordinator sees the calibrated consensus instead of any single primitive's belief), Auditor (Reconciler's per-source outlier KL is a candidate test statistic; Auditor BH-controls FDR across many simultaneous topics), DriftSentinel (running consensus stability is a martingale-difference under common knowledge; CUSUM flags contributor drift), Aligner (preferences over (topic, consensus) pairs become training data for the system's reward model), Mentalist (supplies the rationality posterior the coordinator weights each Mentalist-modelled counterparty's contribution by), Conformal (wraps the consensus pmf with a finite-sample prediction set), AttestationLedger (every register / contribute / consensus / calibration event hash-chains into the ledger), Coordinator (every Goal whose execution depends on more than one primitive's posterior routes through Reconciler — the coordination engine sees one calibrated belief plus the outlier name plus the anytime-valid CI plus the audit-chain head, instead of K conflicting posteriors)
  imaginator.py     # Imaginator — learned-world-model rollouts as a runtime primitive (Sutton 1990 *Dyna*; Kearns-Singh 2002 *Near-Optimal RL in Polynomial Time* — simulation lemma ``|V^π_M̂ − V^π_M| ≤ (γ/(1−γ)²) ε``; Strehl-Littman-Wiewiora 2009 PAC-MDP with sample-complexity ``O((SA/ε²(1−γ)⁴)·log(SAδ⁻¹))``; Strens 2000 / Osband-Russo-Van Roy 2013 PSRL with Bayesian regret ``O(τ √(SAT log T))``; Auer-Jaksch-Ortner 2010 UCRL2; Deisenroth-Rasmussen 2011 PILCO moment-matching; Janner-Fu-Zhang-Levine 2019 *When to Trust Your Model* short-horizon-rollout argument; Hafner-Lillicrap-Ba-Norouzi 2020 DreamerV3 imagined-trajectory policy optimisation); two conjugate dynamics families — discrete-state Dirichlet-multinomial transition + Normal-Gamma reward with closed-form Bayesian updates and Student-t reward predictive, and continuous-state matrix-normal-inverse-Wishart linear-Gaussian with Cholesky-via-Lentz analytic posterior mean dynamics ``[A | B]`` and per-horizon closed-form moment propagation; three rollout-sampling selectors — posterior_mean / Thompson (PSRL — one transition matrix per trajectory) / Bayes-averaged (Madigan-Raftery 1994 BMA over n_models posterior samples); imagine() bundles Monte-Carlo expected-return + std + Maurer-Pontil 2009 empirical-Bernstein LCB/UCB + Howard-Ramdas-McAuliffe-Sekhon 2021 anytime-valid confidence sequence + return quantiles + per-horizon state quantiles + full trajectories; value_iteration() — closed-form DP on posterior-mean transition/reward; thompson_policy() — PSRL one-sample plan-act-repeat; pac_value_bound() — Kearns-Singh simulation-lemma PAC bound composed with per-(s,a) Hoeffding transition radius; required_samples_for_pac() — invert PAC bound to Strehl-Littman-Wiewiora sample complexity; bayes_average_value() — Bayesian Model Averaging value estimate over n_models posterior dynamics; moment_rollout() — PILCO closed-form Σ_{h+1} = A Σ_h Aᵀ + Q linear-Gaussian moment propagation; identifiability_report() — Cao-Cohen-Szepesvári 2021 under-observed (s,a) pairs and per-pair effective Dirichlet concentration; pit_calibration() — Massey 1951 one-sample Kolmogorov-Smirnov test on probability-integral-transform of one-step rewards under Student-t predictive (Stephens 1970 asymptotic correction); tamper-evident SHA-256 fingerprint chain (genesis ``agi.imaginator.v1\x00 + secret_key``) with optional HMAC-SHA-256 over every register / observe / imagine / plan / certify event so AttestationLedger replays the imagined trajectory byte-for-byte from observation stream + RNG seed; export_state()/import_state() round-trip byte-identical chain head; thread-safe re-entrant lock; pure stdlib — list-of-lists matrices, Cholesky-via-Lentz solver, Marsaglia-Tsang gamma draws, hashlib SHA-256, no NumPy / SciPy / PyTorch; the *learn-a-dynamics-model-from-observed-transitions-imagine-with-calibrated-bounds-and-emit-a-replay-verifiable-receipt* primitive — the **model-based-RL inner loop** as a runtime primitive that lets a coordination engine route every Goal whose execution requires reasoning about future world states through `imagine → certify → act` with anytime-valid uncertainty bounds the compliance officer can sign before action — composes with Searcher (Searcher's tree search runs over Imaginator's posterior-predictive successor enumerator), ActiveInferencer (Imaginator supplies the generative model the EFE minimisation requires), Quantilizer (Imaginator.return_quantiles IS the distribution Quantilizer thresholds on — deploy the policy whose imagined return is in the top q-quantile), Distiller (distil the value_iteration policy into an amortised neural / linear policy), Planner (Imaginator's posterior-mean transition matrix is a PDDL-compilable operator schema; Planner solves SAT with the MAP transitions), DriftSentinel (per-step log-loss of one-step predictions is a martingale-difference under correct dynamics; CUSUM flags world drift), Bandit / BayesOpt (Thompson-sampled value is a cheap proxy oracle for hyperparameter search), Curator (Imaginator's identifiability report identifies under-observed (s,a) pairs Curator targets in the next curriculum batch), AttestationLedger (every register/observe/imagine/plan/certify event hash-chains into the ledger), Coordinator (every Goal whose execution requires reasoning over future world states routes through Imaginator — the coordination engine no longer hand-writes the dynamics function; it observes a few real transitions, registers them, and queries imagined value with calibrated uncertainty bounds the compliance officer can sign before action)
  mentalist.py      # Mentalist — Bayesian theory-of-mind as a runtime primitive (Premack-Woodruff 1978; Baker-Saxe-Tenenbaum 2009 *Action understanding as inverse planning*; Baker-Jara-Ettinger-Saxe-Tenenbaum 2017; Foerster-Chen-Al-Shedivat-Whiteson-Abbeel-Mordatch 2018 *Learning with opponent-learning awareness*); Dirichlet posterior over latent state distributions per agent with closed-form online conjugate update; MaxEnt IRL (Ziebart-Maas-Bagnell-Dey 2008) recovering utility weights θ such that ``π(a | s) ∝ exp(β · Q_θ(s, a))`` best explains the observed action stream — closed-form gradient descent with ℓ₂ regularisation and provable convergence to the unique MaxEnt fixed point; online Bayesian rationality estimation (Gamma prior on inverse-temperature β driven by predictive log-likelihood); Beta-Bernoulli capability posteriors per (action, state) with Clopper-Pearson 1934 exact credible intervals; four predict selectors — MAP / softmax-Boltzmann / Thompson sampling with ``O(√(T log T))`` regret against the best fixed agent / Bayes posterior-mean averaging (Madigan-Raftery 1994) minimising log-loss in expectation; rollout simulation under the posterior-mean Boltzmann policy for value-of-information queries; nested theory of mind (``nested_belief(observer="bob", target="alice", …)`` returns Bob's posterior over Alice's policy from Bob's observations alone); McAllester 1999 PAC-Bayes bound on held-out predictive log-loss; identifiability report (Cao-Cohen-Szepesvári 2021) on rank/nullity/conditioning of the feature matrix — the dimensions of utility space the data cannot distinguish; Howard-Ramdas-McAuliffe-Sekhon 2021 anytime-valid confidence sequences + Maurer-Pontil 2009 empirical-Bernstein + Hoeffding 1963 LCB / UCB on every aggregate statistic; tamper-evident SHA-256 fingerprint chain (genesis ``mentalist.v1.genesis``) with optional HMAC-SHA-256 over every register / observe / predict / infer / report event so AttestationLedger replays the mind-modelling trace byte-for-byte; thread-safe re-entrant lock; pure stdlib — list-of-lists matrices, log-sum-exp Boltzmann softmax, hashlib SHA-256, no NumPy / SciPy; the *give-me-a-calibrated-Bayesian-belief-over-what-the-counterparty-believes-wants-and-will-do-next* primitive — the **theory-of-mind kernel** onto which every multi-agent primitive composes when the runtime must reason *about* another mind rather than merely transact *with* it — composes with Negotiator (Mentalist supplies the counterparty utility posterior; Negotiator allocates fairly against it), Coalition (per-member Mentalist policy posteriors feed Shapley-value estimation), Mechanism / Persuader (both need a model of the receiver's utility — Mentalist supplies a Bayesian one from observed behaviour), Diplomat (cheap-talk inference reads Mentalist's nested belief over the other side's belief), Equilibrator (best-response dynamics use Mentalist's Boltzmann policy as opponent-action expectation), Intender (Intender learns the user's reward, Mentalist learns the *other agent's* reward — same MaxEnt IRL machinery, different reference frame), Aligner (deploy a Mentalist-predicted KL-budget against each modelled counterparty), Bandit / BayesOpt (Thompson sampling on Mentalist's posterior gives an opponent-model-aware exploration policy), DriftSentinel (per-step predictive log-likelihood is a martingale-difference under correct opponent modelling — CUSUM detects opponent-policy shifts), AttestationLedger (every register / observe / predict / infer event hash chains into the ledger), Coordinator (every Goal whose execution involves another mind routes through Mentalist — the coordination engine maintains a calibrated belief over what each counterparty believes, wants and will do, with anytime-valid receipts the compliance officer can sign before action)
  pareto.py         # Pareto — multi-objective optimization as a runtime primitive (Deb-Pratap-Agarwal-Meyarivan 2002 NSGA-II fast non-dominated sort with O(MN²) time / O(N²) memory + Deb 2002 §3.3 crowding distance on each rank for diversity-preserving tie-break; Zitzler-Thiele 1998 *Multiobjective optimization using evolutionary algorithms* hypervolume indicator with closed-form 2D sweep, Beume-Fonseca-López-Ibáñez-Paquete-Vahrenhold 2009 *HV by slicing objectives* exact 3D dimension-sweep, While-Hingston 2011 WFG-style dimension-sweep decomposition for M ∈ {4, 5}, inclusion-exclusion over axis-aligned boxes for M ≥ 6 with Monte-Carlo fallback at N > 14; Emmerich-Beume-Naujoks 2005 + Emmerich 2008 *Expected hypervolume improvement* closed-form for M=2 via Yang-Emmerich-Deutz-Bäck 2017 box-decomposition (each strip factorises into 0th and 1st partial Gaussian moments, finite sum), Monte-Carlo EHVI for M ≥ 3 with standard-error report; Steuer 1986 weighted-sum scalarisation / Tchebycheff scalarisation; Knowles 2006 ParEGO augmented Tchebycheff (Steuer-Choo 1983 augmentation breaks weak-Pareto degeneracy — every Pareto-optimal point becomes the strict argmin of *some* augmented Tchebycheff weight vector); Wierzbicki 1980 *The use of reference objectives in multiobjective optimization* achievement scalarising function (aspiration-point method); Das-Dennis 1998 + Zhang-Li 2007 *MOEA/D* penalty boundary intersection PBI = d₁ + θ·d₂; Das-Dennis 1998 uniformly-spaced weight grid C(p+M−1, M−1) on the M-simplex / Marsaglia 1972 Dirichlet(1, …, 1) uniform-simplex sampling via the Gumbel trick; Coello-Sierra 2004 inverted generational distance IGD + Van Veldhuizen 1999 generational distance GD + Schott 1995 spacing + Zitzler 1999 maximum-spread — each metric carries a Maurer-Pontil 2009 empirical-Bernstein half-width on its sample mean when computed from a sub-sample; Deb 2000 *An efficient constraint handling method for genetic algorithms* feasibility-first dominance for constrained problems (feasible dominates infeasible; infeasible compared by aggregate constraint violation); Howard-Ramdas-McAuliffe-Sekhon 2021 anytime-valid confidence sequence + Hoeffding 1963 LCB / UCB on the running hypervolume increment supplies the *stop-expanding-the-front-when-expected-progress-falls-below-ε* receipt; tamper-evident SHA-256 fingerprint chain (genesis ``agi.pareto.v1\x00 + secret_key``) with optional HMAC-SHA-256 over every register / observe / frontier / report / certify event so AttestationLedger replays every multi-objective decision byte-for-byte from the observation stream; snapshot() / restore() round-trip a byte-identical chain head so the coordination engine can hibernate the front, ship it to another host, and resume; thread-safe re-entrant lock; pure stdlib — list-of-lists arithmetic, math.erf-based normal CDF, hashlib SHA-256, no NumPy / SciPy / PyTorch; the *give-me-the-Pareto-rank-1-layer-plus-EHVI-for-the-next-evaluation-plus-an-anytime-valid-progress-certificate-replay-verifiable-by-the-compliance-officer* primitive — the **multi-objective decision kernel** every product story needs (drug discovery weighs affinity vs synthesis vs toxicity, infra weighs accuracy vs latency vs cost, negotiation weighs outcome value vs fairness) — composes with BayesOpt (EHVI acquisition picks the next candidate when the surrogate emits a Gaussian posterior over each objective — multi-objective Bayesian optimisation = BayesOpt with Pareto's EHVI in the acquisition slot), Bandit (register a Tchebycheff or augmented-Tchebycheff scalarisation as the reward channel; sweep the Das-Dennis weight grid to expose the front — every weight vector becomes one bandit instance), Searcher (every tree-search leaf becomes a Pareto candidate with an M-objective cost; the search returns the *Pareto layer* not the argmax — coordination over Pareto-rank-1 plans), PortfolioOptimizer (Pareto sorts (return, risk) candidates before allocating a fixed budget across them — Pareto-rank-1 is the efficient frontier in the Markowitz sense), Strategist (fuses calibration + conformal + causal + OPE on *each objective* and returns the Pareto-rank-1 panel to the coordination engine — risk-adjusted multi-objective recommendation), Coalition (multi-criterion Shapley value: one Pareto rank per criterion, then aggregated), Negotiator (Kalai-Smorodinsky and Nash bargaining are *exactly* the Tchebycheff and weighted-product scalarisations on the disagreement-shifted objective space — Pareto.frontier() bounds the bargaining set), Quantilizer (Pareto-rank-1 ∩ top-q quantile on a scalarisation gives a doubly-certified deliverable panel), Reconciler (when K primitives emit posteriors over M objectives, Reconciler aggregates per-objective and Pareto sorts the consensus point cloud), DriftSentinel (running hypervolume CUSUM rolls back the deployed front when its progress signal regresses), AttestationLedger (every register / observe / frontier / hv / ehvi / certify event hash-chains into the global audit ledger), Coordinator (every Goal whose execution must trade off more than one objective routes through Pareto for a rank-1 candidate panel + a calibrated EHVI to spend the next evaluation budget on + an anytime-valid stop signal the compliance officer can sign before action)
  costs.py          # per-turn + cumulative token usage and $ tracking
  tools.py          # builtin tools: file, shell, web, memory (+ world auto-record)
  __main__.py       # CLI: python -m agi
learner/            # learning track — small open base + LoRA loop
  critic.py         # trace-quality critic (small MLP, trains on CPU)
  traces.py         # append-only JSONL trace logger
  filter.py         # quality gates: eval-pass, score threshold, thumbs
  goals.py          # Goal abstraction; Addition is the first concrete goal
  synth.py          # synthetic labeled data for critic warm-start
  train.py          # LoRA SFT script (HF transformers + PEFT, GPU)
evals/
  tasks.jsonl       # eval tasks (math, file ops, recall, search)
  run.py            # eval runner
tests/              # 1700+ unit tests, all run without an API key
ARCHITECTURE.md     # full design — read this for direction
PLAN.md             # stage roadmap
```

## Setup

```sh
pip install -e .
export ANTHROPIC_API_KEY=...
```

## CLI

```sh
python -m agi                          # interactive REPL
python -m agi "summarize ./README.md"  # one-shot
python evals/run.py                    # run the eval suite
python -m agi.server --port 8765       # start HTTP runtime
```

## Coordinator — the reference driver

The `Coordinator` is a reference implementation of a coordination engine
sitting *above* the Runtime. It accepts a `Goal` (declarative intent +
budget), runs it through a pluggable `decomposer` to produce a `Plan`
of dependent `PlanStep`s, dispatches each step as a `Task` against the
runtime queue, and aggregates step results.

```python
from agi.coordinator import Coordinator, Goal, Plan, PlanStep
from agi.runtime import Runtime

def planner(goal):
    return Plan(steps=[
        PlanStep(id="plan",       role="planner",    prompt=f"plan: {goal.intent}"),
        PlanStep(id="gather",     role="researcher", prompt=f"gather: {goal.intent}", depends_on=["plan"]),
        PlanStep(id="synthesize", role="writer",     prompt=f"summarize: {goal.intent}", depends_on=["gather"]),
    ])

result = Coordinator(Runtime(), decomposer=planner).run(
    Goal(intent="summarize LoRA adapters in production", budget_usd=1.0)
)
print(result.final_text)
```

The Coordinator talks to the Runtime only through its public API
(`create_session`, `chat`, `bus.subscribe`, `metrics`) — any other
planner can use the same surface. See `examples/coordinator_e2e.py`
for a full run including skill mining.

### Three drivers ship on top of the Runtime

These are coordination patterns shipped as small modules — a higher-level
coordination engine composes them or rolls its own:

- **`AutonomousLoop`** (`agi/autoloop.py`) — pursues a `Goal` across many
  attempts. Each failed attempt distills a lesson that is prepended to the
  next attempt's prompt; on success it mines a `SkillCandidate` from the
  winning trajectory. Halts on success, budget exhaustion, deadline, or
  iteration cap. Records every iteration to a `CapabilityRegistry` for
  downstream routing.

  ```python
  from agi.autoloop import AutonomousLoop, promote_skill
  from agi.capabilities import CapabilityRegistry

  caps = CapabilityRegistry()
  loop = AutonomousLoop(Coordinator(rt), max_iterations=4, capabilities=caps)
  result = loop.pursue(Goal(intent="…", acceptance=lambda t: "42" in t, budget_usd=1.0))
  if result.success and result.skill_candidate:
      promote_skill(rt, result.skill_candidate)  # graduate into the skill library
  ```

- **`SessionFork`** (`agi/fork.py`) — races N `SessionConfig` variants of the
  same prompt in parallel against the runtime's task queue and picks a
  winner via a pluggable `judge` (default: critic score, then succeeded,
  then cost). The cheapest way to lift pass rate on hard prompts.

  ```python
  from agi.fork import SessionFork, ForkVariant

  fork = SessionFork(rt, max_workers=4)
  race = fork.race("hard question", [
      ForkVariant("careful", SessionConfig(effort="high", role="planner")),
      ForkVariant("fast",    SessionConfig(effort="medium", role="executor")),
      ForkVariant("opus",    SessionConfig(model="claude-opus-4-7", role="reviewer")),
  ])
  print(race.winner.variant.name, race.winner.result)
  ```

- **`CapabilityRegistry`** (`agi/capabilities.py`) — append-only JSONL store
  of `(prompt_tokens, role, model, skills_used, success, cost, latency,
  critic_score)`. `recommend(prompt, budget_usd=…)` returns the best
  `(role, model)` bucket by similarity-weighted success rate, with a
  budget penalty. A coordinator queries this *before* dispatching work
  so each step picks the most-likely-to-succeed config.

  ```python
  rec = caps.recommend("compile this regex", budget_usd=0.05)
  cfg = rec.to_session_config(base=SessionConfig(max_tokens=8000))
  ```

See `examples/agi_demo.py` for an end-to-end narrated run that wires
all three together without an API key.

### Five more modules a coordination engine cares about

These extend the runtime into a federated, self-learning, externally
drivable engine — investor pitch: "the more you run it, the smarter,
cheaper, and harder-to-break it gets."

- **`PolicyRouter`** (`agi/policy.py`) — Thompson-sampling bandit over
  `(role, model, effort)` arms on top of `CapabilityRegistry`. Each
  decision draws from per-arm Beta posteriors conditioned on prompt
  similarity, penalised by expected cost. Real online learning at the
  routing layer; the policy converges to the right arm faster than
  the registry's similarity-weighted recommender.

  ```python
  from agi.policy import PolicyRouter

  router = PolicyRouter(caps, epsilon=0.05, cost_weight=5.0)
  decision = router.decide("compile this regex", budget_usd=0.05)
  cfg = decision.to_session_config()
  result = rt.chat(rt.create_session(cfg), "compile this regex")
  router.observe(prompt=..., decision=decision, success=True,
                 cost_usd=..., duration_seconds=...)
  ```

- **`RuntimePool`** (`agi/pool.py`) — federation layer. Add many
  `RuntimeNode`s (in-process today, HTTP/JSON-RPC out-of-process
  tomorrow); `pool.dispatch(prompt)` routes by skill match + node
  load + health. `aggregate_capabilities()` is the federation-wide
  view a coordinator sees.

  ```python
  from agi.pool import RuntimeNode, RuntimePool

  pool = RuntimePool()
  pool.add_node(RuntimeNode(node_id="gpu-1", runtime=rt1, tags=("gpu",)))
  pool.add_node(RuntimeNode(node_id="gpu-2", runtime=rt2, tags=("gpu",)))
  d = pool.dispatch("summarize this PDF", require_tag="gpu")
  ```

- **`CoordinationProtocol`** (`agi/protocol.py`) — newline-delimited
  JSON-RPC 2.0 over stdio. Any coordination engine (in any language)
  spawns `python -m agi` as a subprocess and drives it through:
  `runtime.capabilities`, `session.create/chat/cancel/end`,
  `tasks.submit/drain`, `events.subscribe/history`, `skills.save`,
  `tools.synthesize`. Notifications stream events back.

- **`SelfEvalBank`** (`agi/selfeval.py`) — mines `(prompt, expected
  substring/regex/min-length)` items from successful traces. Before
  promoting a new skill or synthesized tool, a coordinator calls
  `bank.gate_promotion(runner, baseline_pass_rate=...)` to refuse
  changes that regress the bank.

  ```python
  from agi.selfeval import SelfEvalBank

  bank = SelfEvalBank()
  bank.auto_mine(prompt=..., final_text=..., critic_score=0.95)
  ok, report = bank.gate_promotion(bank.runtime_runner(rt),
                                    baseline_pass_rate=1.0)
  ```

- **`goalc.heuristic_decomposer` / `goalc.llm_decomposer`** — the
  Coordinator's pluggable decomposer is now production-usable out
  of the box. The heuristic decomposer recognises common shapes
  (analyze / compare / build / find-and-summarize) and emits a
  multi-step DAG; the LLM decomposer asks a planner-role session to
  write a JSON Plan, reading the runtime's capabilities first. Use
  `chained_decomposer` to run heuristic-first, LLM-fallback.

  ```python
  from agi.goalc import chained_decomposer, heuristic_decomposer, llm_decomposer
  from agi.coordinator import Coordinator

  coord = Coordinator(rt, decomposer=chained_decomposer(
      heuristic_decomposer, llm_decomposer(rt), min_steps=2,
  ))
  result = coord.run(Goal(intent="analyze the impact of LoRA"))
  ```

See `examples/runtime_engine_demo.py` for a single narrated run that
exercises all five.

### Four platform layers for production deployments

- **`AutonomyEngine`** (`agi/autonomy.py`) — the *outer* loop. Pulls
  goals from a queue (anything that returns the next `Goal` or `None`),
  pursues each through `AutonomousLoop`, records outcomes to the
  `CapabilityRegistry`, mines skills from successes, gates promotion on
  `SelfEvalBank` regression, and writes new eval items back to the bank
  so the regression suite *grows from real use*. Run it as a heartbeat
  and the system measurably improves between invocations.

  ```python
  from agi.autonomy import AutonomyEngine, GoalQueue

  queue = GoalQueue()
  queue.push(Goal(intent="…", acceptance=lambda t: "42" in t))
  engine = AutonomyEngine(
      rt, Coordinator(rt),
      goal_provider=queue.as_provider(),
      eval_bank=bank,
      eval_runner=bank.runtime_runner(rt),  # gates skill promotion
      capabilities=caps,
      max_iterations=3, max_cost_per_tick_usd=0.50,
  )
  engine.run_forever(max_ticks=100, heartbeat_seconds=5.0, idle_grace_ticks=10)
  ```

  Emits `autonomy.tick_*`, `autonomy.goal_*`, `autonomy.skill_promoted`,
  `autonomy.skill_rejected`, `autonomy.evalbank_updated`, `autonomy.idle`.

- **`KnowledgeGraph`** (`agi/knowledge.py`) — typed nodes (`file`,
  `url`, `session`, `skill`, `project`, `user`, …) + directed relations
  (`depends_on`, `wrote`, `fetched`, `spawned`, …) + timestamped facts.
  `attach_to_bus(kg, runtime.bus)` makes the graph grow automatically
  from agent activity. `kg.neighborhood(node, hops=N)` and
  `kg.context_for(kind, key)` give a coordinator structured context
  to inject into the next prompt — real semantic memory, not keyword
  search.

  ```python
  from agi.knowledge import KnowledgeGraph, attach_to_bus

  kg = KnowledgeGraph()
  attach_to_bus(kg, rt.bus)
  ctx = kg.context_for("project", "agi", hops=2)  # ground next prompt
  ```

- **`PolicyManager` / `GovernedRuntime`** (`agi/governance.py`) — hard
  multi-tenant isolation. Per-tenant daily / hourly / lifetime cost
  caps, max concurrent sessions, prompts-per-minute / per-day rate
  limits, weighted fair-share scheduling across competing tenants, and
  an append-only JSONL audit log of every admission decision. The
  difference between a demo and a SaaS deployment.

  ```python
  from agi.governance import GovernedRuntime, PolicyManager, TenantLimits

  pm = PolicyManager(audit_path="/var/log/agi-audit.jsonl")
  pm.set_limits(TenantLimits("acme",
                             daily_cost_usd=10.0,
                             max_concurrent_sessions=5,
                             max_prompts_per_minute=60))
  gr = GovernedRuntime(rt, pm)
  sid = gr.create_session("acme", SessionConfig())
  text = gr.chat("acme", sid, "…")
  ```

- **`McpServer`** (`agi/mcp.py`) — exposes the Runtime as a Model
  Context Protocol server over stdio JSON-RPC. Claude Desktop, Claude
  Code, or any MCP-aware client connects with one config line and gets
  `agi.create_session`, `agi.chat`, `agi.run_goal`, `agi.recall`,
  `agi.autonomy.tick`, `agi.save_skill`, plus the live session/event
  resource feed. Distribution path: this runtime drops into any MCP
  host.

  ```python
  from agi.mcp import run_stdio
  run_stdio(rt, coordinator=coord, knowledge=kg, autonomy_engine=engine)
  ```

See `examples/agi_autonomy_demo.py` for an end-to-end run that wires
the autonomy engine, knowledge graph, capability registry, policy
router, self-eval bank, and policy manager together — no API key
needed.

### The closed loop: `EvolutionEngine`

- **`EvolutionEngine`** (`agi/evolve.py`) — the driver that turns the
  pieces above into an actual self-improvement loop a coordination
  engine can run on a schedule. Evolutionary search over agent
  `Strategy` variants (model × effort × role × system-prompt nudge ×
  skill overlay), scored on a benchmark from `SelfEvalBank` by
  ``fitness = pass_rate − cost_weight × mean_cost_usd``. Each
  generation: evaluate every strategy, select the top-k, mutate parents
  into children, eval-gate the winner, and *promote* — record outcomes
  in `CapabilityRegistry`, update `PolicyRouter` posteriors so future
  routing biases toward the winning arm, mine a skill from successful
  traces and save it to `SkillLibrary`, and grow the regression bank
  with newly-validated items.

  The artifact is an `EvolutionResult` with per-generation
  `fitness_curve`, `pass_rate_curve`, `mean_cost_curve` curves and a
  list of `PromotionRecord`s — what a UI displays as proof the runtime
  improves itself with use. Promotion is gated by the regression bank,
  so a generation that doesn't beat baseline is *rejected* and nothing
  contaminates the routing or skill layers.

  ```python
  from agi.evolve import EvolutionEngine, default_seed_strategies, runtime_runner

  engine = EvolutionEngine(
      runner=runtime_runner(rt),         # drives a real Runtime
      registry=caps, policy=router,      # closed-loop promotion targets
      skill_library=rt.skills, eval_bank=bank,
      cost_weight=2.0, seed=42,
  )
  result = engine.evolve(
      seed_strategies=default_seed_strategies(),
      benchmark=bank.all(),
      generations=4, top_k=2, children_per_gen=3,
  )
  print(result.summary())  # fitness/pass-rate/cost curves + promotions
  ```

  See `examples/evolve_demo.py` for a hermetic runnable demo that
  shows fitness climbing and cost falling across generations on a
  toy landscape.

## Runtime API — for a coordination engine

The `Runtime` is the integration point. A coordination engine (orchestrator,
planner, scheduler — anything sitting above) creates sessions, drives them,
observes the event stream, enforces budgets, and queries capabilities.

```python
from agi.runtime import Runtime, SessionConfig

rt = Runtime()

# Discover what this runtime can do
caps = rt.capabilities()
# → {models, skills, synthesized_tools, active_sessions, ...}

# Subscribe to the event stream before running anything
rt.subscribe(lambda e: print(e.kind, e.data))

# Spawn a session with a per-session budget
sid = rt.create_session(SessionConfig(
    model="claude-opus-4-7",
    effort="high",
    enable_tool_synthesis=True,   # agent can write new tools at runtime
    enable_delegation=True,        # agent can spawn subagents
    use_skills=True,               # relevant skills auto-loaded into prompt
    cost_ceiling_usd=5.00,         # session ends when budget is hit
))

result = rt.chat(sid, "Plan and execute: …")

# State + accounting available any time
print(rt.get_session(sid).to_dict())

# Persist a learned procedure as a durable skill
from agi.skills import Skill
rt.save_skill(Skill(
    name="bisect_by_test",
    description="locate a regression by running the test against bisected commits",
    body="1. Identify last-known-good commit.\n2. git bisect run …",
    tags=["debugging", "git"],
))
```

## Preflight — economic decisions before dispatch

A coordination engine driving the runtime needs *forecasts*: which task
to schedule now, which to defer, which to downgrade to a cheaper model.
The `Runtime` exposes a preflight estimator and an admission advisor
that produce those forecasts. The estimator self-trains on the runtime's
event stream — every completed chat refines future predictions.

```python
# Forecast cost / duration / p_success before committing to a chat
est = rt.estimate("Summarize this PDF and extract action items.")
# → Estimate(cost_usd=0.17, cost_p10/p90, duration_s=14.0, p_success=0.92,
#            confidence='low'|'medium'|'high', samples=N, breakdown=…)

# One-call admission decision combining preflight + governance + capacity
advice = rt.advise(
    "Render a long report",
    tenant_id="acme",       # optional — checks tenant budget/rate-limit
    config=SessionConfig(model="claude-opus-4-7"),
)
# advice.verdict ∈ {ADMIT, DEFER, DOWNGRADE, REJECT}
# DOWNGRADE carries a concrete alternative (cheaper model + expected savings)
# DEFER carries retry_after_s for the coordinator's scheduler
```

This is the missing piece for risk-aware coordination: instead of
dispatching blindly and burning budget on jobs that will be rate-
limited or fail, a coordinator can rank, defer, downgrade, or reject —
all from a single deterministic verdict.

See `examples/preflight_demo.py` for the full end-to-end walkthrough.

## RuntimeDriver — the one entry point a coordination engine uses

Preflight, admission, governance, dispatch, event streaming and billing
each have their own primitive. A coordination engine wiring them by hand
is brittle. `RuntimeDriver` collapses all of it into a single contract:

```python
from agi import RuntimeDriver, TicketRequest, PolicyManager, TenantLimits

policy = PolicyManager()
policy.set_limits(TenantLimits(tenant_id="acme", daily_cost_usd=10.0))

driver = RuntimeDriver(
    runtime=rt,
    policy=policy,
    receipts_path="receipts.jsonl",
    max_concurrent=8,
)

ticket = driver.submit(TicketRequest(
    intent="Summarize Q4 earnings call",
    tenant_id="acme",
    budget_usd=0.20,      # hard ceiling: passed through to the session
))

# Live progress
for ev in ticket.stream():
    ...

receipt = ticket.result()            # blocking; returns billing-grade Receipt
# receipt.status   ∈ completed | rejected | deferred | failed | cancelled
# receipt.decisions = [estimate → admission → (downgrade)? → route → dispatch → complete]
# receipt.estimated_cost_usd, receipt.actual_cost_usd, receipt.actual_duration_s
```

Every ticket carries a **causal decision trace** — the ordered list of
forks the driver took (estimate, admission verdict, optional downgrade,
node routing, dispatch, completion). The trace is what an operator
replays for audit, billing reconciliation, or post-hoc cost attribution.

Receipts are JSON-serializable and persist as JSONL — one line per
ticket — so a fleet of runtimes can stream billing into the same file
or pipe.

`RuntimeDriver` accepts either a single `Runtime` or a `RuntimePool`;
in the pool case the route decision records which node handled the
ticket, so a coordination engine can attribute cost across the fleet.

See `examples/driver_multi_tenant_demo.py` for the full demo: two
tenants, ten tickets, automatic model downgrade, hard per-ticket
budgets, and a fleet rollup at the end.

### Portfolio submission — fixed budget across many tickets

`RuntimeDriver.submit_portfolio` solves a different problem: you have N
tickets and **one shared budget**. Single-ticket admission is local
("can this one ticket afford to run?"); a portfolio decision is global
("which subset of these tickets, on which models, maximizes total
expected successes within $B?").

```python
requests = [TicketRequest(intent=t) for t in tasks]
tickets, plan = driver.submit_portfolio(
    requests,
    total_budget_usd=0.50,
    value_weights=priorities,    # weight each task's expected success
)

# `plan` is a JSON-serializable PortfolioPlan:
#   - one PortfolioAllocation per request with the chosen model
#     (or "skip" when no allocation is worth the marginal dollar)
#   - expected_cost_usd, expected_value, utilization
#   - `method` ∈ {"dp", "greedy"}; DP is exact, greedy is the fallback
#     for very large portfolios.
```

`driver.portfolio.frontier(requests, budgets=[0.05, 0.25, 1.00, ...])`
returns the budget → expected-value Pareto curve so operators can see
where the next dollar stops paying off.

See `examples/portfolio_demo.py` for an end-to-end walk-through: ten
tasks of varying priority, three budget tiers, a frontier curve, and a
live dispatch under shared accounting.

### SLO submission — declarative outcomes, hedged execution, compliance ledger

The portfolio API answers *"many tickets, one budget"*. The SLO API
answers the dual: *"one ticket, one objective"*. A coordination engine
declares what it wants — minimum success probability, maximum cost,
maximum latency — and the runtime compiles a concrete plan: one model
when feasible, a parallel hedge across several models when not.

```python
from agi import RuntimeDriver, TicketRequest
from agi.contract import TicketSLO

driver = RuntimeDriver(runtime=rt, compliance_path="compliance.jsonl")

slo = TicketSLO(
    min_p_success=0.95,        # I want >= 95% expected success
    max_cost_usd=0.40,         # spend up to 40 cents
    max_latency_s=30.0,        # finish in 30s wall-clock
    hedge_policy="auto",       # parallelize models if needed
    refund_on_breach=1.0,      # full refund credit on miss
)

slo_ticket = driver.submit_with_slo(TicketRequest(intent="..."), slo)

for ev in slo_ticket.stream():       # live progress (fan-in across hedges)
    ...
receipt = slo_ticket.result()        # SLOReceipt with compliance verdict
print(receipt.slo_status)            # "compliant" | "breached" | "infeasible" | "failed"
print(receipt.winner_model)          # which hedged candidate produced final_text
print(receipt.actual_cost_usd)       # aggregate cost across all hedged children
```

The compiler turns the SLO into one of two execution strategies:

  - **`STRAT_SINGLE`** — the cheapest single model whose forecast already
    meets the SLO floor. No hedge, no extra spend.
  - **`STRAT_HEDGE`** — when no single model is good enough within budget,
    greedily add candidates by uplift-per-marginal-dollar until the
    hedged success probability clears `min_p_success`. Children race;
    the first success wins and the rest are cancelled.

If the compiler reports `feasible=False` and the operator passes
`dispatch_infeasible=False`, the driver refuses up front — the SLO
ticket is returned already rejected, no spend, with `slo_status=infeasible`.

`driver.frontier_for_slo(request, slo, budgets=[...])` plots the Pareto
curve so an operator can size `max_cost_usd` on evidence — at $0.05 the
plan might be a single haiku at p≈0.78, at $0.20 it becomes a haiku +
sonnet hedge at p≈0.97, and the curve flattens above $0.50.

`driver.compliance_report()` rolls up the compliance ledger: hit rate,
breaches by kind (`cost` / `latency` / `infeasible_plan`), total
refund-eligible cost. A billing pipeline reads `compliance.jsonl` to
honor SLO refunds without bespoke plumbing.

See `examples/slo_contract_demo.py` for three scenarios — easy SLO,
tight quality (auto-hedge across three models), tight budget (infeasible,
rejected up front) — and the rolled-up compliance summary.

This is the surface a coordination engine actually wants: declarative
goals in, auditable outcomes out, with a paper trail you can bill against.

### TicketMarket — multi-tenant marketplace dispatch

The runtime sells outcomes (SLO tickets). `TicketMarket` adds the layer
a coordination engine needs when those outcomes are sold to many tenants:
per-tenant identity + tier, quote-before-spend pricing, quota enforcement,
tier-weighted fair scheduling, and refund-aware invoicing. The result is
one method call that gives an operator everything a finance dashboard
wants — revenue, refunds, cost-of-goods, gross margin, by-tenant rollup.

```python
from agi import (
    MarketTicket, RuntimeDriver, Tenant, TicketMarket,
    TIER_ECONOMY, TIER_PREMIUM, TIER_STANDARD,
)
from agi.contract import TicketSLO

driver = RuntimeDriver(runtime=rt)
market = TicketMarket(driver, max_concurrent=8, invoices_path="invoices.jsonl")

market.register_tenant(Tenant(
    tenant_id="acme",
    tier=TIER_PREMIUM,            # premium / standard / economy
    monthly_budget_usd=500.0,     # hard quota; rejections when exhausted
    markup_pct=0.50,              # tenant pays 50% over runtime cost
))

# 1. Quote before dispatch — surface the price to the tenant.
quote = market.quote(MarketTicket(
    intent="summarize Q4 earnings",
    tenant_id="acme",
    max_bid_usd=0.80,
))
# quote.list_price_usd     # cost_forecast × (1 + markup_pct)
# quote.margin_usd         # list_price - cost_forecast
# quote.accepted, quote.reason
# quote.fits_bid, quote.fits_budget

# 2. Submit — quote+enqueue+dispatch+settle in one call.
handle  = market.submit(MarketTicket(
    intent="summarize Q4 earnings",
    tenant_id="acme",
    max_bid_usd=0.80,
    slo=TicketSLO(min_p_success=0.90, max_cost_usd=0.80, max_latency_s=30.0),
))
invoice = handle.result()
# invoice.list_price_usd     # what tenant was charged on success
# invoice.refund_usd         # auto-flowed from SLO breach
# invoice.net_charge_usd     # list_price - refund (what we actually bill)
# invoice.cost_of_goods_usd  # what the runtime spent
# invoice.gross_margin_usd   # net_charge - cost_of_goods (what we keep)

# 3. Dashboard rollup — what an operator (and an investor) wants to see.
stats = market.market_stats()
# stats["revenue_usd"], stats["refunds_usd"], stats["net_revenue_usd"]
# stats["cost_of_goods_usd"], stats["gross_margin_usd"], stats["gross_margin_pct"]
# stats["queued_by_tier"], stats["in_flight_by_tier"]
# stats["per_tenant"]   — sorted by net charged, top spenders first
```

Tier-weighted scheduling: under contention the market preempts the queue
so premium tickets dispatch ahead of economy. Tier weights default to
premium=4, standard=2, economy=1 — premium is therefore ~4x more likely
to dispatch than economy when all queues are full. Per-tenant
`monthly_budget_usd` is enforced *including* provisional reservations
from in-flight tickets, so a tenant cannot over-spend by firing many
parallel sub-budget tickets.

If a market ticket carries a `TicketSLO`, refunds from SLO breaches
flow into the invoice's `refund_usd` automatically — no bespoke
reconciliation logic. Infeasible SLOs are caught at the quote stage and
refused (`REASON_INFEASIBLE_SLO`) so a tenant never pays for a
guaranteed miss.

See `examples/market_demo.py` for the three-scene story:
quote → mixed-tier flood → refund-aware SLO ticket → operator rollup.

This is the line between "I sell you outcomes" and "I sell you a
managed AI service with predictable margins." The runtime does the
work; the market does the economics.

### TicketEconomist — closed-loop margin defender + scenario simulator

`TicketMarket` prices and bills work. `TicketEconomist` is the control
plane that watches the marketplace's own economics, recommends pricing /
routing adjustments when margins erode or refunds climb, and (optionally)
applies those adjustments back to the market automatically. It also
exposes a `simulate(scenario)` what-if engine so an operator can stress-
test the business before a real cost shock or traffic surge.

```python
from agi import (
    TicketEconomist, MarginTarget, Scenario,
    TIER_PREMIUM, TIER_STANDARD, TIER_ECONOMY,
)

economist = TicketEconomist(
    market,
    window_s=300.0,                       # rolling evaluation window
    targets=[
        MarginTarget(tier=TIER_PREMIUM,  gross_margin_pct_floor=0.25, refund_rate_ceiling=0.05),
        MarginTarget(tier=TIER_STANDARD, gross_margin_pct_floor=0.15, refund_rate_ceiling=0.12),
        MarginTarget(tier=TIER_ECONOMY,  gross_margin_pct_floor=0.05, refund_rate_ceiling=0.20),
    ],
    adjustments_path="adjustments.jsonl", # append-only audit ledger
)

# Advisory mode: coordination engine pulls a health snapshot + recs.
report = economist.health()
# report.overall.gross_margin_pct, .refund_rate, .net_revenue_usd, ...
# report.by_tier, report.by_tenant
# report.healthy, report.score   (0..1 aggregate)
for adj in report.adjustments:
    # adj.kind in {"raise_markup","pause_tenant","resume_tenant",...}
    # adj.severity in {"info","warn","critical"}
    # adj.rationale carries the evidence the floor/ceiling was breached
    print(adj.rationale)

# Apply (or dry-run).
economist.apply(report.adjustments)               # mutates Tenant fields
economist.apply(report.adjustments, dry_run=True) # observe without acting

# Auto-pilot: background control loop applies adjustments every
# `control_interval_s`. Idempotent; `auto_pilot(enable=False)` stops it.
economist.auto_pilot(enable=True)

# Scenario simulator: project forward without spending. Output carries
# the actions the autopilot *would* have produced — operators see what
# they're about to do before any real spend.
sim = economist.simulate(Scenario(
    traffic_multiplier=2.0,   # double current arrival rate
    cost_multiplier=1.15,     # 15% rise in unit cost
    duration_s=3600.0,        # project one hour forward
))
# sim.projected_revenue_usd, .projected_gross_margin_pct, ...
# sim.per_tier, sim.per_tenant
# sim.actions_recommended    # what the autopilot would do under this scenario
```

The economist holds no LLM dependency. It reads the invoices the market
already wrote, makes deterministic decisions from rolling windows, and
pushes mutations back through the market's existing public surface
(`pause_tenant`, mutating `Tenant.markup_pct`, etc.). When a coordination
engine subscribes to its event bus (`economist.health_reported`,
`economist.adjustment_applied`, `economist.scenario_simulated`), every
control-plane decision is observable in real time.

Every applied adjustment is JSONL-persisted with the evidence that
triggered it — finance and regulators can reconstruct any decision.

See `examples/economist_demo.py` for the three-scene story:
healthy baseline → cost-shock margin erosion + auto-applied raise →
five stress-test scenarios (2x traffic, 15% cost shock, SLO outage, etc.).

This is the line between "we sell a managed AI service" and "we operate
a managed AI business that defends its own gross margin." A coordination
engine plugs the economist in once and gets a self-policing marketplace
plus a stress-test workbench for the operator.

### TicketOracle — counterfactual replay + admission auto-tuner

The economist defends margins at the market layer (markup, pause,
budget). `TicketOracle` defends them one level down: the admission
policy itself (`min_p_success`, `max_cost_per_turn_usd`,
`allow_downgrade`). Every receipt the driver writes carries a full
decision trace — estimate, admission verdict, downgrade, route,
dispatch, complete — and the oracle replays those traces under
alternative knob sets to answer:

  > Given the runs we already paid for, which admission knobs would
  > have saved the most money while preserving our hit rate?

```python
from agi import PolicyKnobs

# Already lazily attached on every RuntimeDriver.
oracle = driver.oracle

# Counterfactual: what would tighter knobs have done over the last 500 tickets?
report = oracle.replay(
    driver.tickets(),
    PolicyKnobs(min_p_success=0.65, max_cost_per_turn_usd=0.10),
)
report.projected_cost_savings_usd       # delta vs baseline
report.alt_success_rate                  # hit rate under the new knobs
report.verdict_changes                   # "ADMIT->REJECT": 12, "ADMIT->DOWNGRADE": 5

# Recommend: grid-search the knob space and surface the best Pareto point.
rec = oracle.recommend(window=500)
print(rec.summary)                       # one-line, ChatOps-postable
print(rec.knobs.to_dict())

# What-if: project a 25% upstream price hike across the same population.
wi = oracle.what_if(cost_multiplier=1.25)
wi.projected_cost_delta_usd              # quarter-over-quarter exposure

# Auto-tune: apply the recommendation back to the live AdmissionAdvisor.
applied = oracle.auto_tune(driver, min_savings_usd=10.0, window=500)
# advisor._min_p_success and ._max_cost_per_turn_usd now reflect the rec.
```

The oracle does **not** re-run the LLM — counterfactuals are
deterministic and free, built from the durable `Receipt.decisions`
trace plus the live `PreflightEstimator` for alt-model branches.
That keeps investor demos and audit replays milliseconds, not
dollars. For ADMITted branches the runtime actually ran, replay uses
the *observed* outcome; for the alt-only branches (REJECT / DEFER /
DOWNGRADE) it falls back to the estimator's forecast.

A coordination engine wires this once and gets:

- **Provable counterfactuals** — every "we would have saved $X" claim
  is backed by replayable receipts.
- **Self-tuning admission** — the longer the runtime runs, the
  cheaper its admission policy becomes for the same hit rate.
- **Pre-emptive what-ifs** — model the impact of upstream price
  shocks or planned policy changes against the actual workload
  before committing.

See `examples/oracle_demo.py` for the four-scene story: historical
workload → oracle recommends → 25% price-shock what-if → auto-tune
& verify the live advisor's new knobs.

### ExperimentRunner — A/B experiments as a runtime primitive

`EvolutionEngine` and `TicketOracle` *propose* changes ("ship a cheaper
model", "raise the cost cap", "swap the system prompt"). On their own,
those proposals are interesting telemetry. `ExperimentRunner` is the
discipline that turns them into safe, measurable production rollouts:

  > Every product change ships behind an experiment with a frozen
  > primary metric, predeclared guardrails, deterministic traffic
  > assignment, and an auditable decision log. Nothing promotes
  > to default without a positive primary-metric outcome that
  > also clears every guardrail.

```python
from agi import (
    Experiment, ExperimentRunner, Guardrail, Variant,
    METRIC_COST_USD, METRIC_P_SUCCESS, METRIC_LATENCY_S,
    INTERPRET_ABS_DELTA, INTERPRET_RATIO,
)

# Lazily attached on every RuntimeDriver.
runner = driver.experiments

runner.register(Experiment(
    name="cheaper-router-v3",
    variants=[
        Variant("control"),
        Variant("treatment", overrides={"model": "claude-haiku-4-5"}),
    ],
    primary_metric=METRIC_COST_USD,
    direction="min",                          # lower cost = better
    traffic_split=[0.5, 0.5],
    min_samples_per_variant=200,
    guardrails=[
        # Don't ship if success rate drops more than 5pp.
        Guardrail(metric=METRIC_P_SUCCESS, direction="min",
                  tolerance=-0.05, interpret=INTERPRET_ABS_DELTA),
        # Don't ship if latency more than 1.5x slower.
        Guardrail(metric=METRIC_LATENCY_S, direction="max",
                  tolerance=1.5, interpret=INTERPRET_RATIO),
    ],
))

# Route a ticket through the experiment — overrides are merged into
# the SessionConfig, observations auto-record on completion.
ticket = driver.submit_with_experiment(
    TicketRequest(intent="...", tenant_id="acme", budget_usd=0.20),
    "cheaper-router-v3",
)

# Inspect / decide.
status = runner.status("cheaper-router-v3")    # full readout
runner.evaluate_all()                          # auto-ship/kill terminal experiments

# Or run the autopilot loop:
runner.start_autopilot(interval_s=60.0)
```

What it gives a coordination engine:

- **Bayesian decisions for binary metrics** (success rate, refund rate,
  reject rate, breach rate) — Beta-Binomial posteriors, P(treatment >
  control) via Monte Carlo. Ships when P ≥ 1−α and the minimum sample
  size has been reached; kills when P ≤ α.
- **Welch's t-test for continuous metrics** (cost, latency, refund
  amount, tokens out) with a CI on the relative lift.
- **A derived `cost_per_success` metric** for the most common ROI
  question — "are we paying less per successful task?"
- **Guardrails with three interpretations** — `abs` (treatment's mean
  must stay below tolerance), `ratio` (treatment / control ≤ tolerance),
  `abs_delta` (treatment − control, signed). Any guardrail breach with
  high confidence triggers an *emergency kill* even before the primary
  metric converges.
- **Deterministic assignment** — `hash((tenant_id or ticket_id) + salt)`,
  so a fleet of runtimes converges on the same assignments without
  coordination, and a given tenant stays on a stable variant across
  many tickets.
- **Auditable log** — every assignment, observation, and decision lands
  in an append-only JSONL file. Reproducible release engineering, finance
  and compliance-friendly.
- **Pause / resume / ship / kill / conclude lifecycle** — a coordination
  engine can flip experiments mid-run without losing accumulated data.
- **Auto-pilot** — a background loop calls `evaluate_all()` on a tick
  and auto-promotes winners or rolls back losers.

The runner is a thin layer that composes with the rest of the runtime:
EvolutionEngine surfaces a winner → register it as an experiment
treatment → ramp traffic → ship or kill on the gate. TicketOracle
identifies cheaper admission knobs → wire them behind an experiment
before the auto-tune pushes them live. The line between "telemetry"
and "release" becomes a single first-class object.

See `examples/experiments_demo.py` for the four-scene story:
clear win ships → clear loss kills → guardrail breach overrides a
tempting cost win → 50 live FakeAgent tickets routed through the
driver's `submit_with_experiment(...)` path.

## PolicyLab — off-policy evaluation of routing policies

`agi.policy_lab.PolicyLab` lets a coordination engine **backtest any
new routing / admission / pricing policy** against the production
receipt log, *without spending a real dollar*. It implements the
contextual-bandit OPE workhorse stack:

- **IPS** (Horvitz-Thompson) — unbiased, high variance.
- **SNIPS** (self-normalised IPS) — biased, much lower variance; the
  production OPE default.
- **DM** (direct method) — fit a reward model `r̂(c, a)` and integrate.
- **DR** (doubly-robust, Dudík-Langford-Li 2011) — unbiased if either
  the propensity *or* the reward model is correct.
- **SWITCH-DR** (Wang-Agarwal-Dudík 2017) — falls back to DM on
  heavy-tailed importance weights; provably lower MSE.
- Confidence intervals via **empirical Bernstein** (Maurer-Pontil 2009)
  on the per-event influence functions, plus Student-t for comparison.

```python
from agi.policy_lab import PolicyLab, PolicyCandidate, LinearRewardModel

lab = PolicyLab(reward_model=LinearRewardModel(ridge=0.1))
for receipt in driver.tickets():        # drain the production log
    lab.record(LoggedEvent.from_receipt(receipt))

est = lab.evaluate(my_new_router, method="dr")
# Estimate(value=0.84, ci_low=0.83, ci_high=0.85, n=5000, n_eff=4250,
#          diagnostics={mean_weight: 1.0, max_weight: 50.0, ...})

cmp = lab.compare(
    target=PolicyCandidate("v2", new_router),
    baseline=PolicyCandidate("v1", current_router),
    method="dr",
)
# cmp.recommend ∈ {ship, kill, inconclusive}; cmp.lift_ci_low > 0 → ship.

rec = lab.recommend(
    candidates=[c_v1, c_v2, c_v3],
    method="dr",
    cost_per_action={"cheap": 0.001, "smart": 0.10},
)
# rec.frontier = Pareto-best (reward, cost) candidates
# rec.best     = single arg-max recommendation
```

Investor framing: *we run last week's traffic through ten candidate
policies in silico and ship the one with the highest expected reward
under a calibrated confidence interval — before paying for a single
live A/B*. See `examples/policy_lab_demo.py`.

## PolicyImprover — safe off-policy optimization with HCPI

`agi.policy_improver.PolicyImprover` is the **dual** of PolicyLab:
where the lab *evaluates* a fixed policy, the improver *finds the best
policy* in a parameterised family and certifies the answer with a
**finite-sample High-Confidence Policy Improvement (HCPI)** guarantee
(Thomas, Theocharous & Ghavamzadeh 2015). It is the bandit-optimization
primitive the coordination engine reaches for when it wants to
*upgrade* the current router / admitter / pricer **and prove the
upgrade won't regress production**.

What it implements

- **Counterfactual Risk Minimization** (Swaminathan-Joachims 2015,
  POEM): clipped IPS objective with optional variance-penalty term
  (`OBJ_CRM_VAR`). Weight clip plays the dual role of regularizer +
  variance control.
- **Multi-start projected gradient ascent** over softmax parameters.
  Multi-start handles non-convexity; the projection keeps θ in a
  bounded box; analytic ∇log π for softmax, numeric for mixture / ε.
- **Three policy spaces**: `SoftmaxPolicySpace` (general contextual),
  `MixturePolicySpace` (α-interpolation baseline ↔ target — the safe
  HCPI dial from Thomas et al. Algorithm 2), `EpsilonGreedyPolicySpace`
  (floor exploration of a fixed inner policy).
- **HCPI safety gate**: empirical-Bernstein lower bound (Maurer-Pontil
  2009) on `V(π_new) - V(π_baseline)`. Verdict is `SAFE`, `UNSAFE`, or
  `UNCERTAIN` — and `safe == True` only when LCB > 0 at level 1 - δ.
- **Diagnostics**: Kish ESS, max/mean importance weight, clipped
  fraction, action coverage, convergence flag — the same numbers a
  coordination engine needs to decide whether to ship, defer, or
  collect more data.
- **`safety_check()`** for arbitrary callable policies (e.g. a
  learned model or a hand-written rule from another team) — the same
  HCPI bound without optimization.

```python
from agi.policy_improver import PolicyImprover, SoftmaxPolicySpace
from agi.policy_lab import PolicyLab

# Baseline value = V(π_logging) on the production log.
baseline_value = lab.evaluate(current_router, method="dr").value

imp = PolicyImprover(
    policy_space=SoftmaxPolicySpace(
        actions=("haiku", "sonnet", "opus"),
        feature_names=("difficulty", "tenant_premium"),
    ),
    baseline_value=baseline_value,
    weight_clip=20.0,
    delta=0.05,
    reward_range=(0.0, 1.0),
)
imp.ingest_from_lab(lab)

result = imp.improve(n_restarts=5, n_iters=200)
if result.safe:
    coordinator.adopt(imp.policy_space.to_policy(result.parameters))
else:
    # LCB(V_new - V_baseline) did not clear 0 — keep incumbent.
    log.info("no safe improvement found", lcb=result.improvement_lcb)
```

Investor framing: *we don't ship a model swap because the average
looked better last week — we ship it only when the math says the worst
case across the run is still above the bar we are already hitting.
Same data, same dollars, with a proof that the next deploy can't
regress*. See `examples/policy_improver_demo.py`.

## Counterfactor — sequential / trajectory off-policy evaluation

`agi.counterfactor.Counterfactor` is the **temporal twin** of PolicyLab.
PolicyLab certifies single-step contextual-bandit policies (IPS / SNIPS /
DM / DR — Dudík et al., Wang et al.). The moment the coordination engine
proposes a *sequence* of decisions — route → tool → verifier — the
importance weight becomes a *product* of step-wise ratios and naive
trajectory-IS variance explodes in horizon.  Twenty-five years of
off-policy RL literature exists precisely to fix this; Counterfactor is
the runtime kernel for that fix.

Give it a log of *trajectories* `[(state, action, reward, behavior_prob), …]`
under whatever logging policy the runtime actually ran, plus any target
policy `π(a | s)`. Counterfactor returns:

- **Point estimates** under eight different off-policy estimators —
  trajectory IS, weighted IS, **PDIS** (Precup-Sutton-Singh 2000),
  **WPDIS**, the **direct method**, **DR-RL** (Jiang-Li 2016), **WDR**
  and **MAGIC** (Thomas-Brunskill 2016).  MAGIC chooses an
  MSE-optimal convex blend over a family of j-step bootstrapped
  rollouts.
- **Finite-sample CIs** under four families — Hoeffding (1963),
  Maurer-Pontil 2009 empirical Bernstein, Student-t, and the
  Vovk 2005 distribution-free conformal envelope.
- **HCOPE** — Thomas-Theocharous-Ghavamzadeh 2015 high-confidence
  off-policy *lower bound*: `V̂^L ≥ V_true` with probability ≥ 1 - α
  for *any* data-dependent stopping rule.  The bound a *safe*
  deployment loop uses: ship a new sequential policy only if its
  HCOPE-lb exceeds the live policy's point estimate.
- **Paired comparisons** — Student-t on per-trajectory paired
  differences; reports `Δ`, its CI, and `P(A > B)`.
- **Diagnostics** — Kong 1992 effective sample size on both the
  trajectory weight bag *and* the per-step weight columns, max
  weight, p99 weight, mean and variance of log-weights,
  overlap-KL between target and behaviour, clip fraction.
  Surfaces `low overlap` and `high tail` warnings *before* the
  estimator is trusted.

Drop-in Q̂ models (`ConstantQModel`, `TabularQModel`, `LinearQModel`
— closed-form ridge via Gauss-Jordan) cover the DM / DR / MAGIC needs
without numpy; drop-in policy adapters (`UniformPolicy`,
`DeterministicPolicy`, `EpsilonGreedyPolicy`, `SoftmaxPolicy`) let any
candidate strategy be evaluated in one line.

```python
from agi.counterfactor import (
    Counterfactor, DeterministicPolicy, TabularQModel,
    METHOD_WPDIS, METHOD_DR_RL, METHOD_PDIS, CI_BERNSTEIN,
)

ctr = Counterfactor(reward_range=(0.0, 1.0), weight_cap=50.0)
for trajectory in coordinator.replay():
    ctr.log_trajectory(trajectory)

q = TabularQModel()
q.fit(ctr.trajectories())

# 1. point estimate with finite-sample CI
rep = ctr.evaluate(target_policy, method=METHOD_DR_RL, q_model=q,
                   ci_method=CI_BERNSTEIN, alpha=0.05)
print(rep.value, rep.ci_lo, rep.ci_hi, rep.ess, rep.digest)

# 2. safe-deployment lower bound
hr = ctr.hcope(target_policy, method=METHOD_PDIS, alpha=0.05)
if hr.lower_bound > live_point_estimate:
    coordinator.adopt(target_policy)

# 3. paired comparison
cmp = ctr.compare(candidate, live, method=METHOD_WPDIS, alpha=0.05)
if cmp.a_dominates and cmp.p_a_better > 0.95:
    coordinator.ship(candidate)
```

Every `evaluate / hcope / compare` call emits a content-hashed receipt
to the optional `AttestationLedger`: the coordination engine can publish
*"policy v17 was certified to dominate v16 at δ=0.05 over 4,193
trajectories"* with a verifiable digest. Stdlib only; threadsafe under a
single `RLock`; 1,800 LoC for the kernel plus 750 LoC of tests covering
every estimator identity, every CI family, every diagnostic warning, and
a 1,000-trajectory paired-comparison end-to-end.

Investor framing: *PolicyLab handles "did this prompt change earn more
per call?" Counterfactor handles "did this **multi-step plan** earn more
per session?" The same finite-sample math, but for the agentic workloads
nobody else can certify — with a pessimistic lower bound for safe
deployment that holds under any stopping rule.* See
`examples/counterfactor_demo.py`.

## ConformalPredictor — distribution-free, finite-sample-valid prediction intervals

`agi.conformal.ConformalPredictor` wraps any base forecaster (cost,
duration, success-probability, multi-class label) with **provable
marginal coverage** of the form

    P( y_test ∈ Ĉ(x_test) ) ≥ 1 − α

for *any* underlying data distribution. The only assumption is
exchangeability of the calibration and test points — dramatically
weaker than i.i.d. and the assumption that actually holds for a
runtime in steady state. The lab implements the conformal-prediction
workhorse stack:

- **Split conformal** (Papadopoulos 2008) — `method="split"`. The
  baseline. Marginal coverage holds exactly.
- **CQR — Conformalized Quantile Regression** (Romano-Patterson-
  Candès 2019) — `method="cqr"`. Heteroscedasticity-aware widths
  via underlying quantile predictions.
- **Mondrian conformal prediction** (Vovk-Lindsay-Nouretdinov-
  Gammerman 2003) — `method="mondrian"`. Group-conditional
  coverage per tenant / model / task class.
- **Jackknife+** (Barber-Candès-Ramdas-Tibshirani 2021) —
  `method="jk+"`. Leave-one-out aggregation with a 1 − 2α coverage
  bound; no train/calibration split needed.
- **Adaptive Conformal Inference — ACI** (Gibbs-Candès 2021) —
  online learning rate on α that recovers long-run coverage under
  arbitrary distribution shift.
- **RAPS — Regularized Adaptive Prediction Sets** (Angelopoulos-
  Bates-Jordan-Malik 2021) — `method="raps"`. Conformal prediction
  *sets* for multi-class problems, tight on easy points and
  conservative on hard ones.

```python
from agi.conformal import ConformalPredictor

cp = ConformalPredictor(target_coverage=0.95, adaptive=True)
cp.attach_to_driver(driver)        # drain the production receipt stream

pi = cp.predict_interval(
    prediction=0.42, method="mondrian", group="tenant-a",
)
if pi.upper > tenant.budget_remaining:
    coordinator.defer(ticket)      # admission gate
```

Investor framing: *with one line, a coordination engine gets a 95%
upper bound on cost — finite-sample, distribution-free — to bound
billing risk, gate admission, and detect forecaster drift in real
time.* See `examples/conformal_demo.py`.

PolicyLab gives a coordination engine confidence intervals on
*policies*; ConformalPredictor gives the same on *individual
outcomes*. Together they complete the uncertainty stack:
calibration → off-policy evaluation → per-decision intervals.

## CausalLab — heterogeneous treatment effects, per-context counterfactuals

`agi.causal.CausalLab` is the layer above PolicyLab. PolicyLab answers
"what is the *average* lift of switching policies on the traffic we
logged?" CausalLab answers a sharper question: **for this specific
context, what is the counterfactual lift of action τ versus baseline
action β?** That object is the Conditional Average Treatment Effect:

    τ(c) := E[Y(τ) − Y(β) | C = c]

It is the routing primitive a coordination engine wants. Investor
framing: *we don't just know our new policy beats the old one on
average; we know which 23% of incoming requests it wins on, which 12%
it loses, and we route accordingly — with finite-sample confidence
intervals.*

The lab implements the four canonical CATE meta-learners:

- **T-learner** — separate outcome regressors per arm. Robust when arms
  are balanced; high variance otherwise.
- **S-learner** — single model with treatment as a feature (plus
  action × feature interactions, so it can express heterogeneity
  under a linear backbone).
- **X-learner** (Künzel et al. 2019) — cross-fit imputed treatment
  effects combined via propensity weights. SOTA when arms are
  unbalanced — exactly the LLM-routing case.
- **DR-learner** (Kennedy 2020) — doubly-robust pseudo-outcome
  regression with influence-function CIs for free.

On top of CATE the lab ships **Qini uplift curves** (Radcliffe 2007),
a **permutation test for heterogeneity** (Chernozhukov-Demirer-Duflo
2018), and a **Best Linear Predictor of CATE** that drops out
interpretable, OLS-style rules a coordinator can ship without an ML
stack.

```python
from agi.causal import CausalLab, LEARNER_DR
from agi.policy_lab import LoggedEvent

lab = CausalLab(treatment="cheap-arm", control="strong-arm")
for ev in policy_lab.events():
    lab.record(ev)

# Per-request counterfactual lift, with a 95% CI.
point = lab.cate(context={"task_difficulty": 0.7}, learner=LEARNER_DR)
if point.ci_low > 0:
    route_to("cheap-arm")    # provably positive lift here
elif point.ci_high < 0:
    route_to("strong-arm")   # provably negative; keep the strong model

# Sanity check: is *any* heterogeneity present?
het = lab.test_heterogeneity(n_permutations=200)
if not het.is_heterogeneous:
    # The new policy moves the average; it does not segment.
    fall_back_to_average_policy()

# Interpretable rule for the coordinator.
blp = lab.best_linear_predictor()
print(blp.intercept.coef, [c.feature for c in blp.coefficients])
```

The lab is stdlib-only, drains directly from a `PolicyLab` log via
`attach_to_policy_lab(...)`, and is honest about positivity
violations (it surfaces a `support_score` per context and a `low_data`
flag when the effective sample size is below floor). See
`examples/causal_demo.py` for the end-to-end coordination-engine flow.

CausalLab × PolicyLab × ConformalPredictor is the full uncertainty
stack: average effects, per-context effects, and per-outcome
intervals — all distribution-free, all stdlib, all callable inline
from a coordination engine.

## Strategist — top-level meta-decision API

`agi.strategist.Strategist` is the surface a coordination engine
actually integrates against. The runtime's seven forecasters
(`PreflightEstimator`, `CalibrationEngine`, `ConformalPredictor`,
`CausalLab`, `PolicyLab`, `PortfolioOptimizer`, `TicketOracle`) each
answer a precise statistical question. None of them on their own answers
the operational one a coordination engine has to make turn after turn:

> *"Given these candidate actions, this context, this SLO, and what
> we have learned so far, what is the right thing to do, and how
> confident are we in the answer?"*

`Strategist` is that one call. It fuses every wired forecaster into a
structured `StrategyRecommendation` carrying one of five verdicts:

| Strategy | When | What the coordinator does |
|----------|------|---------------------------|
| `single` | One candidate's calibrated `p_success` meets the SLO and `EV_LB ≥ 0` | dispatch the single arm |
| `hedge`  | No single arm meets the target; a parallel set does and fits budget | race the K cheapest arms |
| `explore`| Best mean EV is positive *but* the arm has insufficient evidence | run with logging for information value |
| `defer`  | Mean EV positive but lower-bound EV negative under risk-aversion | wait for more signal |
| `reject` | No feasible plan; all candidates over budget or under floor | tell the caller to abort |

The math (razor's-edge of the OPE / risk-quantification literature):

  - **Calibrated `p_success`** via `CalibrationEngine.calibrate(...)`.
  - **Conformal cost upper bound** via
    `ConformalPredictor.predict_interval(...)` — Mondrian when
    candidates carry a `group`, split otherwise. Distribution-free,
    finite-sample-valid.
  - **CATE vs. baseline** via `CausalLab.cate(context, learner=DR)`
    — doubly-robust influence-function CIs. Candidates with
    confidently negative CATE are flagged.
  - **OPE value** via `PolicyLab.evaluate(...)` — DR (with SNIPS
    fallback for heavy-tailed importance weights).
  - **Bayesian model averaging** across heterogeneous estimators
    (calibrated-prior, DR-OPE, SNIPS-OPE) by inverse variance. The
    more confident estimator dominates; disagreement contributes to
    the candidate's `risk_score`.
  - **Risk-adjusted EV:**

        EV     = p_cal · payoff − (1 − p_cal) · refund − cost_mean
        EV_LB  = p_cal_lower · payoff − (1 − p_cal_lower) · refund
                  − cost_p95 − λ · (cost_p95 − cost_mean)

    `EV_LB` is what gets ranked. Over-confident point estimates
    cannot whiplash a coordinator into a bad bet.

  - **Pareto frontier** across `(calibrated_p_success ↑, cost_p95 ↓)`
    so a UI can render a leaderboard, not just the winner.

  - **Provenance** via `AttestationLedger.append(...)` — every
    recommendation carries a 64-char `attestation_hash` a coordinator
    can publish for replay or audit.

  - **Self-evaluation:** `observe(rec, outcome)` forwards realised
    outcomes into every wired forecaster *and* the strategist's own
    log. `strategist.coverage_report()` then reports the strategist's
    own calibration (Brier / ECE on `p_success`, cost-p95 breach rate,
    EV bound coverage, per-strategy realised vs. predicted EV).

```python
from agi.strategist import Strategist, Candidate, StrategyConstraints, StrategyOutcome

strat = Strategist(
    calibration=cal_engine,
    conformal=cost_conformal,
    causal=causal_lab,
    policy_lab=pol_lab,
    ledger=attest_ledger,
    baseline_action_id="claude-opus-4-7",
)

rec = strat.recommend(
    candidates=[
        Candidate(id="haiku",  raw_p_success=0.74, raw_cost_usd=0.05, samples=80),
        Candidate(id="sonnet", raw_p_success=0.85, raw_cost_usd=0.18, samples=80),
        Candidate(id="opus",   raw_p_success=0.92, raw_cost_usd=0.40, samples=80),
    ],
    constraints=StrategyConstraints(
        target_p_success=0.95,
        max_cost_usd=0.50,
        payoff_usd=2.00,
        risk_aversion=1.5,
    ),
    context={"task_difficulty": 0.7, "tenant": "acme"},
)

if rec.strategy == "single":
    coordinator.dispatch(rec.primary.candidate)
elif rec.strategy == "hedge":
    coordinator.hedge([a.candidate for a in rec.hedged_arms])
elif rec.strategy == "explore":
    coordinator.dispatch_with_logging(rec.primary.candidate)
else:
    coordinator.defer_or_reject(rec.strategy, rec.rationale)

# Close the loop — feed the realised outcome into every wired layer.
strat.observe(rec, StrategyOutcome(
    recommendation_id=rec.id,
    chosen_arm_id="sonnet",
    success=True,
    cost_usd=0.17,
))

# Honest self-eval (Brier / ECE / cost_p95 breach rate / per-strategy EV).
print(strat.coverage_report().to_dict())
```

The strategist is the line between "we have ML primitives in a library"
and "we have a runtime that *makes provably-safe routing decisions with
calibrated uncertainty* and reports honestly on its own track record."
It is stdlib-only; an 8-candidate decision with all forecasters wired
runs in single-digit milliseconds. See `examples/strategist_demo.py`
for the five-verdict walkthrough.

## ExperimentDesigner — Bayesian Optimal Experiment Design

`Strategist` decides what to do *with the data we have*. The
coordination engine still has to answer the meta-question above the
strategist:

> "Of the N candidate tickets / traces / contexts we *could* run next,
> which subset, in what order, will most sharpen the policy per dollar?"

Spending the entire learning budget on the highest-EV ticket is locally
optimal but globally wasteful: the runtime ends up data-rich on actions
it was already confident in, and data-poor on the marginal cases that
actually move policy value. `ExperimentDesigner` is the principled
answer (Lindley 1956; Chaloner-Verdinelli 1995). The objective is the
**Expected Information Gain** between a design `d` and its outcome `y`
under the current parameter posterior:

```
EIG(d) = E_{θ, y ~ p(θ) p(y|θ,d)} [ log p(y|θ,d) − log p(y|d) ]
       = H[Y|d] − E_θ[ H[Y|θ,d] ]    (mutual information I(Y; Θ | d))
```

The runtime ships everything needed to compute and act on it:

| Primitive | Reference | What it answers |
|---|---|---|
| `eig_discrete` | Lindley 1956 | Exact EIG for finite Θ × Y. The truth tests pin the MC estimators to. |
| `eig_nested_mc` | Foster et al. 2019 | NMC estimator for general black-box models; tracks the O(1/N_inner) bias bound explicitly. |
| `bald_score` | Houlsby et al. 2011 | Decomposes total predictive entropy into **epistemic** (informative — chase it) and **aleatoric** (irreducible — don't bother) for an ensemble. |
| `knowledge_gradient` | Frazier-Powell-Dayanik 2008 | One-step value-of-information acquisition under a Gaussian posterior. Knows the *value* of learning, not just the bits. |
| `thompson_top_k` | Russo 2020 | Diversified, posterior-consistent batch acquisition with no tuning. |
| `BayesianBatchPlanner` | Minoux 1978 (lazy); Sviridenko 2004 (knapsack) | Submodular greedy with the (1 − 1/e) optimality bound — equal-cost or budget-constrained. |
| `DOptimalDesigner` | Fedorov 1972 | D-, A-, and E-optimal designs over a row pool. The classical answer for picking informative covariates. |

Example — a coordination engine that has logged a pool of candidate
tickets to run as exploration calls:

```python
from agi import ExperimentDesigner, DesignRequest, ExperimentCandidate

candidates = [
    ExperimentCandidate(
        id=ticket.id,
        eig=designer.score_bald(ticket.committee_predictions).bald,
        cost=ticket.est_cost_usd,
    )
    for ticket in unlabeled_pool
]

designer = ExperimentDesigner()
resp = designer.design(DesignRequest(
    candidates=candidates,
    budget=12.50,            # explore-budget for this batch
    min_eig_per=1e-3,        # filter out aleatoric-only candidates
))

for ticket_id in resp.plan.selected:
    coordinator.dispatch_with_logging(ticket_id)

# resp.eig_per_dollar tells the coordinator whether the next dollar of
# explore-budget is worth spending; resp.binding_constraint says which
# constraint stopped this batch (k / budget / min_eig_per).
```

Composition with the rest of the stack:

* `PolicyLab` tells you what you'd earn from a new policy on existing
  data. `ExperimentDesigner` decides which **next** contexts to log so
  next week's `PolicyLab` answer is sharper.
* `Strategist.recommend(...)` returns `STRAT_EXPLORE` when the data is
  too thin to commit. `ExperimentDesigner` is what makes "explore"
  honest — it scores which contexts deserve the explore budget instead
  of exploring uniformly.
* `PortfolioOptimizer` allocates against a known utility surface.
  `ExperimentDesigner` decides which experiments refine the utility
  surface itself. They compose: the portfolio reserves a fraction of
  the budget for designer-selected exploration tickets.
* `SelfEvalBank` mines a regression suite. The designer picks which of
  N unlabeled candidate traces are highest-information for the critic.

Investor framing: without principled experiment design, every dollar
spent on learning buys a *random* amount of information about the
policy. With BOED, every dollar buys the **maximum possible**
information per dollar, with provable optimality bounds. The same
budget produces a policy that converges faster — which is the only
dimension of the harness that compounds over time.

## Deliberator — adaptive sequential sampling kernel

Every primitive above answers a question *before* compute starts:
`Strategist` picks the candidate, `ExperimentDesigner` picks the explore
budget, `PreflightEstimator` predicts cost. None of them answer the
runtime's *innermost* question:

> "I've drawn k stochastic samples from this model. The samples cluster
> into a few candidate answers. Should I draw one more, commit to the
> modal answer, escalate to a stronger model, or give up?"

That question shows up everywhere a runtime makes adaptive-compute
decisions: self-consistency sampling, best-of-N against a reward model,
cross-model ensembling, online tool-call refinement, speculative-decoding
stopping. The naive answer — "draw a fixed K and majority-vote" —
overspends on easy queries and underspends on hard ones. The wishful
answer — "stop the first time the leading cluster has a plurality" — has
a known and ugly failure mode: classical confidence intervals lose
validity under data-dependent stopping. (P-hacking is the same bug.)

`Deliberator` solves both jointly. Sample-by-sample it maintains:

* a **Dirichlet-Multinomial Bayesian posterior** over candidate-answer
  clusters,
* an **anytime-valid (1 − α)-lower confidence bound** on the probability
  of the leading cluster via the Waudby-Smith–Ramdas (2024)
  predictable-mixture capital process — the current state of the art
  for finite-sample mean estimation of bounded random variables, and
* an **expected-information-gain** estimate for one more sample (the
  expected reduction in posterior entropy).

It stops when *any* of these fire — and reports *which* fired:

| Stop reason          | When it fires                                                | What the coordinator does  |
|----------------------|--------------------------------------------------------------|----------------------------|
| `STOP_EVIDENCE`      | LCB on top cluster ≥ commit threshold (anytime-valid)        | commit                     |
| `STOP_CONVERGENCE`   | EIG below floor with ≥ 2 clusters: posterior stable but ambiguous | escalate to stronger model |
| `STOP_BUDGET`        | `max_samples` / `max_cost` hit                               | defer                      |
| `STOP_INFEASIBLE`    | sampler produced zero samples                                | abort                      |

```python
from agi import Deliberator, DeliberatorSample, STOP_EVIDENCE, STOP_CONVERGENCE

deliberator = Deliberator(bus=bus)

def sample_once():
    out = agent.chat(prompt, temperature=0.7)
    return DeliberatorSample(
        answer=out.text,
        cluster_key=canonicalize(out.text),
        cost=out.usage.cost_usd,
    )

delib = deliberator.deliberate(
    sample_once,
    max_samples=16,
    max_cost=0.50,
    alpha=0.05,
    commit_threshold=0.5,
    eig_floor=0.005,
)

if delib.stop_reason == STOP_EVIDENCE:
    coordinator.commit(delib.answer, cost=delib.cost)
elif delib.stop_reason == STOP_CONVERGENCE:
    coordinator.escalate(delib)         # ambiguous; route to a stronger model
else:
    coordinator.defer(delib)
```

What it gives a coordination engine:

* **One dial — quality level α — and the runtime decides how much
  compute each query deserves.** Confident queries finish in a handful
  of samples; ambiguous queries are flagged for escalation before they
  burn budget. The `examples/deliberator_demo.py` workload shows 40%
  median compute savings vs. fixed K=16 across a difficulty mix.
* **Statistically honest early stopping.** The LCB is valid at every
  step under any stopping rule. Classical fixed-n CIs lose validity
  under early stopping; ours does not. Citation: Waudby-Smith & Ramdas,
  *Estimating means of bounded random variables by betting*, JMLR 2024.
* **A typed event stream.** `deliberator.started / sampled / committed /
  escalated / exhausted` — the runtime's per-sample telemetry channel
  for budget enforcement, observability, and trace logging.
* **A receipt** with a SHA-256 content hash that an `AttestationLedger`
  can chain into a tamper-evident replay log.
* **Self-evaluation.** `coverage_report()` checks that
  `STOP_EVIDENCE` deliberations achieved their nominal (1 − α) success
  rate — if they didn't, the underlying sampler violates exchangeability
  and the coordinator should pause or refit.

Composition with the rest of the stack:

* `Strategist.recommend(...)` returns the *ex-ante* verdict —
  SINGLE / HEDGE / EXPLORE / DEFER / REJECT. `Deliberator` is the
  *in-flight* kernel that decides whether the chosen candidate's
  output is good enough to ship or needs escalation.
* `CalibrationEngine` subscribers can watch `deliberator.observed`
  events to refine the runtime's view of when commits succeed.
* `RiskController` and `Deliberator` share the WSR primitives but
  answer orthogonal questions: RiskController picks a fixed threshold
  with population-level FWER control; Deliberator picks how many
  samples to draw on a single query with anytime-valid per-query
  control.

Investor framing: the same runtime that confidently dispatches a
trivial query in 2 samples will spend 16 samples on a hard one — and
escalate to Opus when even 16 doesn't break the tie. Adaptive compute
across the fleet, statistically defensible, one dial for the
coordinator.

## DriftSentinel — anytime-valid sequential drift detection

Every other forecaster in the stack — `CalibrationEngine`,
`ConformalPredictor`, `PolicyLab`, `CausalLab`, `Strategist` — assumes
that the data they were fit on and the data flowing through production
are exchangeable. The moment the world shifts (a new prompt mix, a
model swap upstream, a tool change, a tenant going adversarial) that
assumption silently breaks. Calibrators stay confident, conformal
intervals stay narrow, policy estimates stay sharp, and the
coordination engine keeps committing to decisions whose statistical
foundation has quietly evaporated.

`DriftSentinel` is the runtime primitive that detects exactly this. It
sits on the event bus, watches any scalar stream of interest — `p_success`
residuals, `cost` log-ratios, reward signal, critic score, tenant
outcome rate — and emits a `drift.detected` event the moment the stream
has shifted by enough that the runtime should treat its other forecasters
as stale.

The guarantee is **anytime-valid**:

```
P_H0( ∃ t : sentinel.update(x_t).triggered )  ≤  α.
```

You can peek between every sample without inflating the false-alarm
probability. P-hacking and classical sequential testing fail because
they don't have this property; `DriftSentinel` is built around it.

Three detectors compose:

* **Page-Hinkley CUSUM** — classic, mean-shift sensitive, O(1) per
  sample. Wald threshold `log(1/α)`.
* **Bayesian Online Changepoint Detection (BOCPD; Adams-MacKay 2007)**
  — maintains a posterior over the *current run length* under a
  Gaussian-NIχ² conjugate predictive. Triggers when posterior mass on
  `r_t < short_run` exceeds `alarm_mass`. Gives a calibrated
  changepoint location estimate `τ̂` so the runtime can roll stateful
  forecasters back to a known-good cut.
* **Betting Martingale (Waudby-Smith-Ramdas 2024 + Shin-Ramdas-Rinaldo
  2024)** — the current state of the art for anytime-valid mean testing
  on bounded random variables. The capital process
  `K_t = Π (1 + λ_i (x_i − μ_0))` with predictable aGRAPA bets
  `λ_i` is a non-negative martingale under H_0; by Ville's inequality
  `P(sup_t K_t ≥ 1/α) ≤ α`. Two processes (upper/lower) union-bounded
  at α/2 give a two-sided test, distribution-free.

The composition gives the runtime *both* fast detection of obvious
shifts (CUSUM/BOCPD typically trigger first) *and* a nonparametric
anytime-valid guarantee that survives heavy tails, mis-specified
predictives, or non-Gaussian noise.

```python
from agi.drift import DriftSentinel, DRIFT_DETECTED

sentinel = DriftSentinel(
    reference_mean=0.85,        # calibrated baseline (e.g. p_success)
    reference_var=0.02,
    value_range=(0.0, 1.0),     # bounded outcome
    alpha=0.01,                 # ≤ 1% false-alarm rate uniformly over time
    bus=bus,
    name="p_success",
)

bus.subscribe(kind=DRIFT_DETECTED, callback=lambda e: calibration.refit())
bus.subscribe(kind=DRIFT_DETECTED, callback=lambda e: conformal.invalidate_calibration())
bus.subscribe(kind=DRIFT_DETECTED, callback=lambda e: policy_lab.flag_stale())

for ticket in driver.completed():
    obs = sentinel.update(ticket.actual_p_success)
    if obs.triggered:
        coordinator.enter_safe_mode(
            reason=obs.method,
            since=obs.changepoint_estimate,
        )
```

The coordination engine treats `sentinel.is_drift_active()` as a
kill-switch on aggressive routing decisions. Until the sentinel resets
— typically after the calibrators have refit on the post-changepoint
window — the coordinator falls back to the safe-default action.

Composition with the rest of the stack:

* `CalibrationEngine` refits on the post-changepoint window using
  `obs.changepoint_estimate` to pick the cut.
* `ConformalPredictor` invalidates its calibration set on drift and
  falls back to its ACI variant for online recovery.
* `PolicyLab` / `CausalLab` flag their stored estimates as stale; the
  `Strategist` downweights their evidence in its EV computation until
  enough post-drift data has accumulated.
* `AttestationLedger` records the drift event with the witness sample
  and running statistic, so an auditor can replay the detection bit-
  for-bit.
* `Coordinator` / `AutonomousLoop` enter a safe-mode where new tickets
  route through the conservative baseline action and risky exploration
  is paused.

`DriftSentinelGroup` multiplexes many sentinels under one surface —
per-tenant, per-model, per-tool — so the coordinator can ask "which
streams are drifting right now?" in O(N) and route accordingly.

Investor framing: the most damaging failure mode of a production AI
deployment is silent regression — the model still produces output, the
SLO dashboard still looks green, but the underlying capability has
quietly degraded. `DriftSentinel` is the first-class runtime primitive
that catches this with a finite-sample, distribution-free, peek-anytime
guarantee. One dial (α) controls how aggressive the trust gate is.
Honest engineering for an honest dashboard.

See `examples/drift_demo.py` for an end-to-end run.

## Arbiter — fixed-confidence Best-Arm Identification

Every coordination engine on top of this runtime sooner or later faces
the same question: of these K candidate strategies — model variants,
prompt templates, tool implementations, fine-tuned adapters, sub-agent
roles — **which one is best**, and how many samples does the runtime
need to commit with a specified confidence?

`Deliberator` answers "should I sample one more time *within* a single
query?" `Arbiter` answers the cross-strategy dual: "should I sample one
more time *across* competing strategies before I commit?". It is the
classical Best-Arm Identification (BAI) problem — distinct from regret
minimisation (UCB/Thompson) that overpulls the empirical leader and
therefore *under*-explores runners-up.

Three algorithms ship under one unified API, spanning the practical
Pareto frontier between sample optimality, implementation complexity,
and operational fit:

* **Track-and-Stop (Garivier-Kaufmann, COLT 2016).** Asymptotically
  optimal: as δ → 0, no algorithm beats it in expected sample
  complexity. Tracks the game-theoretic optimal proportion vector
  `w*` solving

      w* = argmax_w min_{a ≠ a*} ( w_{a*} d(μ_{a*}, x_a) + w_a d(μ_a, x_a) )

  with `x_a` the pooled midpoint. The C-tracking rule picks the arm
  that maximises the cumulative deficit `t · w*_a − N_a(t)`, with
  forced exploration `√t/K` so every arm gets sampled infinitely
  often. Stopping is the GLR statistic
  `Z_{a*,a}(t) = N_{a*} d(μ̂_{a*}, x̂_a) + N_a d(μ̂_a, x̂_a)` against
  the Kaufmann-Koolen threshold `β(t, δ) = log((log(t)+1)/δ) +
  (K-1) log log t`.

* **KL-LUCB (Kalyanakrishnan-Tewari-Auer-Stone, ICML 2012).** The
  pragmatic workhorse: identify the empirical leader `h_t` and its
  KL-UCB challenger `l_t`, pull both, stop when
  `U_{l_t} − L_{h_t} ≤ ε`. Bounds use the KL inversion
  `U_a = sup{q : N_a · d(μ̂_a, q) ≤ β(t, δ)}`. Not asymptotically
  optimal but matches Track-and-Stop within ~2× on most regimes and
  is robust to model mismatch in finite samples.

* **Sequential Halving (Karnin-Koren-Somekh, ICML 2013).** Fixed-
  *budget* alternative: given a total sample count `T`, split into
  `⌈log₂ K⌉` rounds, eliminate the bottom half by empirical mean
  each round. With probability ≥ 1 − exp(−T / (8 H₂ log₂ K)) the
  returned arm is best. The right primitive when budget is the
  constraint, not confidence.

```python
from agi.arbiter import (
    Arbiter, ALGO_TRACK_AND_STOP, REWARD_BERNOULLI, VERDICT_BEST,
)

arbiter = Arbiter(bus=bus, attestor=attestor)

# Streaming — feed observations from the live runtime, ask for next pulls.
cid = arbiter.start_campaign(
    arms=["haiku", "sonnet", "opus"],
    algorithm=ALGO_TRACK_AND_STOP,
    delta=0.05,
    reward_model=REWARD_BERNOULLI,
)
for obs in driver.completed():
    arbiter.observe(cid, obs.arm_id, float(obs.success))
    plan = arbiter.next_pulls(cid, batch=4)
    if plan.stopped:
        break
report = arbiter.report(cid)
if report.verdict == VERDICT_BEST:
    coordinator.promote(report.best_arm, certificate=report.pac_receipt_hash)

# Synchronous — Arbiter drives the sampler itself.
def sample(arm_id: str) -> float:
    return float(driver.dispatch(prompt, model=arm_id).success)

campaign = arbiter.run(
    arms=["v1", "v2", "v3"], sampler=sample,
    algorithm=ALGO_TRACK_AND_STOP, delta=0.01, max_samples=10_000,
)
```

Composition with the rest of the stack:

* `Strategist.recommend(...)` returning `STRAT_EXPLORE` ("the data is
  too thin; run for data") hands the candidate set to
  `Arbiter.campaign(...)`. Strategist re-runs with the winner pinned
  and typically returns `STRAT_SINGLE` on the next call. Arbiter is
  the *commit-with-confidence* loop the runtime closes around
  Strategist's exploration verdict.
* `PolicyImprover.safety_check(...)` HCPI-certifies the winning arm's
  induced policy against the baseline before promotion. Arbiter says
  *which*; PolicyImprover says *whether it's safe to ship*.
* `CalibrationEngine` / `ConformalPredictor` consume per-arm means
  and residuals from `arbiter.history()` to bootstrap calibrators on
  new variants.
* `AttestationLedger` records the campaign as `arbiter.committed`
  with arm counts, stopping statistic, and PAC certificate. Replay
  the decision under the same δ and reproduce the winner bit-for-bit.
* `DriftSentinel` subscribes to `arbiter.observed`; a drift trigger
  on the winning arm's reward stream invalidates the certificate and
  the coordinator must re-arbitrate.

Investor framing: shipping a new model variant has historically been a
months-long judgement call held together by gut feel and "more is
better" testing. `Arbiter` collapses that to a one-dial decision: pick
δ, hand over the candidates, get back a winner with a tamper-evident
PAC certificate in *asymptotically optimal* sample count. On a 3-arm
Bernoulli problem at δ=0.05 with realistic gaps (0.55 / 0.75 / 0.82) a
campaign typically commits in 4–5k samples — a fraction of the
sample budget naïve fixed-N A/B testing would burn. One dial.

See `tests/test_arbiter.py` for the verified statistical contracts
(PAC coverage, sample-complexity monotonicity in δ and gap,
Track-and-Stop vs. KL-LUCB head-to-head).

## CausalDiscoverer — structure learning from observational data

`CausalLab` estimates **effects under a given DAG**. `CausalDiscoverer`
learns the **DAG itself** from observational logs. The two are dual:

  - CausalDiscoverer answers *"which features causally drive outcomes,
    and which are spurious confounders?"*
  - CausalLab answers *"for this context, what is the per-arm lift?"*

Together they close a loop a coordination engine has so far had to
fake: discover → identify → estimate.

```
observational logs  ──►  CausalDiscoverer ──► CPDAG ──► CausalLab
                                                  │
interventions you   ◄─────────────────────  intervention_targets(...)
run next sprint
```

### Algorithms

* **PC algorithm** (Spirtes-Glymour-Scheines 1991). Constraint-based.
  Builds a skeleton via Fisher-z conditional independence tests, then
  orients v-structures from the recorded sepsets, then propagates
  Meek's R1–R4 rules to a CPDAG. We implement **PC-stable**
  (Colombo-Maathuis 2014): per-edge orientation votes are collected
  and only unanimous directions are committed, so finite-sample
  test errors never produce conflicting orientations.

* **GES** (Chickering 2002). Score-based hill-climber with add /
  remove / **reverse** operators on Gaussian BIC. The reverse
  operator is essential because BIC is score-equivalent (any two DAGs
  in the same Markov equivalence class get the same score), so a
  pure add/remove search can tie-break into the wrong class and stay
  there. A constraint-based skeleton-refinement pass after
  convergence (the MMHC hybrid of Tsamardinos-Brown-Aliferis 2006)
  removes spurious edges hill-climbing couldn't escape and recovers
  v-structures from the cleaned skeleton.

* **Bootstrap stability** (Friedman-Goldszmidt-Wyner 1999). Run PC or
  GES on B nonparametric bootstrap resamples and report each edge's
  inclusion frequency. The resulting confidence is what an operator
  actually wants to see: "this edge appears in 97% of bootstraps;
  this one in 38%." Edges below a confidence threshold are dropped.

* **Active intervention selection** (Hauser-Bühlmann 2014).
  `intervention_targets(cpdag, budget)` ranks variables by the
  expected number of currently-undirected edges that intervening on
  them would orient — direct edges incident to the variable, plus
  edges that fall out via Meek-rule propagation after the direct ones
  are oriented. The top-K is what a coordinator hands to
  `ExperimentRunner` for next-sprint experiments.

### Surface

```python
from agi import CausalDiscoverer, DiscoveryRequest, intervention_targets

discoverer = CausalDiscoverer(event_bus=bus, attestor=ledger)
report = discoverer.discover(
    rows, variables,
    request=DiscoveryRequest(method="bootstrap_pc", n_bootstrap=50,
                             edge_threshold=0.7, alpha=0.05),
)

for a, b, kind, conf in report.graph.edge_summary():
    print(f"{a} {kind} {b}   (confidence {conf:.2f})")

# Minimal sufficient feature set for routing.
mb = report.graph.markov_blanket("success")

# Next-sprint experiment plan.
targets = intervention_targets(report.graph, budget=3)
for t in targets:
    print(t.variable, t.expected_orientations, t.rationale)
```

### Composes with

* `CausalLab` — the discovered DAG restricts CATE estimation to
  parents of the treatment (avoids post-treatment-variable bias).
* `ExperimentDesigner` — `intervention_targets(...)` is a BOED-style
  routine: it picks the variables whose intervention maximises
  expected information about the remaining undirected edges.
* `Strategist` — Markov blanket of the outcome variable is the
  minimal sufficient feature set for routing decisions. Strategist's
  candidate-scoring loop becomes provably more sample-efficient when
  conditioned on the Markov blanket rather than every available
  feature.
* `DriftSentinel` — drift on an edge's marginal correlation is a
  signal to re-run discovery.
* `AttestationLedger` — every discovery emits a tamper-evident
  `causal_discovery.committed` receipt: the input data digest, the
  method, α / bootstrap settings, the CPDAG, the BIC score. A
  regulator can replay discovery on the same digest and reproduce
  the structure.
* `EventBus` — `causal_discovery.started` / `.tested` /
  `.edge_dropped` / `.bootstrapped` / `.committed` / `.failed`.

### Investor framing

The most common silent failure of a production AI deployment is
correlation-as-causation: the runtime conditions on a feature that
correlates with success this week, so it moves traffic toward where
the feature is high, the feature stops predicting, and the team
can't diagnose why the wins evaporated. `CausalDiscoverer` is the
runtime primitive that catches this — it distinguishes features that
**causally** drive outcomes from features that are merely
**conditionally correlated** with them, with finite-sample
confidence and tamper-evident receipts. The result is a coordination
engine that doesn't just chase whatever explains last week's data;
it routes by **why** outcomes happen.

See `tests/test_causal_discovery.py` for the verified contracts (PC
v-structure recovery, GES + refinement consistency with PC,
no-conflicting-direction invariant across all bootstraps, Markov
blanket correctness, active intervention orientation gain).
`examples/causal_discovery_demo.py` walks through end-to-end on a
synthetic routing DAG with one true confounder and one spurious
correlate.
## Cartographer — zone-of-proximal-development curriculum kernel

`Arbiter` answers *"which of these K is best?"*. `Cartographer` answers
the upstream meta-question every long-running runtime that *learns*
eventually has to face: **which task should we attempt next?**. Without
an answer, the runtime drifts — overspending on tasks it already
masters, ignoring tasks just beyond reach, never discovering the ones
locked behind unfilled prerequisites.

Three literatures converged on the same answer:

* **Vygotsky's Zone of Proximal Development.** Learning fastest in
  tasks the learner cannot yet do alone but can do with scaffolding.
  The frontier of competence is where growth happens.
* **Oudeyer & Kaplan's Intrinsic Motivation Systems.** The agent that
  maximises its own *learning progress* — the rate of decrease of
  prediction error — develops more general competencies than one
  driven by extrinsic reward.
* **Graves et al.'s Automated Curriculum Learning.** Treats
  curriculum as a non-stationary bandit over tasks where reward is
  gain-in-competence; EXP3.S handles the non-stationarity.

`Cartographer` maintains a Beta-Binomial posterior over competence per
task, derives an anytime-valid Wilson confidence interval, computes a
signed learning-progress signal on a sliding window, and emits
curriculum recommendations under six policies:

* `POLICY_LP` — Oudeyer learning-progress greedy: `argmax v_i · |LP_i|`
* `POLICY_UCB` — Wilson upper-bound on competence: `argmax v_i · U_i`
* `POLICY_INFOGAIN` — one-step posterior-variance reduction per cost
* `POLICY_THOMPSON` — Beta-sample → rank by `v_i · μ̃_i`
* `POLICY_KNAPSACK` — Sviridenko cost-greedy submodular knapsack at a
  hard budget B, `(1 − 1/e)`-of-OPT when per-item costs are small
* `POLICY_ROUND_ROBIN` — deterministic frontier rotation for cold start

Statuses partition tasks into `LOCKED` (unmet prereqs) / `NOVICE`
(`U < entry`) / `FRONTIER` (the ZPD) / `MASTERED` (`L ≥ mastery`) /
`FRAGILE` (previously mastered, mean has dropped). Mastery propagates
through the prereq DAG on every `tick()`; cycles are rejected at
registration.

```python
from agi.cartographer import (
    Cartographer, POLICY_LP, POLICY_KNAPSACK,
)

cart = Cartographer(bus=bus, attestor=attestor,
                     mastery_threshold=0.8, entry_threshold=0.2)
cart.register_task("add-1d", value=1.0, cost=0.002)
cart.register_task("add-2d", value=2.0, cost=0.003, prereqs=("add-1d",))
cart.register_task("long-div", value=8.0, cost=0.012,
                    prereqs=("add-2d", "mul-2d"))

for outcome in driver.completed_for("add-1d"):
    cart.observe("add-1d", float(outcome.success))

# Sample-efficient: pick the frontier task with the largest LP signal.
curriculum = cart.recommend(policy=POLICY_LP, k=4)

# Budget-aware: max total LP × value under a hard cost cap.
curriculum = cart.recommend(policy=POLICY_KNAPSACK, k=10, budget=0.50)

for item in curriculum.items:
    coordinator.queue(item.task_id, pulls=8)
```

Composition with the rest of the stack:

* `Strategist` answers "what should I do *for this ticket*?";
  Cartographer answers "what should I do *next overall*?". Every N
  tickets the coordinator calls `cartographer.recommend(...)` to
  update the active task set; Strategist routes within it.
* `Arbiter`: Cartographer's frontier subset becomes Arbiter's
  candidate arms. Cartographer says *which K are worth running*;
  Arbiter says *which of those K is best*.
* `ExperimentDesigner`: the Bayesian dual. EIG over a model vs.
  posterior-variance reduction over a task — both feedable via the
  `score_fn=` hook on `recommend`.
* `DriftSentinel`: subscribe to a mastered task's reward stream; on
  drift the coordinator calls `cart.regress(task_id)` and the
  curriculum re-opens the task. Status flips to `FRAGILE` when the
  empirical mean drops by more than `regression_margin`.
* `AttestationLedger`: mastery transitions emit `cartographer.advanced`
  with a content-hash receipt. A third party can replay the trace
  and reproduce the mastery decision.
* `PolicyImprover`: before a mastery transition ships, the
  coordinator HCPI-certifies the induced policy. If unsafe,
  Cartographer keeps the task on the frontier.

Investor framing: a runtime that doesn't pick its own training
targets is a runtime that needs a human in the loop for the
*hardest* part of learning — knowing what to learn next.
`Cartographer` is that human-in-the-loop replaced by an anytime-valid
statistical kernel: it knows what it knows (Wilson CI), knows what
it's still learning (LP signal), and knows what it can't yet attempt
(prereq DAG). On a 7-task curriculum the demo
(`examples/cartographer_demo.py`) walks the system from cold start to
6/7 mastered in ~120 batched observations per task, demotes a drifted
skill, and prints a calibration report comparing predicted competence
to latent ability.

See `tests/test_cartographer.py` for the verified contracts (Wilson /
Clopper-Pearson CIs, learning-progress monotonicity, prereq-DAG cycle
rejection, fragile-status on regression, snapshot/restore round-trip,
end-to-end mastery propagation).

## Forecaster — anytime-valid probabilistic forecasting

Every other primitive in the stack ultimately *consumes* or *produces*
a probability — Arbiter compares arm means, Auditor batches p-values,
Robustifier worst-cases over ambiguity sets, Cartographer tracks
competence, CausalDiscoverer estimates ATEs.  None of them *commit
to a calibrated, score-valid probability distribution* and then prove
that commitment under arbitrary stopping. The `Forecaster` does.

It is the primitive that gives the rest of the stack a single rigorous
answer to **"what is your forecast, what's your score, and is it
calibrated?"** — with finite-sample, anytime-valid guarantees that hold
under *any* data-dependent stopping rule a coordination engine could
construct.

Three things make it different from the usual calibration toolchain:

1. **An anytime-valid e-process** for the PIT, derived from the
   Vovk-Wang test-supermartingale construction with predictable
   aGRAPA betting (Waudby-Smith & Ramdas, 2024). The wealth process
   ``W_t = ∏ (1 + θ_s (2 u_s − 1))`` is a non-negative martingale
   under H₀; by Ville's inequality
        P_{H₀}(∃t : W_t ≥ 1/α) ≤ α
   for *every* stopping rule. The runtime can therefore monitor live
   forecasts, stop the moment it sees enough evidence, and still
   report a valid α-level rejection — no sample-size pre-registration,
   no asymptotics.
2. **Strictly proper scoring rules** with the closed forms expected
   from production: Brier / log / spherical / quadratic for discrete,
   CRPS for continuous (closed-form Gaussian via Gneiting-Raftery,
   O(n) rank-sum form for empirical distributions), pinball and linex
   for asymmetric decision-loss.
3. **Hedge ensembling with a regret bound**. Multiple forecast
   streams can be combined into one with an exponentially-weighted
   average aggregator whose cumulative regret against the best
   stream is bounded by ``√((T/2) log K)`` (Cesa-Bianchi & Lugosi,
   2006, Thm 2.2). Polynomial-weights is the parameter-free variant.

```python
from agi.forecaster import (
    Forecaster, GaussianForecast, BernoulliForecast,
    CALIB_E_PROCESS, SCORE_CRPS, POOL_HEDGE, RECAL_PIT,
)

fcst = Forecaster(bus=bus, attestor=attestor)
fcst.register_stream("model-a")
fcst.register_stream("model-b")

# Stream observations as forecast/outcome pairs.
for t, y in zip(forecasts_a, observed_latencies):
    fcst.record("model-a", t, y)

# Strict propriety enforced — Brier / log / spherical / CRPS / pinball / linex.
crps_a = fcst.score("model-a", SCORE_CRPS).mean

# Anytime-valid: stop reading the moment Ville rejects.
report = fcst.calibration_test("model-a", method=CALIB_E_PROCESS, alpha=0.05)
if report.rejected:
    fcst.recalibrate("model-a", method=RECAL_PIT)

# Online conformal interval at the requested miscoverage.
iv = fcst.interval("model-a", alpha=0.1)
# iv.lower, iv.upper bracket the next y with marginal probability 1-α.

# Hedge ensemble — cumulative regret bound is part of the report.
ens = fcst.ensemble("router", ["model-a", "model-b"],
                     method=POOL_HEDGE, rule=SCORE_CRPS)
print(ens.weights, ens.cumulative_regret_bound)

# Live forecast for the next outcome.
next_forecast = fcst.forecast(ensemble_id="router")
```

Composition with the rest of the stack:

* `Arbiter`: a Forecaster stream's score history is exactly the
  reward stream Arbiter wants. Plug the per-step CRPS or log-score
  into Arbiter's KL-LUCB and get an anytime-valid best-model
  decision.
* `Auditor`: when N streams are scored simultaneously, the per-stream
  e-values feed straight into the e-BH (e-value Benjamini-Hochberg)
  procedure and the runtime gets FDR-controlled calibration-breach
  decisions across the whole fleet.
* `Equilibrator`: log-scoring is mathematically a Kelly bet on the
  outcome's identity; a calibrated forecaster is exactly the
  Bayes-optimal play of the prediction game. Equilibrator lets us
  reason about *adversarial* forecast markets the same way.
* `Strategist`: every decision a coordinator makes is implicitly a
  cost-weighted expected loss under some forecast. Forecaster gives
  the proper score; Strategist gives the budget-aware route. The
  paired call is "score this candidate, pick that one".
* `DriftSentinel`: a Forecaster's e-process rising past the rejection
  boundary is itself a drift signal. The two primitives meet on the
  PIT — DriftSentinel watches the marginal data, Forecaster watches
  the *predictive accuracy*.
* `Conformal`: the in-stream conformal interval delegates to
  `agi.conformal` for the heavier nonparametric machinery (CQR,
  RAPS, ACI).
* `AttestationLedger`: every observation, score, calibration test,
  recalibration, ensemble update, and interval emission writes a
  content-hashed receipt. A third party can replay the JSONL stream
  and recompute the e-process, score totals, and Hedge weights bit
  for bit.

Investor framing: a coordination engine that doesn't know the
*calibrated probability* it's acting on can only justify decisions
retrospectively. `Forecaster` gives it forward-looking, anytime-valid
probabilistic ground truth — and a receipt for every test it ran along
the way.

See `examples/forecaster_demo.py` for the end-to-end three-stream
flow with biased detection, recalibration, and Hedge mixing.
`tests/test_forecaster.py` verifies the contracts (strict propriety,
closed-form CRPS, Ville's inequality under H₀, e-process rejection
under controlled bias, Hedge regret-bound, conformal coverage,
threadsafety, attestation receipts) — 93 tests, all passing.

## Refuter — automated falsification as a runtime primitive

Every other primitive in this runtime *makes claims*: `Synthesizer`
says "this program is correct on the spec", `Forecaster` says "this
distribution is calibrated", `Sampler` says "this is the posterior",
`Cartographer` says "this task is in the learner's frontier",
`ConformalPredictor` says "this interval covers", `Submodular` says
"this is a (1 − 1/e)-approximation".  **The `Refuter` is the primitive
that tries to break them** — Popperian conjecture-and-refutation as a
runtime mechanism.

The pitch is Popper, automated.  A hypothesis ``H: X → bool`` is
*refuted* by a single ``x ∈ X`` with ``H(x) = False``.  Refutation is
asymmetric: one witness destroys the hypothesis; no number of
confirmations proves it — confirmations only constrain the *rate* of
failure.

```python
from agi.refuter import (
    Refuter, ContinuousSpace, ListSpace, IntegerSpace, Product, cegis_loop,
)

R = Refuter(seed=0)

# 1. Refute a known-false hypothesis: x² ≥ x is false on (0, 1)
def H(x): return x["v"] ** 2 >= x["v"]
rep = R.try_refute(H, Product(v=ContinuousSpace(-3.0, 3.0)), n_trials=1000)
# rep.refuted == True; rep.counterexample.x  ⇒ {"v": 0.4…}

# 2. Metamorphic refutation — no oracle, but a relation must hold:
#    sorted(reversed(L)) must equal sorted(L)
rep = R.try_refute_relation(
    f=sorted,
    relation=lambda x, fx, x2, fx2: fx == fx2,
    space=ListSpace(IntegerSpace(0, 50), max_len=8),
    x_to_x2=lambda L: list(reversed(L)),
    n_trials=300,
)
# rep.refuted == False; rep.support_claim() prints the CP-UCB on failure rate

# 3. Bound refutation — drive a scalar toward a tight upper bound
rep = R.try_refute_bound(
    scalar=lambda x: x["v"]**2 - 4*x["v"] + 5,
    threshold=2.5, direction="<=",
    space=Product(v=ContinuousSpace(0.0, 5.0)),
    n_trials=400,
)
# rep.refuted == True at x=0 (f(0) = 5 > 2.5)

# 4. CEGIS — refute-then-resynthesise loop (Solar-Lezama 2008)
space = ListSpace(IntegerSpace(0, 15), min_len=1, max_len=3)
final_c, witnesses = cegis_loop(
    candidate0=0,
    refute=lambda c: R.try_refute(lambda L: max(L) <= c if L else True,
                                  space, n_trials=200),
    resynthesise=lambda c, cex: max(c, max(cex.x)),
    max_rounds=20,
)
# final_c = 15, the smallest constant validating ∀L: max(L) ≤ c

# 5. Sequential / anytime-valid rate refutation
rep = R.refute_until(predicate=H, space=..., p0=0.01, alpha=0.05, n_max=5000)
# rep.e_value ≥ 1/α  ⇒  reject under any stopping rule (Ville 1939)
```

### Strategies (portfolio search, deterministic under a fixed seed)

| Strategy        | What it does                                                                  |
|-----------------|-------------------------------------------------------------------------------|
| `boundary`      | IEEE-754 / interval corners (lo, hi, mid, zero-cross, ±∞, NaN)                |
| `halton`        | Halton (1960) low-discrepancy quasi-random over continuous coordinates        |
| `random`        | Uniform i.i.d. samples — coverage baseline                                    |
| `evolutionary`  | (1+λ) ES on the satisfaction margin with Rechenberg 1/5 step-size adaptation  |
| `nelder_mead`   | Optional: derivative-free simplex search for low-dimensional continuous boxes |
| `shrink`        | QuickCheck-style structural minimisation of any found witness (Hughes-Claessen 2000) |

### Statistical witness strength

When the search ends without a counterexample, the report carries a
*finite-sample* upper bound on the failure rate:

  * **Clopper-Pearson 1934** — exact (1-α) one-sided UCB on the
    Bernoulli rate.  For `k = 0` failures in `n` trials it collapses
    to the closed form ``1 - α^{1/n}`` and to the famous rule-of-three
    ``≈ 3/n`` at α = 0.05 (Hanley-Lippman-Hand 1983).
  * **Hoeffding 1963** — sub-Gaussian UCB on bounded means; useful
    when the predicate emits a satisfaction-margin score instead of a
    bit.
  * **Vovk-Wang 2021 e-value** — anytime-valid evidence ``e_n``
    against the null ``Pr[fail] ≤ p₀``.  Ville's inequality gives
    ``Pr_{H₀}(∃ t: e_t ≥ 1/α) ≤ α`` for every stopping rule, so
    `refute_until` can be polled arbitrarily often and the rejection
    decision is still valid.

### What it composes with

| Primitive            | Refuter's role                                                       |
|----------------------|----------------------------------------------------------------------|
| `Synthesizer`        | CEGIS: refute a candidate program before commit; resynthesise on cex |
| `Forecaster`         | Refute calibration claims via metamorphic PIT-uniformity checks      |
| `Sampler`            | Posterior-predictive stress test: refute that observed moments fall in PP-quantiles |
| `ConformalPredictor` | Coverage stress-test on adversarial points found by ES               |
| `CausalDiscoverer`   | Refute conditional-independence claims that the score relied on      |
| `Submodular`         | Refute marginal-gain decrease (≡ submodularity certificate)          |
| `Skills`             | A skill's pre/post-conditions become refutable claims before action  |
| `AttestationLedger`  | Every `RefutationReport` carries a SHA-256 fingerprint over its inputs and witnesses |
| `Auditor`            | Multiple Refuter calls feed e-values for FDR-controlled multiple-refutation control |

### Investor framing

> *"Every claim our agent makes is automatically stress-tested.  Each
> output ships with a finite-sample bound on its failure rate and a
> tamper-evident receipt — Popperian falsification reduced to a
> runtime call."*

See `examples/refuter_demo.py` for the six end-to-end scenarios
(point-refutation / support / metamorphic / bound / CEGIS / sequential).
`tests/test_refuter.py` verifies the mathematical contract (boundary
corners, Halton determinism, list-shrinking minimisation, exact
Clopper-Pearson at `k=0`, e-value monotonicity, fingerprint stability,
CEGIS convergence, NaN-corner refutation, walltime budget, replay) —
44 tests, all passing.

## PrivacyAccountant — differential privacy as a runtime primitive

Every other primitive in this runtime can consume sensitive data:
the trace logger writes user prompts; Cartographer records task
identifiers; Forecaster ingests held-out labels; PolicyLab logs
reward signals.  A production deployment that touches *any* user
data inherits a regulatory obligation — GDPR, HIPAA, CCPA, the EU
AI Act — to bound information leakage about individual records.
The `PrivacyAccountant` is the primitive that supplies the **proof**:
a tamper-evident ledger of every noisy release the runtime has
emitted, with a finite-sample (ε, δ)-DP bound on the *joint* privacy
loss across all of them.

```python
from agi.privacy import PrivacyAccountant

# Allocate a (1.0, 1e-6)-DP budget for this session
A = PrivacyAccountant(epsilon=1.0, delta=1e-6, composition="basic", seed=42)

# ε-DP release of a count with sensitivity 1
noisy = A.laplace(value=true_count, sensitivity=1.0, epsilon=0.1)

# (ε, δ)-DP release with Balle-Wang 2018 analytic Gaussian σ
noisy = A.gaussian(value=mean, sensitivity=1.0, epsilon=0.1, delta=1e-7)

# McSherry-Talwar 2007 exponential mechanism for private "best of N"
chosen = A.exponential(items, utility=score_fn, sensitivity=1.0, epsilon=0.2)

# Sparse Vector Technique (Lyu-Su-Li 2017): answer many threshold
# queries but pay only on positives
svt = A.sparse_vector(threshold=10.0, sensitivity=1.0,
                       epsilon_threshold=0.1, epsilon_answer=0.1,
                       max_positive=5)
for q in stream:
    if svt.query(q):
        record(q)

# When the budget is exhausted, the next release raises BudgetExhausted
A.spent_epsilon, A.remaining_epsilon

# Audit trail
A.releases          # tuple of immutable Release records, each with a SHA-256 fingerprint
A.ledger_hash()     # one tamper-evident hash over the whole session
A.summary()         # JSON-able for compliance dashboards
```

### Composition theorems

| Theorem            | Use when                                           | Joint cost                                                  |
|--------------------|----------------------------------------------------|-------------------------------------------------------------|
| Basic (Dwork+ 2006)| ≤ tens of releases or rough auditing               | Σ ε_i, Σ δ_i                                                 |
| Advanced (DRV 2010)| many small ε releases                              | ε √(2k ln(1/δ')) + k ε (e^ε − 1), kδ + δ'                  |
| RDP (Mironov 2017) | many releases at low ε; Gaussian-heavy             | additive in ε(α) for each α; convert at the end             |
| zCDP (Bun-Steinke 2016) | Gaussian-only sessions                       | ρ-zCDP ⇒ (ρ + 2√(ρ ln(1/δ)), δ)-DP                          |
| Subsampled-RDP (Mironov-Talwar-Zhang 2019; Wang-Balle-Kasiviswanathan 2019) | DP-SGD-style minibatch updates | tight bound for Poisson sampling + Gaussian noise           |

`PrivacyAccountant` ships all four; pick via the `composition` argument
or compute manually with `basic_composition`, `advanced_composition`,
`zcdp_to_epsilon_delta`, and the `RenyiAccountant` helper.

### Mechanisms shipped

| Mechanism               | Reference                          | Guarantee                       |
|-------------------------|------------------------------------|---------------------------------|
| Laplace                 | Dwork-McSherry-Nissim-Smith 2006  | ε-DP                            |
| Gaussian (analytic σ)   | Balle-Wang 2018                    | tight (ε, δ)-DP                 |
| Gaussian (classical σ)  | Dwork-Roth 2014                    | (ε, δ)-DP for ε ≤ 1             |
| Snapping (anti-side-channel) | Mironov 2012                  | ε'-DP with side-channel safety  |
| Exponential             | McSherry-Talwar 2007               | ε-DP private selection          |
| Sparse Vector Technique | Lyu-Su-Li 2017 (corrected)         | ε-DP threshold-query streaming  |
| Binary-tree counter     | Chan-Shi-Song 2010                 | ε-DP continual release          |

### What it composes with

| Primitive            | Role                                                                   |
|----------------------|------------------------------------------------------------------------|
| `AttestationLedger`  | Every Release ships a SHA-256 fingerprint; ledger hash is replay-verifiable |
| `Auditor`            | Refuses further ingestion once the privacy odometer trips              |
| `Sampler`            | DP-SGD style training: the moments accountant gates each gradient step |
| `Forecaster`         | DP score release on held-out labels                                    |
| `Cartographer`       | Per-task counters released through Laplace / Gaussian                  |
| `Coordinator`        | Per-user accountant on Session boundaries                              |
| `Refuter`            | Refute the claim "this release is (ε, δ)-DP" via metamorphic invariance |

### Investor framing

> *"Every byte the agent writes about a user passes through a
> calibrated noise mechanism and gets debited from a regulatory-grade
> privacy budget.  At any moment the runtime can produce a tamper-
> evident hash that proves the joint (ε, δ) leakage across the entire
> session — the compliance receipt enterprise procurement asks for
> first and accepts last."*

See `examples/privacy_demo.py` for the seven end-to-end scenarios
(Laplace / analytic Gaussian / budget exhaustion / exponential / SVT /
Rényi accountant / audit trail).  `tests/test_privacy.py` verifies
the mathematical contract (standard-normal CDF round-trip, analytic
σ tighter than classical, Gaussian RDP closed form, Laplace empirical
mean/variance, exponential mechanism bias, ledger-hash determinism,
SVT positive cap, RDP accountant additivity, advanced composition
tightness) — 38 tests, all passing.

## Ranker — paired-comparison and partial-ranking inference as a runtime primitive

Every other primitive in this stack consumes *absolute* signals: reward,
cost, latency, p_success.  But the richest annotation a runtime ever
gets in production is **relative**: "the LLM judge preferred output A
over output B on this prompt", "the user clicked variant 2 over variant
1", "trader X outperformed trader Y this hour".  Chatbot Arena, MT-Bench,
AlpacaEval, modern RLHF preference datasets, search rerankers, esports
ladders — they all collapse to **pairwise** or **partial-rank**
observations.

`Ranker` is the primitive that turns those into a **posterior over
skills** with finite-sample anytime-valid confidence intervals.  It is
the relative-information dual of `Bandit` (cumulative reward) and
`Arbiter` (PAC best-arm).

```python
from agi.ranker import (
    Ranker, BRADLEY_TERRY_MM, BRADLEY_TERRY_MAP, PLACKETT_LUCE_MM,
    THURSTONE_MM, ELO, GLICKO, GLICKO2, TRUE_SKILL,
    hox_sample_complexity, rank_correlation_kendall,
)

# Register the candidates (models, prompts, judges, content items, …)
R = Ranker(items=["gpt-4", "claude", "llama-3", "gemini"],
           algorithm=BRADLEY_TERRY_MM, seed=0)

# Ingest pairwise verdicts (or full / partial rankings).
for prompt in stream:
    winner, loser = judge(prompt)
    R.observe_pair(winner, loser)            # or .observe_pair(a, b, draw=True)

# Full-ranking observations (Plackett-Luce / Thurstone / BT).
R.observe_ranking(["claude", "gpt-4", "gemini", "llama-3"])

# Global ranking + per-item posterior with standard error.
R.rank()                          # ['claude', 'gpt-4', 'gemini', 'llama-3']
R.rate("claude")                  # ItemRating(name='claude', mean=2.17, stderr=0.31, ...)

# Pairwise win-probability with an anytime-valid CI.
cp = R.compare("claude", "gpt-4", delta=0.05)
cp.mean_win_prob, cp.ci_low, cp.ci_high, cp.is_significant
# (0.71, 0.63, 0.79, True)

# Top-K with a PAC-style certification (Hajek-Oh-Xu 2014).
dec = R.top_k(2, delta=0.05)
dec.items, dec.pac_certified, dec.margin
# (['claude', 'gpt-4'], True, 1.20)

# Full diagnostic report.
rep = R.report(delta_bound=0.05)
rep.identifiable                  # True iff comparison graph is strongly connected
rep.pseudo_r2                     # McFadden goodness-of-fit
rep.scc_size, rep.isolated_items  # Tarjan SCC; items outside the largest component
rep.sample_complexity_to_topk_99  # how many more comparisons for δ=0.01 top-K?
rep.fingerprint                   # tamper-evident SHA-256

# Sample-complexity envelope without instantiating a Ranker.
hox_sample_complexity(k=10, gap=0.1, delta=0.01)   # → 18,421

# Replay-deterministic state.
state = R.state()
R2 = Ranker.from_state(state)
assert R2.rank() == R.rank()
```

### Algorithms shipped

| Algorithm                | Reference                           | Property                                                 |
|--------------------------|-------------------------------------|----------------------------------------------------------|
| `bradley_terry_mm`       | Hunter 2004 / Bradley-Terry 1952    | Globally convergent MM under strong-connectivity (Ford 1957) |
| `bradley_terry_map`      | Rasch-like ridge prior              | Newton with Gaussian prior — identifiable on disconnected graphs |
| `plackett_luce_mm`       | Plackett 1975 / Luce 1959 / Hunter 2004 | Full or partial rankings; ℓ∞-tight (Hajek-Oh-Xu 2014) |
| `thurstone_mm`           | Thurstone 1927 / Mosteller 1951     | Gaussian latent link — thinner tails than BT             |
| `elo`                    | Elo 1978                            | Classical online update, 400-point scale                 |
| `glicko`                 | Glickman 1995                       | Elo + per-player rating deviation φ                      |
| `glicko2`                | Glickman 2012                       | Elo + φ + volatility σ, Illinois bracketing              |
| `trueskill`              | Herbrich-Minka-Graepel 2007         | Microsoft factor-graph EP; Gaussian skill belief μ ± σ   |

### Anytime-valid certificates

| Method                                | Use when                                                                |
|---------------------------------------|-------------------------------------------------------------------------|
| Hoeffding (1963) half-width           | Distribution-free, fixed-time CI on each ``W_ab / N_ab``                 |
| Empirical-Bernstein (Maurer-Pontil 2009) | Variance-adaptive — tighter when one side dominates                  |
| Howard-Ramdas-McAuliffe-Sekhon (2021) anytime CS | Time-uniform: valid simultaneously for all t, not only final |
| Hajek-Oh-Xu (2014) ℓ∞                 | Prospective: "how many more comparisons for top-K at (ε, δ)?"            |

### Composition with the rest of the runtime

| Primitive             | Role                                                                  |
|-----------------------|-----------------------------------------------------------------------|
| `Arbiter`             | Ranker's full-ranking dual of Arbiter's PAC best-arm ID                |
| `Bandit`              | Dueling bandits (Yue-Joachims 2009) — Ranker's CIs feed Bandit's UCB    |
| `Strategist`          | "Is candidate A better than incumbent B?" → Ranker pairwise CI          |
| `Auditor`             | BH/FDR over many pairwise LR tests                                      |
| `TruthSerum`          | Use judge trust-scores as Ranker observations                           |
| `Diplomat`            | Rank *players* in the extensive-form game CFR runs over                 |
| `DriftSentinel`       | Cross drift threshold → `Ranker.forget(item, halflife)`                 |
| `Refuter`             | Falsify stochastic-transitivity (Tversky 1969); fingerprint inversion   |
| `PrivacyAccountant`   | DP-private ``W_ab`` releases (Hay-Rastogi-Miklau-Suciu 2009)            |
| `AttestationLedger`   | Every committed top-K hashes into the chain (replay-verifiable)         |

### Investor framing

> *"Every paired verdict — judge, click, A/B test, esports ladder match
> — flows into a single posterior with calibrated confidence intervals
> that hold at *every* time step, not just the final one.  The runtime
> can certify the top-K leaderboard at (ε, δ) and tell the coordinator
> exactly how many more annotations close the gap to certification.
> Eight peer-reviewed algorithms behind one API."*

See `examples/ranker_demo.py` for the eight-algorithm bake-off, the
Hajek-Oh-Xu sample-complexity envelope, an online TrueSkill trace, and
the PAC-certified top-K decision; `tests/test_ranker.py` verifies the
mathematical contract — 120 tests, all passing.

## Compressor — Minimum Description Length hypothesis selection as a runtime primitive

The compression principle — *the shortest description of the data is
the best hypothesis* — is the deepest unifying thread in the
foundations of intelligence: Solomonoff 1964, Rissanen 1978, Hutter
2005.  Every other primitive in this runtime returns a *decision*
(which arm to pull, which experiment to design, which plan to
execute).  None return the meta-decision of *which model class itself
best supports the data*.  `Compressor` does.

```python
from agi.compressor import (
    Compressor, BERNOULLI, MULTINOMIAL, MARKOV, GAUSSIAN, UNIFORM_DISCRETE,
    NML, TWO_PART, PREQUENTIAL, BIC, AIC,
    elias_gamma_bits, elias_delta_bits, rissanen_logstar_bits,
    log_bernoulli_nml_constant, log_multinomial_nml_constant,
    kt_codelength_binary, kt_codelength_multinomial,
)

C = Compressor()
C.register("ber",  BERNOULLI)
C.register("uni",  UNIFORM_DISCRETE, k=2)
C.register("k4",   MULTINOMIAL, k=4)
C.register("m1",   MARKOV, k=2, r=1)
C.register("gauss", GAUSSIAN, sigma_min=0.01, sigma_max=10.0)

# Codelength of a candidate model on a data prefix.
cl = C.codelength("ber", [0, 1, 1, 0, 1, 0, 1])
cl.ml, cl.parametric_complexity, cl.stochastic_complexity  # NML = ml + pc
cl.prequential, cl.two_part, cl.bic, cl.aic
cl.bits                                                    # headline (NML in bits)

# Pick the MDL-optimal model with a regret certificate.
sel = C.select(stream)
sel.winner                # "m1"
sel.gap_bits              # winner is 234.7 bits shorter than runner-up
sel.bayes_factor          # exp(|gap|) — 10^70 in favour of winner
sel.per_symbol_regret_bits  # Vovk (1990) strong-aggregator bound
sel.fingerprint           # tamper-evident SHA-256

# Pairwise comparison.
cmp = C.compare("ber", "uni", stream)
cmp.delta_bits, cmp.bayes_factor_for_a, cmp.sym_kl_predictive, cmp.cv_delta_nats

# Anytime online (Dawid 1984 prequential).  Each call returns one
# codelength increment; the total is the prequential codelength of the
# whole prefix and stays valid at every stopping time.
for x in stream:
    C.online_observe("ber", x)
C.online_state("ber").prequential_nats         # running total (equals KT)

# Aggregate report — every fit, every selection, the audit-trail fingerprint.
rep = C.report()
rep.fingerprint        # replay-verifiable via AttestationLedger
```

### Algorithms shipped

**Universal codes for the positive integers** (Rissanen 1983) — used
whenever the codelength of a structural choice is itself encoded.

| Code            | Reference        | Codelength formula                                      |
|-----------------|------------------|---------------------------------------------------------|
| `elias_gamma`   | Elias 1975       | `2⌊log₂ n⌋ + 1`                                         |
| `elias_delta`   | Elias 1975       | `⌊log₂ n⌋ + 2⌊log₂(⌊log₂ n⌋+1)⌋ + 1`                    |
| `rissanen_logstar` | Rissanen 1983 | `log₂ c₀ + log₂ n + log₂ log₂ n + …`  (`c₀ = 2.865064`) |

**Refined-MDL parametric complexity (`log C_n`)** — the Shtarkov sum
(Shtarkov 1987, Rissanen 1996) of the Normalized Maximum Likelihood
code, computed exactly when closed-form available:

| Model class             | Exact `log C_n`                                                | Reference                            |
|-------------------------|----------------------------------------------------------------|--------------------------------------|
| `bernoulli`             | `∑ₖ binom(n,k)(k/n)^k((n-k)/n)^(n-k)` — exact + asymptotic     | Rissanen-Roos-Myllymäki 2010         |
| `multinomial(k)`        | Mononen-Myllymäki linear-time recurrence                       | Mononen-Myllymäki 2008               |
| `geometric`             | `½ log(n/2π) + log π`                                          | Grünwald 2007 Ch. 7                  |
| `poisson` (bounded λ)   | `½ log(n/2π) + log 2(√λ_max − √λ_min)`                         | Rissanen 1996                        |
| `gaussian_known_sigma`  | `log((μ_max − μ_min)/(σ√(2πe/n)))`                             | Grünwald 2007 eq. 11.5               |
| `gaussian`              | `log n − log π + log Γ((n-1)/2) − log Γ(n/2) + log(σ_max/σ_min)` | Rissanen 1996                       |
| `markov(k, r)`          | `∑ₛ log C_{n_s}(k)` over context states `s`                    | Krichevsky-Trofimov 1981             |
| `histogram(m)`          | Multinomial-NML with density correction `n log(width)`         | Rissanen-Speed-Yu 1992               |
| `uniform_discrete(k)`   | `0` (no free parameters)                                       | baseline                             |
| `constant`              | `0` if the data matches; `∞` otherwise                         | hard baseline                        |

**Prequential / sequential codes** (Dawid 1984) — anytime-valid
plug-in predictives whose total codelength is computable online and
matches NML up to ``O(1)``:

| Code                       | Reference                  | Use                                  |
|----------------------------|----------------------------|--------------------------------------|
| Krichevsky-Trofimov binary | Krichevsky-Trofimov 1981   | `(n_x + ½)/(t+1)` — minimax-regret   |
| KT multinomial             | Krichevsky-Trofimov 1981   | Symmetric Dirichlet(½) plug-in       |
| Laplace's rule             | Laplace 1814               | Dirichlet(1) — looser sanity check   |
| Gaussian Student-t mixture | Bernardo-Smith 1994        | Bayes-marginal under Jeffreys prior  |

**Codelength selection methods.**

| Method        | Headline use                                                                 |
|---------------|------------------------------------------------------------------------------|
| `ml`          | Maximum-likelihood log-loss (always over-fits; included for diagnostics)     |
| `nml`         | Refined-MDL: minimax-optimal regret (Shtarkov-Rissanen)                      |
| `two_part`    | Classical Rissanen two-part — sanity-checks NML where the recurrence is hard |
| `prequential` | Sequential / Dawid online; anytime-valid                                     |
| `bic`         | Schwarz 1978 — `(k/2) log n` parametric penalty                              |
| `aic`         | Akaike 1974 — `k` parametric penalty                                         |

### Anytime-valid online prediction

KT and Dirichlet plug-ins are intrinsically sequential.  Two
coordinators sharing a stream can fork at any prefix; the codelengths
recombine additively.  The runtime exposes this as a streaming model-
selection endpoint: `Compressor.online_observe(model, x)` returns the
codelength increment of `x` under `model`'s current online state,
mutates that state in-place, and accumulates into a running total that
matches the batch KT codelength bit-exactly.

### Regret certificates

For two models with codelengths `L_a, L_b` on the same data, the gap
`ΔL = L_a − L_b` gives a Bayes factor of `exp(|ΔL|)` (nats).  The
`select(...)` call returns the gap to the runner-up, the Bayes factor,
and the **Vovk (1990) strong-aggregating-algorithm** bound on the
runner-up's per-symbol excess loss:

    `regret_per_symbol ≤ max(0, gap − log K) / n`

where `K` is the number of registered candidates.  The bound is
free-of-distribution and holds even if the data is generated outside
the union of registered model classes.

### Composition with the rest of the runtime

| Primitive          | Role                                                                       |
|--------------------|----------------------------------------------------------------------------|
| `Sampler`          | Compressor picks the model class; Sampler draws from the posterior in it    |
| `Forecaster`       | Compressor monitors prequential codelength → triggers re-fit on misspec     |
| `DriftSentinel`    | Compressor's online codelength on a rolling window IS the drift statistic   |
| `Refuter`          | Codelength gap → Bayes factor → reject/accept the "M is best" claim         |
| `Reasoner`         | Compressor scores competing boolean encodings of a constraint               |
| `Composer`         | Compressor ranks candidate plan structures by joint formula + outcome MDL   |
| `PrivacyAccountant`| Codelengths can be released under DP — additive composition over streams    |
| `AttestationLedger`| Every event in the codelength chain is canonicalised SHA-256                |

### Investor framing

> *"Compression *is* intelligence.  Every data stream the user routes
> through the runtime is automatically scored against every registered
> model class with refined-MDL codelengths.  The runtime returns the
> shortest description plus a Bayes-factor-grade regret certificate
> against the runner-up, in bits, with a free-of-distribution
> per-symbol bound on what we'd lose by deploying the wrong model.
> Solomonoff 1964 to Hutter 2005, behind one API."*

See `tests/test_compressor.py` for the mathematical contract — 102
tests, all passing: universal-code formulas, exact NML constants,
Mononen recurrence consistency, KT-vs-NML regret bracket, balanced /
biased / correlated / Gaussian / constant recovery, online ≡ batch
equivalence, tamper-evident fingerprint chain.

## Predictor — universal sequence prediction as a runtime primitive

`Compressor` selects the best of a *finite* catalogue of model
classes for a stream — Bernoulli, Markov-of-order-r, Gaussian, etc.
That's the right move when the coordinator knows the family the data
lives in. The non-parametric question — *what is the predictive
distribution of the next symbol from an **unknown** source, with a
redundancy certificate that holds against any tree source up to
depth D?* — is the universal-prediction question (Solomonoff 1964;
Cover-Thomas 1991; Rissanen 1984). `Predictor` answers it.

```python
from agi.predictor import Predictor, SELECT_MAP

pred = Predictor.create(alphabet_size=2, depth=8, seed=0)
for s in stream:
    pred.observe(s)
p          = pred.predict()                  # {0: p0, 1: p1}
code_bits  = pred.code_length_bits()         # universal code length
entropy    = pred.entropy_rate_estimate()    # bits per symbol
map_tree   = pred.map_tree()                 # CTM MAP variable-order model
bound      = pred.redundancy_bound()         # universal certificate
e          = pred.e_process_vs_uniform()     # anytime-valid e-process
next_sym   = pred.select(SELECT_MAP)
report     = pred.report()
```

It implements **Context Tree Weighting** (Willems-Shtarkov-Tjalkens
1995) — the exact-mixture sequential predictor that averages over the
exponentially large class of variable-order Markov models of depth
≤ D in **O(D) per-symbol time**.  The redundancy bound is

```
-log₂ P_CTW(x₁ⁿ)  ≤  -log₂ P_S(x₁ⁿ)
                   + (|S|·(A-1)/2) · log₂(n / |S|)
                   + 2|S| - 1
```

for every tree source `S` with `|S|` leaves — Krichevsky-Trofimov
parameter redundancy + CTW model redundancy.  Per symbol the
redundancy vanishes as `O(log n / n)`.

It also ships the **Context-Tree Maximisation (CTM)** MAP tree
(Willems-Shtarkov-Tjalkens 1993) — same recursion with `max` replacing
the `½ + ½` mixture — so a coordination engine can extract an
interpretable variable-order Markov model from the same posterior.

Why this primitive matters as the **universal-prediction half of
MC-AIXI-CTW** (Veness-Ng-Hutter-Uther-Silver 2011 *A Monte-Carlo AIXI
Approximation*) — the most credible AGI approximation in the
literature.  MC-AIXI-CTW is a UCT planner whose generative model is
exactly this predictor; that planner half is `Composer` / `Active
Inference` in this stack.

| Composes with     | What it buys                                                                 |
|-------------------|------------------------------------------------------------------------------|
| `Forecaster`      | CTW predictive IS a calibrated forecast (PIT-uniformity on ranks)            |
| `Compressor`      | CTW code length IS the universal benchmark for finite-catalog MDL            |
| `Hedger`          | Register many depths as experts; AdaHedge picks D online                     |
| `Abductor`        | CTW codelength IS the marginal log-likelihood for Bayes-factor ratios         |
| `DriftSentinel`   | Running CTW log-loss is a martingale — CUSUM detects regime change           |
| `Refuter`         | CTW e-process refutes uniformity / i.i.d. / fixed-d Markov anytime-valid     |
| `Reasoner`        | MAP tree → variable-order rules → encode in Horn program for symbolic query  |
| `Sampler`         | Predictive distribution IS the proposal in SMC on discrete streams           |
| `ActiveInferencer`| CTW IS a learned generative observation model for EFE planning               |
| `Coordinator`     | Every symbol-stream goal routes through `Predictor.observe` / `.predict`     |

The `Predictor` is anytime, thread-safe, reproducible from a seed,
and emits hash-chained receipts (genesis `0`-block, SHA-256 step on
every observe / predict / select / report) that an `AttestationLedger`
can replay byte-for-byte.  Stdlib-only — no NumPy, no SciPy, no
dependencies.

**Pitch.**  Every other primitive commits to a model family before
seeing the data.  `Predictor` commits to *no* family — it averages
over them all up to depth D — and proves it cannot lose to the best
one by more than `O(log n)` bits.  When the coordination engine has
*no* prior on the structure of the symbol stream it just observed,
this is the primitive to call.

```text
$ python examples/predictor_demo.py
1. Universal compressor on structured binary streams
  all-zeros (n=400)             CTW =    5.15 bits ( 1.29% of naive)
  alternating 01 (n=400)        CTW =   10.29 bits ( 2.57% of naive)
  period-4 0011 (n=400)         CTW =   19.58 bits ( 4.89% of naive)
  Markov sticky (n=400)         CTW =  133.72 bits (33.43% of naive)
  uniform random (n=400)        CTW =  404.70 bits (101.18% of naive)

2. Entropy-rate estimation on biased sources
  P(1) = 0.50   true H = 1.0000   CTW H = 1.0013   gap = +0.0013
  P(1) = 0.70   true H = 0.8813   CTW H = 0.8720   gap = -0.0093
  P(1) = 0.90   true H = 0.4690   CTW H = 0.4684   gap = -0.0006
  P(1) = 0.99   true H = 0.0808   CTW H = 0.0849   gap = +0.0041

4. Anytime-valid e-process vs H₀: iid Uniform Bernoulli
  uniform random      e-value = 5e-02      do not reject H_0
  structured 0011     e-value = 2.7e+144   REJECT H_0 at α = 1e-9
```

## Quantilizer — safety-bounded optimisation as a runtime primitive

Every other decision primitive in this runtime — Bandit, BayesOpt,
Arbiter, PolicyImprover, Persuader, Strategist — answers the question
"*what is the best action?*".  Each is excellent when the proxy
utility being optimised *is* the true utility.  None of them defend
against the case it is **wrong** — the classical Goodhart pathology
(Manheim-Garrabrant 2018 *Categorizing Variants of Goodhart's Law*)
in which optimising a proxy past the threshold at which it correlates
with the truth amplifies hidden costs without bound.

`Quantilizer` (Taylor 2016 *Quantilizers: A Safer Alternative to
Maximizers for Limited Optimization*, AAAI-16 AI Ethics workshop)
fills that gap.

Given a base distribution `b` over actions and a proxy utility `U`,
the `q`-quantilizer is the distribution that samples uniformly from
the top-`q`-quantile of `b` ranked by `U`.  When `q = 1` it is the
base distribution.  When `q → 0` it is argmax `U` — the
Goodhart-vulnerable optimiser.  In between it interpolates along a
precisely characterised safety / performance frontier with three
distribution-free certificates the coordinator can take to the
auditor:

  * **KL bound** (Taylor 2016, Theorem 1):
    `KL( q-quant ‖ b ) ≤ log(1 / q)`.  Tight in the worst case.

  * **Cost-amplification bound** (Taylor 2016, Theorem 2): if a
    hidden cost `c` satisfies `E_b[c] ≤ C`, then
    `E_{q-quant}[c] ≤ C / q`.

  * **Total variation bound**: `TV( q-quant ‖ b ) ≤ 1 − q`.

### Algorithms shipped

  * **Hard discrete quantilizer** (Taylor 2016) — exact, deterministic
    SHA-256 tie-break on actions, returns the realised KL (≤ the
    `log(1/q)` worst case).

  * **Top-K quantilizer** — the discrete-action specialisation,
    KL bound `log(1 / W_K)` for kept base mass `W_K`.

  * **Soft / Boltzmann quantilizer** — KL-budget-constrained Gibbs
    distribution `π_β(a) ∝ b(a) exp(β U(a))` with `β` solved by
    bisection so the KL of `π_β` from `b` lands *exactly* on a
    coordinator-supplied budget.  The continuous-temperature dual of
    the hard quantilizer; composes with any utility (clipped or
    unbounded) and any sampler-only base distribution.

  * **Sample-based quantilizer** — empirical analogue with a
    Massart-DKW 1990 finite-sample band on the `(1-q)`-quantile of
    `U` under `b`, distribution-free.

### Anytime-valid certificates

Every `Selection` carries the KL/TV/cost bounds, the realised KL,
the chosen-action probabilities under both base and quantilizer, the
realised quantile threshold, and a SHA-256 hash that chains into
`AttestationLedger`.  The `Quantilizer` class additionally provides:

  * **Hoeffding 1963** distribution-free LCB / UCB on expected
    utility under the quantilizer.

  * **Maurer-Pontil 2009** empirical-Bernstein LCB / UCB (sharper
    when observed variance is low).

  * **Howard-Ramdas-McAuliffe-Sekhon 2021** anytime-valid LCB / UCB
    — the bound holds *simultaneously for every n ≥ 1*, so a
    coordinator may stop adaptively without invalidating the
    certificate.

  * **Massart 1990 DKW** band on the empirical `(1-q)`-quantile of
    `U` under `b` from accumulated base-source observations.

  * **Taylor-2016 cost UCB**: given a base-distribution cost UCB,
    return the quantilizer's cost UCB scaled by `1 / q`.

  * **Divergence conversions** — Pinsker 1964, Bretagnolle-Huber
    1979, Le Cam — for the half-dozen distance-from-base certificates
    a coordinator might ask for in one call.

### Composition with the rest of the runtime

  * **Bandit / BayesOpt / Arbiter** — wrap the inner-loop selection
    in `Quantilizer.select(base=algorithm_distribution,
    utility=estimated_reward, q=q)` to bound KL from a safe baseline.
    This is the *exploration safety budget* that makes any bandit run
    Goodhart-robust at the cost of `1/q` regret amplification.

  * **PolicyImprover** — `soft_quantilize(deployed_policy, score,
    kl_budget=B)` is the KL-bounded safe-improvement step that lands
    exactly on the budgeted frontier — `log(1/q)` becomes the safety
    constant in the HCPI Bernstein-LCB gate.

  * **Persuader** — `q`-quantilize over signal schemes bounds the
    information design's deviation from a truthful disclosure
    baseline.

  * **Strategist** — quantilize over recommendations for a
    risk-adjusted, KL-bounded meta-decision.

  * **PrivacyAccountant** — quantilization is *post-processing* of
    the base distribution; the DP guarantee on `b` transfers verbatim
    to the quantilizer with no additional ε spent.

  * **AttestationLedger** — every `Selection` chain-hashes the
    cryptographic commit to the base, the proxy, `q`, the seed, and
    the chosen action.

### Investor framing

> *"Goodhart's law has a closed-form defence.  Hand the runtime any
> base distribution and any proxy utility, and `Quantilizer` returns
> a chosen action with **three distribution-free bounds** the
> compliance officer can sign: the KL of the chosen policy from the
> safe base is at most `log(1/q)` nats; the total variation is at
> most `1 − q`; any hidden cost the proxy didn't model is amplified
> by at most `1/q` over its base-policy average.  Taylor 2016 to
> Howard-Ramdas-McAuliffe-Sekhon 2021 anytime-valid LCB, behind one
> API — the safety dial every autonomous AGI runtime needs."*

See `tests/test_quantilizer.py` for the mathematical contract — 113
tests, all passing: KL bound tightness on extreme atoms, soft-
quantilizer bisection lands on the budget, Massart-DKW quantile band
covers the empirical quantile, Hoeffding / Bernstein / anytime LCB
shrinkage with sample size, end-to-end Goodhart scenario showing
quantilizer recovers the true utility on the same proxy that
argmax-trapped.

## Hedger — universal prediction with experts as a runtime primitive

`Bandit` chooses an arm under a stochastic-reward assumption.
`BayesOpt` chooses a continuous candidate under a smooth-surrogate
assumption.  `Arbiter` certifies a best arm at fixed confidence.
`Strategist` recommends *which strategy* under its own model. Every
decision primitive in this runtime is right under its own modelling
assumption and **wrong** outside it. In production no single
assumption holds — losses are non-stationary, models are
misspecified, and the coordination engine has *several* candidate
primitives competing for the same decision.

`Hedger` (Vovk 1990 *Aggregating strategies*; Littlestone-Warmuth
1994 *The Weighted Majority Algorithm*; Freund-Schapire 1997
*A decision-theoretic generalisation of on-line learning*) is the
runtime primitive that solves the meta-decision: take a *fixed pool
of experts* (each expert being any other primitive, any model
version, any prompt, any decision rule) and an *incoming stream of
losses*, and at every round return a distribution over experts whose
cumulative loss tracks the best expert in hindsight up to a
*vanishing per-round regret* — without knowing in advance which
expert will be best, without needing losses to be stationary, and
without making any distributional assumption on the loss sequence.

### Algorithms shipped

  * **Hedge / EWA / Multiplicative Weights** (Vovk 1990;
    Littlestone-Warmuth 1994; Freund-Schapire 1997) — `w_t(i) ∝
    exp(−η L_{t-1}(i))`. Regret bound `R_T ≤ √(T log N / 2)` at
    `η = √(8 log N / T)`.

  * **AdaHedge** (de Rooij-van Erven-Grünwald-Koolen 2014) —
    parameter-free adaptive `η_t = log N / Δ_{t-1}`, with `Δ_t` the
    cumulative mixability gap. Bound `R_T ≤ 2√(V_T log N) + O(log N)`
    second-order in `V_T = sum_t δ_t`. Never loses to any
    fixed-η Hedge pointwise.

  * **NormalHedge** (Chaudhuri-Freund-Hsu 2009) — anytime,
    parameter-free, no learning rate to set. Per-rank regret
    `R_T(rank) ≤ √(2T (log(rank+1) + log N))`. Bounds hold for every
    `T` simultaneously.

  * **Squint** (Koolen-van Erven 2015) — improper-prior aggregation
    with second-order *K-quantile* regret. Optimal whenever a small
    top-`K` of experts is consistently good. Closed-form integral
    over `η ∈ [0, 1/2]` evaluated via 64-point Simpson's rule with
    log-max stabilisation.

  * **ML-Prod / Prod** (Cesa-Bianchi-Mansour-Stoltz 2007) —
    polynomial-weighted update with bound
    `R_T ≤ √(8 V_T log N) + 5 log N`.

  * **FTRL-Entropy / FTRL-L2** (Shalev-Shwartz 2007; Hazan 2019) —
    generic Follow the Regularised Leader with entropic (= Hedge) or
    L2 (= projected OGD) regulariser. The L2 variant uses
    Wang-Carreira-Perpinan 2013 linear-time exact projection.

  * **FTPL** (Hannan 1957; Kalai-Vempala 2005) — Follow the
    Perturbed Leader with exponential noise. The only family that
    works on combinatorial action spaces without an inner LP.
    Deterministic given seed; Monte-Carlo-estimated weights.

  * **Online Mirror Descent (entropic)** (Beck-Teboulle 2003) —
    coincides with Hedge; ships for API symmetry with mirror-
    descent literature.

  * **BOA** (Wintenberger 2017 *Optimal learning with Bernstein
    online aggregation*) — per-expert learning rate adapted to the
    expert's loss variance, with second-order Bernstein-tilted
    weight update `w(i) ∝ π(i) exp(η_i R(i) − η_i² V(i))`. Bound
    `R_T ≤ √(2 V_T (1 + log N)) + 2 (1 + log N)`.

### Anytime-valid certificates

Every `HedgerReport` carries

  * **First-order regret upper bound** in closed form (algorithm-
    specific): Vovk 1990, AdaHedge, NormalHedge, Squint, ML-Prod,
    BOA.

  * **PAC-Bayes regret bound** for any reference distribution `π`:
    `R_T(π) ≤ √(T KL(π ‖ uniform) / 2)` (McAllester 1999;
    Catoni 2007).

  * **Anytime confidence sequences** on every expert's mean loss
    via Howard-Ramdas-McAuliffe-Sekhon 2021. Stop at any data-
    dependent time without invalidating the certificate.

  * **Empirical Bernstein** (Maurer-Pontil 2009) per-expert loss
    LCB / UCB — sharper than Hoeffding when realised variance is
    small.

  * **Hoeffding** (Hoeffding 1963) distribution-free finite-sample
    LCB / UCB on the same.

  * **Realised KL** `KL(w_t ‖ π_0)` exactly, in nats, relative to
    the prior — measures how far the algorithm has drifted from the
    safe starting point.

  * **Tamper-evident SHA-256 fingerprint** chaining every
    `predict` / `select` / `observe` event into
    `AttestationLedger` — replay-deterministic given the seed.

### Specialists / sleeping experts

`observe_partial(losses, sleeping=...)` implements the
Freund-Schapire-Singer-Warmuth 1997 specialist setting where some
experts abstain on some rounds. The per-round regret bound becomes
a per-specialist bound on the rounds where the expert was active —
exactly what the coordination engine needs when an expert
primitive may be temporarily unavailable (rate-limited, in cooldown,
domain-mismatched).

### Composition with the rest of the runtime

  * **Bandit / BayesOpt / Arbiter / Strategist** — register each
    as an expert. Hedger combines their per-round regrets and
    returns a decision whose cumulative loss tracks the *best
    primitive in hindsight*. The universal "meta-bandit" the
    coordination engine wires its decision channels into.

  * **Forecaster** — hedge a panel of probabilistic forecasters
    under a proper scoring rule. Vovk-1990's aggregating algorithm
    gives the log-loss case constant regret `R_T ≤ log N` (no `√T`
    term) — the "universal predictor".

  * **PolicyImprover** — the PAC-Bayes regret bound *is* an HCPI-
    style safety gate. Coordinator can refuse to switch experts
    unless the Hedger's regret bound is below threshold.

  * **Quantilizer** — quantilize over the Hedger's weight
    distribution to bound KL drift from a safe-expert baseline.

  * **DriftSentinel** — the AdaHedge mixability gap `δ_t` is a
    martingale drift signal under the null "no expert is
    consistently better"; a CUSUM on `δ_t` detects regime change.

  * **Refuter** — refute claims about expert dominance via the
    per-expert Howard-Ramdas-McAuliffe-Sekhon 2021 anytime
    confidence sequence.

  * **AttestationLedger** — every `predict` / `select` / `observe`
    chains a SHA-256 fingerprint into the ledger.

  * **Coordinator** — the natural target. Every Goal whose execution
    picks among candidate primitives, model versions, prompts, or
    tools is routed through `Hedger.select()`. The coordinator
    *learns at runtime* which primitive to call in each situation,
    with bounded regret.

  * **Composer** — a Plan-level Hedger lets the coordinator hedge
    over candidate Plans with composed reliability bounds; the
    Hedger's KL bound sets the safety constant in Composer's PAC
    certificate.

### Investor framing

> *"The runtime's coordination engine has K candidate primitives,
> M model versions, P prompts — a `K · M · P`-arm meta-decision
> problem. Hedger reduces it to one API call that returns a
> distribution over experts whose cumulative loss is **provably**
> within `√(T log(K · M · P))` of the best one in hindsight, with
> no assumption on the loss sequence. The same theorem (Vovk 1990
> applied to log-loss) gives **constant regret `log N`** for proper
> scoring rules — `O(1)` cost above the best expert *forever*. The
> runtime no longer has to know which primitive is best; it learns
> at runtime, with anytime-valid certificates a compliance officer
> can sign before any action is taken."*

See `tests/test_hedger.py` for the mathematical contract — 80 tests,
all passing: weight distributions sum to 1 under every algorithm,
Hedge with minimax-optimal `η` honours the `√(T log N / 2)` bound on
IID Bernoulli streams, AdaHedge's mixability gap is non-negative
every round, NormalHedge concentrates on the winning expert without
any learning-rate tuning, FTRL-entropy matches Hedge byte-for-byte,
FTPL is replay-deterministic given the seed, sleeping-experts
correctly leave abstaining experts' cumulative losses unchanged,
snapshot/restore round-trips state exactly, and the realised
cumulative regret respects the closed-form bound across all six
algorithms on a 200-round IID benchmark.

## Intender — inverse reinforcement learning as a runtime primitive

`Bandit`, `BayesOpt`, `Strategist`, `Composer`, `ActiveInferencer`,
`Quantilizer` — every other decision primitive in this runtime
optimises *against a given reward function*. In every realistic
deployment that reward is **not given**: users hand the coordination
engine demonstrations, partial trajectories, thumbs-up/down on
candidate plans, and pairwise preferences. Before any downstream
optimiser is allowed to run, the runtime has to *infer what the user
values*.

`Intender` (Ng-Russell 2000 *Algorithms for inverse reinforcement
learning*; Ziebart 2008; Ramachandran-Amir 2007; Christiano et al.
2017) is the runtime primitive that solves the four canonical
preference-inference problems under a single typed API: maximum-entropy
IRL from trajectories, Bayesian IRL with posterior credible regions,
Bradley-Terry preference learning from pairwise comparisons, and
behavioural cloning as the baseline policy. Every fit returns a
closed-form feature-matching residual, the soft-optimal policy under
the learned reward, the KL distance from behavioural cloning (the
natural safe-deployment KL budget for Quantilizer), an explicit
identifiability bound (Cao-Cohen-Szepesvári 2021) on the dimensions
of reward space the data cannot distinguish, anytime-valid finite-
sample confidence sequences on every aggregate statistic, and a
tamper-evident SHA-256 fingerprint chain over every observation, fit,
and report event.

### Algorithms shipped

  * **MaxEnt IRL** (Ziebart-Maas-Bagnell-Dey 2008 *Maximum entropy
    inverse reinforcement learning*; Ziebart 2010 thesis). Concave
    log-likelihood

        `L(θ) = (1/N) Σ_i Σ_t θᵀ φ(s_t^i, a_t^i) − log Z(θ)`

    with closed-form gradient `μ̂_E − E_{π_soft(θ)}[φ]` (the *feature-
    matching residual*); optimised by gradient ascent with L2 prior
    and adaptive step. Soft value iteration uses log-sum-exp with
    explicit max-subtraction for numerical stability.

  * **Bayesian IRL** (Ramachandran-Amir 2007). Random-walk
    Metropolis-Hastings on the Boltzmann posterior
    `p(θ | τ) ∝ exp(β Σ_t (Q*(s_t, a_t; θ) − V*(s_t; θ))) · N(0, σ_p²)`
    with Roberts-Rosenthal 2009 adaptive proposal scale converging
    toward the 0.234 acceptance target; Geweke 1992 stationarity test
    on the chain before reporting credible regions.

  * **Preference-based reward learning** (Christiano-Leike-Brown-
    Martic-Legg-Amodei 2017; Bradley-Terry 1952). Convex negative
    log-likelihood

        `−Σ_k log σ(β θᵀ (Φ(τ_winner_k) − Φ(τ_loser_k)))`

    fitted by gradient ascent with L2 regularisation; returns training
    and held-out agreement rate with anytime-valid CS.

  * **Apprenticeship learning** (Abbeel-Ng 2004). Max-margin
    projection step that returns a unit-L2 reward perpendicular to the
    closest-seen feature-expectation; the inner loop of an
    apprenticeship-learning outer iteration.

  * **Behavioural cloning** (Pomerleau 1989 *ALVINN*). α-Laplace-
    smoothed empirical state-conditional action policy; the baseline
    against which IRL fits are compared.

  * **Soft Q-iteration** (Haarnoja-Tang-Abbeel-Levine 2017). The
    inner solver used by MaxEnt and BIRL — returns the soft Q-function,
    soft value function, and the stochastic policy
    `π_soft(a | s) ∝ exp(Q(s, a))`.

### Anytime-valid certificates

  * **Feature-matching residual** ‖μ̂_E − E_{π_soft}[φ]‖ — at the MAP
    fit this is ≤ ε_optim of the gradient-descent tolerance, the
    closed-form certificate that the learned reward *reproduces* the
    expert's observed behaviour.

  * **Posterior credible region** on θ — elementwise α and 1−α
    quantiles from the BIRL chain, gated by a Geweke 1992 stationarity
    check (|z| < 1.96 ⇒ stationary at 95%).

  * **Preference agreement rate** with a Howard-Ramdas-McAuliffe-
    Sekhon 2021 anytime-valid confidence sequence — stop at any
    data-dependent time without invalidating coverage.

  * **Identifiability bound** (Cao-Cohen-Szepesvári 2021 *Identifiability
    in inverse reinforcement learning*) — the rank, nullity, and
    conditioning of the centred feature matrix. Nullity > 0 means the
    data cannot distinguish a non-trivial linear subspace of rewards;
    downstream primitives can refuse to optimise in that subspace.

  * **Soft KL bound** `KL(π_soft(θ̂) ‖ π_BC)` — the deployment KL
    budget Quantilizer consumes.

  * **Empirical Bernstein** (Maurer-Pontil 2009) / **Hoeffding** finite-
    sample LCB / UCB on every aggregate statistic.

  * **Tamper-evident SHA-256 fingerprint chain** — every observation,
    fit, and report event chain-hashes into `AttestationLedger`;
    replay-deterministic given the seed.

### Composition with the rest of the runtime

  * **ActiveInferencer** — the learned reward `θᵀφ` becomes the
    log-preference term `log P(o | C)` in the active-inference
    generative model. Intender closes the loop in which the
    coordination engine must learn user preferences *before* planning
    under them.

  * **Strategist** — risk-adjusted action selection consumes
    `E[r(s, a)]` from Intender's posterior; the credible region is the
    uncertainty input to risk-sensitive policies.

  * **Quantilizer** — Intender's `KL(π_soft ‖ π_BC)` is the natural
    safe-deployment KL budget for KL-regularised quantilisation.

  * **Bandit / BayesOpt** — pointwise reward queries on novel (s, a)
    use `θ̂ᵀφ(s, a)` from MAP or the posterior mean from BIRL.
    Acquisition functions read the BIRL posterior variance for
    Thompson sampling and UCB.

  * **Composer** — plans whose terminal value is `θᵀφ` get
    parameterised by the posterior; Composer's PAC certificate
    carries Intender's identifiability bound forward.

  * **Ranker** — Ranker fits a *ranking* over items; Intender fits a
    *reward* over states. They compose: Ranker's pairwise comparisons
    feed Intender as preference observations, Intender's reward feeds
    Ranker as item utility.

  * **Mechanism / Persuader** — both require a model of the receiver's
    utility; Intender supplies a *learned* model from observed
    behaviour rather than assuming a known one.

  * **PolicyImprover** — Intender supplies the reward; PolicyImprover
    deploys safely under HCPI. End-to-end RLHF pipeline.

  * **Refuter** — refute candidate rewards via QuickCheck-style stress
    on the feature-matching residual.

  * **DriftSentinel** — the per-trajectory log-likelihood under the
    fitted reward is a martingale-difference under the null
    "no preference drift"; CUSUM detects user-preference shifts.

  * **AttestationLedger** — every observe / fit / preference event
    chain-hashes into the ledger.

### Investor framing

> The runtime cannot align an agent to a user's preferences without
> first *inferring* those preferences from observed behaviour. The
> Intender is the universal preference-elicitation primitive: it turns
> demonstrations and thumbs-up/down signals into a calibrated reward
> function the rest of the runtime can optimise — with explicit
> identifiability bounds, posterior uncertainty, anytime-valid
> confidence sequences, and tamper-evident audit trails. This is the
> RLHF kernel as a runtime call: one API the coordinator hands expert
> trajectories to, one API that returns a posterior over rewards
> every other primitive composes with.

See `examples/intender_demo.py` for the end-to-end loop and
`tests/test_intender.py` for the contracts: MaxEnt IRL recovers a
positive goal-feature weight on a 3×3 gridworld, Bradley-Terry
preference fitting concentrates on the winning trajectory direction,
BIRL's MCMC chain produces sensible posterior credible regions with
acceptance rate in the Roberts-Rosenthal band, and the soft-optimal
policy is a valid conditional distribution everywhere.

## Topologist — topological data analysis as a runtime primitive

Every primitive in this runtime that "looks at the shape of data" does
so through a statistical lens — moments, quantiles, kernel densities,
mixture parameters, calibration histograms. Statistics is a contraction
that throws away geometric information. A point cloud that has three
well-separated clusters, a point cloud that lies on a circle, and a
point cloud that lies on a sphere with a hole punched in it **can all
share the same mean and covariance** while having radically different
latent structure. No other primitive in this runtime distinguishes them.

The `Topologist` is the runtime primitive that closes that gap. It
implements **persistent homology** (Edelsbrunner-Letscher-Zomorodian
2002) on a filtered Vietoris-Rips complex (Vietoris 1927; Rips 1981)
and returns, for every requested homological dimension, the multiset
of `(birth, death)` pairs that summarise the topological invariants
of the data at every scale. Dimension 0 counts connected components
(clusters); dimension 1 counts independent loops; dimension 2 counts
independent voids. Each pair has a persistence `death − birth`
measuring how robust the feature is to scale perturbation, and the
diagram as a whole comes with a **stability certificate**:

> For any two finite metric spaces *X*, *Y* and any homological
> dimension *k*,
>
>     d_B(D_k(X), D_k(Y)) ≤ d_H(X, Y)
>
> where *d_B* is the bottleneck distance between persistence
> diagrams and *d_H* is the Hausdorff distance between point
> clouds. (Cohen-Steiner-Edelsbrunner-Harer 2007.)

Stability holds for **any** underlying distribution — no smoothness,
ergodicity, or i.i.d. assumption is required. This is the one
primitive that is genuinely *distribution-free* and *model-free*.

### Runtime API

```python
from agi import Topologist

top = Topologist.create(max_dim=1, max_scale=2.5, seed=0)
for p in cloud:
    top.observe(p)

diag = top.compute()                       # PersistenceDiagram
barcode = diag.barcode(dim=0)              # cluster stability ranking
loops = diag.diagram(dim=1)                # circular structure
top3 = diag.k_most_persistent(0, 3)        # top-3 clusters by persistence
ls = diag.landscape(dim=1, num_levels=3)   # vectorised feature
betti = diag.betti(scale=1.2)              # β_0, β_1, ... at given scale

band = top.bootstrap_band(n_resamples=50, alpha=0.05)   # Fasy et al. 2014
sig = diag.significant_features(dim=1, threshold=band.dim(1))

drift = diag.bottleneck_distance(reference_diagram, dim=1)
cert = top.stability_certificate(hausdorff_perturbation=0.05)
report = top.report()
```

Every `observe`, `compute`, `bootstrap_band`, `bottleneck_to` and
`report` is hashed into a SHA-256 fingerprint chain compatible with
`AttestationLedger`.

### What a coordination engine uses it for

| Question                                                              | Call                                                  |
|-----------------------------------------------------------------------|-------------------------------------------------------|
| How many distinct modes are these LLM rollouts clustered into?        | `diag.significant_features(0, threshold=band.dim(0))` |
| Did the world-model close a loop in its latent space?                 | `diag.diagram(1)` with persistence above noise        |
| Is this batch of embeddings still on the policy's training manifold?  | `diag.bottleneck_distance(reference, dim=1)`          |
| Has a new failure mode opened a hole in the calibration curve?        | `diag.betti(scale)` at the operating scale            |
| With what confidence can I claim "the data has *k* clusters"?         | `bootstrap_band` quantile vs feature persistence      |

### Mathematical roots

  * **Vietoris 1927; Rips 1981 — Vietoris-Rips complex.** The
    abstract simplicial complex on `(X, d)` at scale `r` whose
    `k`-simplices are the size-`(k+1)` subsets of diameter ≤ `r`.
  * **Edelsbrunner-Letscher-Zomorodian 2002 — Persistent homology.**
    The filtered chain complex induced by the inclusion
    `VR(X, r) ⊆ VR(X, r')` for `r ≤ r'` yields homology classes
    that are born at one scale and die at another; their
    `(birth, death)` pairs form the *persistence diagram* `D_k(X)`.
  * **Elder rule (dim 0).** Connected components admit a closed-form
    algorithm: process edges in nondecreasing scale order, track
    components with a union-find, and on every merge kill the
    *younger* (later-born) component.
  * **Standard matrix reduction (dim ≥ 1).** All simplices ordered
    by `(filt_value, dim, index)`; boundary matrix over `𝔽_2`
    reduced left-to-right; unpaired columns = essential classes,
    paired columns = `(birth, death)` pairs of the lower dimension.
  * **Cohen-Steiner-Edelsbrunner-Harer 2007 — Stability.**
    `d_B(D, D')` is 1-Lipschitz in the Hausdorff distance between
    point clouds: a perturbation of size `ε` moves every persistence
    point by at most `ε` in `ℓ_∞`.
  * **Bubenik 2015 — Persistence landscapes.** The `k`-th
    landscape function `λ_k(t) = k`-th max of the tent functions
    `tent_{(b,d)}(t) = max(0, min(t − b, d − t))`. Each landscape
    is 1-Lipschitz in bottleneck distance, giving a stable vector
    representation for downstream models.
  * **Fasy-Lecci-Rinaldo-Wasserman-Balakrishnan-Singh 2014 —
    Subsampled bootstrap.** The empirical `1 − α` quantile of the
    bottleneck distance between subsample diagrams and the full
    diagram is an asymptotic `1 − α` confidence band for the
    population diagram; features above `2 ·` quantile from the
    diagonal are statistically significant at level `α`.

### Investor framing

Every other primitive in the stack commits to a parametric model
class before it sees the data (mixture of Gaussians, tree source,
Gaussian process, Markov chain). The `Topologist` is the only
primitive that supplies a **model-free, geometry-only** answer to the
shape question. The output is a structured diagram with a
finite-sample stability certificate; the coordination engine routes
the decision through the same audit ledger every other primitive emits.

### What it deliberately doesn't claim

  * A full GUDHI / Ripser replacement. The runtime is pure-Python
    and tuned for **coordination-scale** clouds (≤ a few hundred
    points, fast, deterministic). The 2-skeleton reduction is
    `O((|X|²)ᵂ)` in the worst case; the user caps `max_scale`,
    `max_points`, `max_simplices`, or `max_dim` to stay tractable.
    For very large clouds and higher-dimensional homology, an
    external library remains the right tool.
  * A statistical test of "this data has a loop". The bootstrap
    band is a confidence statement on the population diagram; the
    user still has to decide what "significant persistence" means
    for their application.

## Embedder — distortion-bounded text embeddings as a runtime primitive

Every learning / retrieval primitive in this runtime that "compares two
pieces of text" was historically doing it through keyword overlap:
``Memory.search`` matches by literal tokens, ``SkillLibrary`` reranks via
the LLM, ``Cartographer`` clusters by a hand-supplied feature vector the
coordinator computes itself.  Both ``PLAN.md`` (Stage 3) and
``ARCHITECTURE.md`` (Long-term memory) call out the gap explicitly:

> Pluggable embedding backend — *Memory.search() becomes semantic.*
> (Anthropic doesn't ship embeddings.)

The ``Embedder`` is the primitive that closes that gap **without an
external embedding service, without a learned model, and with a
finite-sample distortion certificate**.  It composes three classical,
pure-Python, deterministic transforms:

```
text  ── HashingVectorizer (Weinberger et al. 2009) ──>  sparse ℝ^N
      ── sparse Random Projection (Achlioptas 2003)  ──>  dense  ℝ^d
      ── L2 normalisation                            ──>  unit-norm v
```

### Runtime API

```python
from agi import Embedder, embedder_jl_dimension, embedder_jl_certificate

emb = Embedder.create(dim=128, n_gram_range=(2, 4), seed=0)
v   = emb.embed("the quick brown fox")                # Embedding (unit-norm)
doc = emb.add("the lazy dog", payload={"src": "doc1"})
hits = emb.search("the brown dog", k=5)               # cosine top-K
emb.build_lsh_index(n_bands=8, bits_per_band=8)
fast = emb.search_lsh("the brown dog", k=5)           # sub-linear via SimHash
cr   = emb.cluster(k=3, max_iter=50, seed=7)          # k-means++ + Lloyd
cert = emb.jl_certificate(n_items=1000, eps=0.1)      # JL bound
rep  = emb.report()
```

Every ``embed``, ``add``, ``search``, ``cluster`` and ``report`` is hashed
into a SHA-256 fingerprint chain compatible with the
``AttestationLedger`` every other primitive uses.

### Mathematical roots

  * **Weinberger-Dasgupta-Langford-Smola-Attenberg 2009 — Feature
    Hashing.**  The estimator
    `⟨φ(x), φ(y)⟩` is *unbiased* for `⟨x, y⟩` with variance bounded by
    `(‖x‖² ‖y‖² + ⟨x, y⟩²) / N'`.  No vocabulary file, no streaming
    counts: a hash + sign is the entire model.
  * **Achlioptas 2003 — Database-friendly Random Projections.**
    Replaces the Gaussian projection matrix by a sparse `±1/√d`-valued
    matrix.  The resulting embedding still satisfies Johnson-
    Lindenstrauss, but the projection is `2/3`-sparse and exactly
    representable in integer arithmetic.
  * **Johnson-Lindenstrauss 1984; Dasgupta-Gupta 2003.**
    For any `n`-point set in any Hilbert space and `ε ∈ (0, 1/2)`,
    embedding into `d ≥ ⌈8 ln n / ε²⌉` dimensions preserves every
    pairwise squared distance within `(1 ± ε)` with probability at
    least `1 − 1/n`.  Distribution-free.
  * **Charikar 2002 — SimHash.**  For unit-norm `u, v` and a random
    Gaussian `r`, `P[sign(⟨u, r⟩) = sign(⟨v, r⟩)] = 1 − θ(u, v) / π`.
    Banded signatures give sub-linear nearest-neighbour retrieval.
  * **Arthur-Vassilvitskii 2007 — k-means++.**  `D²`-weighted seeding
    is `O(log k)`-competitive against optimal cost.  Combined with
    Lloyd iterations on cosine distance for spherical clustering.

### What a coordination engine uses it for

| Question                                                              | Call                                          |
|-----------------------------------------------------------------------|-----------------------------------------------|
| Are these two prompts asking for the same thing?                      | `emb.embed(a).cosine_to(emb.embed(b))`        |
| Which past skill is closest to this prompt?                           | `emb.search(prompt, k=3)` over the skill set  |
| Find duplicate traces in the session log                              | `emb.cluster(k, seed=…)` + tight cluster size |
| Sub-linear retrieval over 10⁴ memos                                   | `emb.search_lsh(query, k, n_bands=…, bits=…)` |
| What dimension do I need for ε=0.1 distortion on n=10⁵ items?         | `embedder_jl_dimension(100000, 0.1)`          |
| Has this batch of embeddings drifted from the training manifold?      | `Topologist.bottleneck_distance(...)` on emb. |

### Investor framing

Every other primitive in this stack already produces calibrated, audited
artefacts: forecasts, decisions, certificates.  *Until the runtime can
embed text it cannot apply any of those primitives to its own memory.*
The ``Embedder`` is the connective tissue that turns the rest of the
audit-able stack onto the runtime's own conversation history, skills,
traces and tickets — without an external API call, without a learned
model file, with a JL certificate the operator can show to a regulator.

### What it deliberately doesn't claim

  * A replacement for a learned semantic embedder (Voyage AI,
    OpenAI, sentence-transformers, BERT).  Those embeddings capture
    distributional semantics from large pretraining corpora; this
    primitive captures **lexical / syntactic proximity** with a JL
    certificate.  When a deployment wants higher-quality semantic
    retrieval, a learned backend can be wired in behind the same
    ``EmbeddingProvider`` protocol without changing downstream code.
  * A vector database.  The internal index is in-memory and tuned for
    coordination-scale corpora (up to ~10⁴ items).  Larger corpora
    should plug an external store (pgvector, qdrant, etc.) in front
    of the same ``EmbeddingProvider``.

## Scientist — sparse symbolic law discovery as a runtime primitive

Every other learning primitive in this stack fits a model whose *form*
is fixed before fitting begins.  `Forecaster` fits a parametric mean
process.  `Predictor` mixes a fixed class of variable-order Markov
models.  `Filterer` runs Bayesian state estimation against a *known*
linear-Gaussian dynamics.  None of them returns an interpretable,
closed-form *law* — a finite arithmetic expression an investor, a
domain expert or a downstream verifier could read on a slide.

`agi.scientist.Scientist` closes that gap.  Given a stream of
`(x ∈ ℝᵈ, y ∈ ℝ)` pairs it discovers a sparse linear combination of
*symbolic basis functions* — monomials, sines/cosines, exponentials,
logarithms, plus any user-supplied callable — that explains `y` as a
function of `x`, and returns a `Law` object carrying

* a printable closed-form expression (`"y ≈ -4.905·x0² + 3.00·x0 + 100.00"`);
* per-coefficient bootstrap 95% confidence intervals (Efron 1979);
* per-term stability-selection inclusion frequencies (Meinshausen-Bühlmann 2010);
* AIC (Akaike 1973), BIC (Schwarz 1978), and MDL (Rissanen 1978) ranking;
* in-sample R², out-of-sample R² on a held-out set, and Akaike-corrected
  small-sample AICc;
* the full **Pareto frontier** of (complexity, residual) so the coordinator
  can route on the bias / sparsity tradeoff explicitly;
* a SHA-256 fingerprint chain over every `observe` / `fit` / `report` call,
  compatible with `AttestationLedger`.

```python
from agi import Scientist, SCIENTIST_SELECT_AIC

sci = Scientist.create(input_dim=1, max_degree=3, seed=0)
for t, y in falling_body_data:           # noisy (time, altitude) pairs
    sci.observe([t], y)
law = sci.fit(criterion=SCIENTIST_SELECT_AIC)
print(law)
# y ≈ 99.9437 + 3.04983·x0 − 4.91241·x0^2          ← gravity recovered

# Pareto front: complexity vs. residual
for p in sci.pareto():
    print(p.k, p.lam, p.rss, p.law)

# 95% bootstrap CIs and stability selection
boot = sci.bootstrap(law=law, n_resamples=200)
stab = sci.stability_selection(n_resamples=100, pi_thr=0.6)

# Out-of-sample R² on held-out data
sci.evaluate_r2(test_xs, test_ys, law=law)
```

### Algorithms

* **STLSQ — Sequential Thresholded Least Squares.**  Brunton-Proctor-Kutz
  2016 *Discovering governing equations from data by sparse identification
  of nonlinear dynamical systems*.  Alternate least-squares with hard
  thresholding `|ξ_j| < λ`; the fixed point is the ℓ⁰-constrained
  projection of OLS onto the basis subset that survives the threshold.
  `Scientist` sweeps a grid of `λ` and exposes the full Pareto front.
* **AIC.**  Akaike 1973 *Information theory and an extension of the
  maximum likelihood principle*.  `AIC = n·log(RSS/n) + 2k`.
* **BIC.**  Schwarz 1978 *Estimating the dimension of a model*.
  `BIC = n·log(RSS/n) + k·log(n)`; consistent for the sparse support
  as `n → ∞`.
* **MDL.**  Rissanen 1978 *Modeling by shortest data description*.
  Two-part code length in bits-per-sample, length-independent so it
  composes with `Compressor` on the universal-codelength side.
* **Bootstrap CI.**  Efron 1979 *Bootstrap methods: another look at
  the jackknife*.  Empirical percentile interval at the selected
  support.
* **Stability selection.**  Meinshausen-Bühlmann 2010 *Stability
  selection*.  Resample, refit, count inclusion frequency; controls
  per-family error under mild exchangeability.
* **Pareto knee.**  Satopää-Albrecht-Irwin-Raghavan 2011 *Finding a
  Kneedle in a Haystack: Detecting Knee Points in System Behavior*.
  Maximum-distance-from-chord elbow rule on the (complexity, log RSS)
  curve.

### How it composes with the rest of the runtime

| Question                                                                | Composition                                                                               |
|--------------------------------------------------------------------------|--------------------------------------------------------------------------------------------|
| Did the discovered law generalise?                                       | `ConformalPredictor` wraps `Scientist.predict` for finite-sample prediction intervals.    |
| Is the law a *false discovery*?                                          | `Refuter` runs metamorphic / boundary attacks against `Law.predict`.                       |
| What's the chance another law fits as well?                              | `sci.akaike_weights()` returns posterior weights over the Pareto frontier.                 |
| Is the law numerically stable under data perturbation?                   | `sci.stability_selection()` — Meinshausen-Bühlmann inclusion frequencies.                 |
| Encode the law as a tool the agent can call.                             | `Synthesizer` lifts `Law.predict` into a typed `Tool`.                                     |
| Combine the law with a CTW universal predictor.                          | `Compressor` benchmarks Law's MDL bits/sample against `Predictor` codelength.              |
| Plug the discovered dynamics into a state filter.                        | `Filterer` consumes `Law.predict` as a non-linear `f`; particle filter resolves the noise. |
| Cross-validate the law on N held-out folds, parallel.                    | `Pool` shards `evaluate_r2` calls across runtimes; `Capabilities` routes folds by latency.|
| Persist the discovered Law in a knowledge graph.                         | `KnowledgeGraph.add_fact(law_id, "explains", target, weight=law.r2)`.                      |
| Refuse to act on a law whose CI crosses zero on the leading coefficient. | `Quantilizer.act` gates on `boot.contains_zero(name) is False`.                            |
| Replay a Law's full derivation byte-for-byte.                            | `AttestationLedger` consumes the fingerprint chain that hashes every `observe` / `fit`.   |

### Investor framing

A coordination engine that can call `Scientist.fit(observation_stream)`
closes a loop none of the other primitives close: from *observations*
to *interpretable mechanism*.  `Forecaster` predicts the next number;
`Filterer` tracks a latent state; `CausalDiscoverer` finds an arrow.
Only `Scientist` returns the **formula**.  That formula is then:

* an audit artefact a regulator can read;
* a hypothesis `Refuter` can try to break;
* a closed-form prior `Filterer` can plug into its dynamics;
* a typed program `Synthesizer` can lift into a tool;
* a step in a `KnowledgeGraph` fact whose edges carry numerical
  coefficients.

When the slide says "the AI discovered the law for falling bodies from
80 noisy observations in 200 milliseconds, with a 95 % bootstrap CI
that excludes zero on every term and an MDL certificate bounding the
description length to 0.1 bits per sample" — `Scientist` is the
primitive doing the work.

### What it deliberately doesn't claim

  * A general symbolic regressor.  The search is *linear-in-basis*:
    coefficients are real-valued, the structure of the law is a
    selection from a fixed library.  Genetic-programming-style
    expression-tree search (PySR, Eureqa) is more general but more
    expensive and lacks the closed-form guarantees STLSQ + AIC carry.
  * A causal discovery primitive.  A discovered law fits observational
    data; whether it is a *causal* law is what `CausalDiscoverer` and
    `Refuter` are for.

## Conjecturer — automated mathematical conjecture generation as a runtime primitive

Every other learning primitive in this stack returns a **parametric**
mechanism — real-valued coefficients on a fixed basis (`Scientist`), a
stochastic next-symbol distribution (`Predictor`), a posterior over
latent state (`Filterer`).  None of them returns an **integer-coefficient
identity** — the kind of statement a working mathematician would call a
*conjecture*: ``φ² − φ − 1 = 0``, ``π = 16·arctan(1/5) − 4·arctan(1/239)``,
``ζ(2) = π²/6``.  An AI system that aspires to *discover laws* — not just
fit functions — must also produce these.

`agi.conjecturer.Conjecturer` closes that gap.  Given a stream of
high-precision numerical observations and a set of named constants, it
searches the lattice of integer linear combinations for one whose
numerical value is indistinguishable from zero at the working precision —
and re-evaluates each candidate at *doubled* precision to reject
spurious matches.  Each surviving conjecture ships with

  * a printable closed-form (`"phi2 −phi −one = 0"`);
  * the integer coefficients ``mᵢ ∈ ℤ`` of the relation;
  * the residual ``|Σ mᵢ · vᵢ|`` at the working precision;
  * a *false-discovery* bound — the Bonferroni probability that a
    relation of equal precision could be found by chance in the search
    cube ``{‖m‖_∞ ≤ M}``;
  * a *verification record* — the doubled-precision residual, which
    must shrink consistently with the precision increase if the
    identity is genuine;
  * a SHA-256 fingerprint chain over every `observe` / `propose` /
    `verify` / `report` call, compatible with `AttestationLedger`.

```python
from agi import Conjecturer
import math

cj = Conjecturer.create(precision_digits=14, seed=0)
phi = (1 + math.sqrt(5)) / 2
cj.observe("phi", phi)
cj.observe("phi2", phi * phi)
cj.with_constants(("phi2", "phi", "one"))
for c in cj.propose(max_coeff=3):
    print(c.signature, "  residual=", float(c.residual))
# phi2 −phi −one = 0   residual= 0.0
```

```python
# Machin's formula recovered from float values of arctan(1/5), arctan(1/239)
cj = Conjecturer.create(precision_digits=14)
cj.observe("a", math.atan(1/5))
cj.observe("b", math.atan(1/239))
cj.with_constants(("pi", "a", "b"))
print(cj.propose(max_coeff=20)[0].signature)
# pi −16·a +4·b = 0
```

```python
# Single-constant closed-form recognition
cj = Conjecturer.create(precision_digits=14)
recs = cj.recognize_constant((1 + math.sqrt(5))/2, basis=("one", "sqrt5"))
print(recs[0].expression)        # (one +sqrt5)/2
```

### Algorithms

* **PSLQ — Ferguson-Bailey 1992** *A polynomial time, numerically stable
  integer relation algorithm*.  Given a real vector finds either an
  integer relation or a certificate of its absence.
* **LLL — Lenstra-Lenstra-Lovász 1982** *Factoring polynomials with
  rational coefficients*.  Lattice basis reduction with worst-case
  guarantee ``‖b₁‖ ≤ 2^{(n-1)/4} λ₁(L)``.  Implemented in **exact
  rational arithmetic** (Python `fractions`) with incremental
  Gram-Schmidt updates so the answer is determined by the input
  precision alone.
* **Continued fractions — Khinchin 1935; Lochs 1964**.  Best rational
  approximations under denominator budgets, with auto-truncation
  before "huge" quotients (the empirical signal that the irrational
  tail has decayed below working precision).
* **Stern-Brocot tree.**  Binary descent for best rational with
  denominator ``≤ D`` (Hardy-Wright 1979 §3.7).
* **Ramanujan Machine — Raayoni et al. 2021** *Nature 590, 67–73*.
  Numerical search → high-precision re-evaluation; only matches that
  survive precision-doubling are reported.  `Conjecturer.verify`
  implements exactly this discipline.
* **Bonferroni FDR control.**  Search space ``(2M+1)ⁿ`` candidates;
  each candidate's chance to be spurious at precision ``d`` is
  ``≤ 2 · 10^{-d}``.  Per-conjecture false-discovery bound shipped
  with every result.
* **Plouffe's Inverse Symbolic Calculator (1995).**  Closed-form
  recognition for a single real constant via LLL against an open
  registry of named constants — extensible at runtime.

### How it composes with the rest of the runtime

| Question                                                                  | Composition                                                                              |
|----------------------------------------------------------------------------|------------------------------------------------------------------------------------------|
| Is this measured constant a rational?                                      | `cj.recognize_constant(x)` returns CF + Stern-Brocot best rational with denom budget.    |
| Is it a known transcendental combination?                                  | `cj.recognize_constant(x, basis=("pi","e","gamma",…))` runs LLL over the basis.          |
| Does my fitted `Scientist.Law` have integer coefficients in disguise?      | Feed `law.coefficients` into `cj.observe(…)` + `cj.propose()`; integer relation ⇒ yes.   |
| Did my conjecture survive higher-precision evaluation?                     | `cj.verify(conjecture, factor=2)` re-runs at doubled precision; sets `verified` flag.    |
| Replay the discovery byte-for-byte for an audit.                           | `AttestationLedger` consumes the SHA-256 chain on every observe / propose / verify.      |
| Refuse to act on a conjecture whose FDR bound exceeds my threshold.        | `Quantilizer.act` gates on `conjecture.fdr_bound ≤ ε`.                                   |
| Plug a verified identity as an exact constraint into a state filter.       | `Filterer` substitutes ``Σ mᵢ vᵢ = 0`` as an algebraic invariant on the latent state.    |
| Turn a verified identity into a typed compile-time constant in a tool.     | `Synthesizer` lifts `Conjecture.coeffs` into a static field of a generated Tool.         |
| Record the identity as an edge with **integer** weight in a knowledge graph| `KnowledgeGraph.add_fact(c.signature, "holds", target, weight=c.coeffs)`.                |

### Investor framing

A coordination engine that can call `Conjecturer.propose(observations)`
closes a loop none of the other primitives close: from *numbers* to a
*combinatorial closed-form identity*.  `Scientist` returns a real-valued
formula; `Conjecturer` returns an *integer-coefficient* formula whose
discovery survives precision-doubling — the discipline that distinguishes
a genuine identity from a numerical coincidence.

When the slide says "the AI rediscovered Machin's formula from float-
precision values of two arctangents, verified it at 30 decimal digits,
and produced a SHA-256 audit chain that lets a regulator replay the
discovery byte-for-byte" — `Conjecturer` is the primitive doing the
work.  It is the runtime substrate for *AI for mathematics* (FunSearch,
Ramanujan Machine, AlphaProof) reduced to a single composable call.

### What it deliberately doesn't claim

  * A general theorem prover.  An integer relation is a *conjecture* —
    a candidate identity numerically consistent at the working
    precision.  Formal proof of the relation requires a downstream
    proof assistant (Lean, Coq, Isabelle).  `verify` raises the
    *evidence*, not the *certainty*, of the claim.
  * A free-form symbolic regressor.  The search is *linear-in-columns*:
    the user supplies the columns (built-in constants + observations),
    the primitive finds short integer coefficients.  Quotient or
    exponential combinations require extending the column set.
  * A high-precision arithmetic library.  Built-ins ship at ≤ 100
    decimal digits; for higher precision the user supplies their own
    evaluator via the `builtin_constants={"name": fn}` constructor
    parameter.

## Solver — CDCL satisfiability as a runtime primitive

Every other primitive in this stack returns a *statistical* answer —
a posterior, a forecast, a calibrated prediction interval, a ranked
list of conjectures, a sparse linear law.  Statistical answers carry
probabilities and finite-sample error bars but **never** a logical
certificate.  A coordination engine driving safety-critical actuation
needs the complementary capability: given a Boolean specification
``φ`` over discrete decision variables, return either

  * a satisfying assignment ``α`` with ``α ⊨ φ`` — a concrete plan, an
    actuator setting, a hardware configuration — together with a
    machine-checkable confirmation that ``φ(α) = ⊤``;
  * a **proof of unsatisfiability** — a sequence of resolvents that
    derives the empty clause from ``φ`` — guaranteeing that no
    satisfying assignment exists, end of discussion.

`agi.solver.Solver` is the runtime primitive that closes this gap.
It is a **from-scratch Conflict-Driven Clause Learning (CDCL)** SAT
solver with every refinement the SAT-competition-winning solvers of
the last twenty years have settled on:

  * two-watched-literal unit propagation;
  * 1-UIP conflict analysis with self-subsuming-resolution clause
    minimisation (Sörensson-Biere 2009);
  * VSIDS variable activity with multiplicative decay
    (Moskewicz et al. 2001);
  * phase saving (Pipatsrisawat-Darwiche 2007);
  * Glucose-style LBD clause deletion (Audemard-Simon 2009);
  * Luby restart schedule (Luby-Sinclair-Zuckerman 1993);
  * incremental SAT under assumptions, with deletion-based MUS
    extraction (Belov-Marques-Silva 2012);
  * UNSAT-core MaxSAT via selector-relaxation and cardinality
    expansion (Fu-Malik 2006);
  * DRAT-style RUP proof emission and an embedded **re-checker** so
    every UNSAT verdict ships with a machine-verifiable certificate
    (Wetzler-Heule-Hunt 2014);
  * SHA-256 attestation chain over every `add_clause`, `assume`,
    `solve`, `extract_mus`, `report` — compatible with the rest of the
    runtime's `AttestationLedger`.

```python
from agi import Solver

sv = Solver.create(seed=0)
sv.reserve_vars(3)
sv.add_clause([1, 2, -3])
sv.add_clause([-1, 3])
sv.add_clause([-2, -3])
r = sv.solve()
print(r.status)        # → "sat"
print(dict(r.model))   # → {1: True, 2: False, 3: True}
```

```python
# Pigeonhole 5→4 — UNSAT, with a DRAT proof verified by the runtime
sv = Solver.create()
sv.reserve_vars(20)
def x(i, j): return (i-1)*4 + j
for i in range(1, 6):
    sv.add_clause([x(i, j) for j in range(1, 5)])
for j in range(1, 5):
    for i1 in range(1, 6):
        for i2 in range(i1+1, 6):
            sv.add_clause([-x(i1, j), -x(i2, j)])
r = sv.solve()
print(r.status)            # → "unsat"
print(sv.check_proof())    # → True   (DRAT proof self-verified)
```

The cardinality layer encodes ``Σ lits ≤ k``, ``≥ k``, or ``= k`` via
Sinz's sequential counter (Sinz 2005) — ``O(n·k)`` clauses, arc-
consistent under unit propagation:

```python
sv = Solver.create()
sv.reserve_vars(64)
def x(r, c, v): return ((r-1)*4 + (c-1))*4 + v
# 4×4 Sudoku: each cell / row / col / box has each value exactly once.
for r in (1,2,3,4):
    for c in (1,2,3,4):
        sv.add_exactly([x(r,c,v) for v in (1,2,3,4)], 1)
# … and so on for rows, columns, 2×2 boxes …
sv.assume(x(1,1,1))
sv.assume(x(2,3,3))
print(sv.solve().status)   # → "sat"
```

The DSL layer (`var`, `land`, `lor`, `lnot`, `ximp`, `xeqv`, `xite`,
`at_most`, `at_least`, `exactly`) compiles to CNF via the Tseitin
transformation; fresh auxiliary variables are allocated strictly
above the user-reserved range, so user-named variables and internal
encodings never collide:

```python
from agi.solver import var, land, lor, lnot, ximp, exactly

a, b, c = var(1), var(2), var(3)
sv = Solver.create()
sv.add_formula(land(lor(a, b), ximp(a, c)))   # asserts (a∨b) ∧ (a→c)
sv.add_formula(exactly(2, [a, b, c]))         # exactly two of {a,b,c}
print(sv.solve().status)                      # → "sat"
```

UNSAT-core extraction and MaxSAT round out the picture:

```python
# Minimal Unsatisfiable Subset over the assumption set
sv.assume(-3); sv.assume(1); sv.assume(2)     # only -3 is essential
sv.solve()
print(sv.extract_mus())                       # → (-3,)

# Weighted MaxSAT: minimum-weight violation under hard constraints
sv = Solver.create(); sv.reserve_vars(3); sv.add_clause([1, 2, 3])
cost, model, violated = sv.solve_max_sat(
    soft=[[-1], [-2], [-3]],
    weights=[10, 1, 1],
)                                             # falsify a cheap soft
print(cost, violated)                         # → 1  (2,)
```

### Algorithms

* **Cook-Levin 1971/1973 — NP-completeness of SAT.**  Every decision
  problem in NP reduces in polynomial time to a SAT instance, which
  is why a fast SAT engine is a fast *general combinatorial reasoner*.
* **Davis-Putnam-Logemann-Loveland 1962 — DPLL.**  Unit propagation
  and pure-literal subsumption — the propagation skeleton of every
  modern CDCL solver.
* **Marques-Silva-Sakallah 1996 — GRASP.**  *Conflict analysis*: the
  1-UIP asserting-clause-plus-backjump engine.
* **Moskewicz-Madigan-Zhao-Zhang-Malik 2001 — Chaff / VSIDS + two-
  watched literals.**  Variable activity + watched-literal unit
  propagation — the data structures that made million-clause SAT
  instances tractable.
* **Audemard-Simon 2009 — Glucose / LBD.**  *Literal Block Distance*
  as the empirical correlate of learnt-clause usefulness; periodic
  deletion of high-LBD clauses keeps the database bounded.
* **Luby-Sinclair-Zuckerman 1993 — Universal restart schedule.**
  Worst-case-optimal Las Vegas restart sequence.
* **Sinz 2005 — Sequential cardinality encoding.**  ``O(n·k)`` clauses
  for ``Σ lits ≤ k``; arc-consistent under unit propagation.
* **Tseitin 1968 — Polynomial CNF transformation.**  Every
  propositional formula compiles to an equisat CNF in linear time
  via fresh auxiliary variables.
* **Fu-Malik 2006 — UNSAT-core MaxSAT.**  Iterated selector-relaxation
  cardinality tightening.
* **Belov-Marques-Silva 2012 — Deletion-based MUS extraction.**
  Worst-case linear in the assumption-set size.
* **Wetzler-Heule-Hunt 2014 — DRAT proof system.**  RUP-based clause
  addition with deletion records — the proof format every SAT
  competition since 2014 has adopted.

### How it composes with the rest of the runtime

| Question                                                                | Composition                                                                              |
|--------------------------------------------------------------------------|------------------------------------------------------------------------------------------|
| Does the synthesised tool's precondition hold in the current world?      | `sv.solve(precondition)` — SAT ⇒ a concrete world; UNSAT ⇒ a precondition obligation.    |
| Is my fitted `Scientist.Law` falsifiable by a Boolean side-condition?    | Encode the sign / parity constraints; `Solver.solve(¬law_constraint)` — SAT ⇒ refutation.|
| What is the minimum-cost discrete plan under hard safety clauses?         | `Solver.solve_max_sat(soft, weights=utilities, hard=safety_clauses)`.                    |
| Which assumptions in my context window are *jointly* inconsistent?        | `sv.extract_mus()` — returns a smallest UNSAT subset, the "reason for inconsistency".    |
| Replay an UNSAT verdict byte-for-byte for a regulator.                    | `AttestationLedger` consumes the SHA-256 chain; `sv.check_proof()` re-verifies the DRAT. |
| Refuse to act on a Boolean obligation whose UNSAT proof exceeds budget.   | `sv.solve(max_conflicts=B)`; `Quantilizer.act` gates on `result.status == "sat"`.        |
| Encode a `Topologist` persistence-bar witness as a Boolean obligation.    | Tseitin-encode the witness predicate via `solver_var`, `solver_and`, `solver_at_most`.   |
| Verify the integer-coefficient `Conjecturer` identity has a sign pattern. | Encode the sign predicate as CNF, `Solver.solve` over a discrete sample space.           |

### Investor framing

When the slide says *"the AI handed the auctioneer a discrete bid
schedule together with a machine-checked proof that no other
schedule satisfies the regulator's hard constraints at strictly
lower cost"* — the SAT solver is the primitive doing the work.
``Solver.solve_max_sat`` returns the minimum-cost plan, ``check_proof``
re-verifies the UNSAT certificate of every cheaper alternative, and
the attestation ledger replays the derivation byte-for-byte.

The same primitive is what makes *"formally verified AI safety"*
more than a slogan.  ``Solver.solve(¬unsafe_state)`` either returns
SAT — the AI cannot enter that state — or a counter-example showing
exactly which inputs do.  No statistical caveat, no calibration
budget; a Boolean certificate.

### What it deliberately doesn't claim

  * Not an SMT solver.  Variables are pure Boolean; integer,
    bit-vector, array, or floating-point theories require a
    downstream Z3, cvc5, or Yices.
  * Not a parallel / portfolio solver.  Every search call is single-
    threaded; coordinators wanting portfolio behaviour should run
    ``Solver`` instances under :class:`Coordinator` and aggregate.
  * Not a probabilistic / weighted model counter.  Exact ``#SAT`` is
    ``#P``-complete and beyond the deterministic-decision contract
    of this primitive.
  * Not a competitive solver on the SAT-Race scale.  The
    implementation is in pure Python and intentionally readable.
    It is the *interface* and *audit chain* that are the
    contribution; a coordination engine can swap in an industrial
    back-end behind the same public API.

## Planner — SAT-compiled classical planning as a runtime primitive

A coordination engine driving discrete actuation needs to do more than
*react* to the current state — it needs to *plan* a sequence of
actions that achieves a goal.  None of the primitives shipped so far
returns a *plan*; what they return is a posterior, a forecast, a
discovered law, a satisfying assignment.

`agi.planner.Planner` closes that gap.  Given a STRIPS-style domain
(Boolean **fluents**, **actions** with positive/negative preconditions
and add/delete effects, an **initial state**, a **goal**), it returns
either

  * a **plan** — a finite sequence of actions whose deterministic
    execution from the initial state achieves the goal, together with
    an explicit horizon (length), an action cost, and a SHA-256
    attestation chain that a regulator can replay byte-for-byte;
  * or, given an explicit horizon bound, a **proof that no plan of
    bounded length exists** — the DRAT proof emitted by the underlying
    :class:`Solver` on the bounded plan-existence formula.

The `Planner` sits one composition layer above `Solver`: every plan
query is compiled to a SAT instance via the Kautz-Selman-1992
*SATPlan* encoding, dispatched to `Solver`, and the returned model
(or UNSAT proof) is decoded back into the planning vocabulary.

```python
from agi import Planner

pl = Planner.create(seed=0)
pl.add_fluent("at_A")
pl.add_fluent("at_B")
pl.add_action("move_AB", pre=["at_A"], add=["at_B"], delete=["at_A"])
pl.set_initial({"at_A": True})
pl.set_goal({"at_B": True})
plan = pl.solve()
print(plan.actions)        # → ("move_AB",)
print(plan.horizon)        # → 1
print(plan.cost)           # → 1
```

```python
# Tour: visit A, B, C, D starting at A
for loc in "ABCD":
    pl.add_fluent(f"at_{loc}")
    pl.add_fluent(f"v_{loc}")
for X in "ABCD":
    for Y in "ABCD":
        if X != Y:
            pl.add_action(f"go_{X}_{Y}",
                          pre=[f"at_{X}"],
                          add=[f"at_{Y}", f"v_{Y}"],
                          delete=[f"at_{X}"])
pl.set_initial({"at_A": True, "v_A": True})
pl.set_goal({f"v_{X}": True for X in "ABCD"})
plan = pl.solve()
# horizon = 3 — minimum number of moves to visit B, C, D from A
```

```python
# Parallel mode: non-interfering actions co-fire at the same step
pl.solve(parallel=True).parallel_steps  # → (("do_a","do_b","do_c","do_d"),)
```

The relaxed-reachability machinery is exposed as composable building
blocks: ``reachable_fluents`` returns the delete-relaxed-reachable
set, ``h_max`` returns a Bonet-Geffner-2001 lower bound on plan
length, ``relaxed_plan`` returns a greedy heuristic plan in the
delete-relaxed problem.

```python
pl.h_max()              # → 0..∞   lower bound on optimal horizon
pl.reachable_fluents()  # → frozenset of delete-relaxed reachable fluents
pl.relaxed_plan()       # → heuristic action sequence (FF-style)
```

### Algorithms

* **Fikes-Nilsson 1971 — STRIPS.**  Actions are
  ``(precondition, add-effect, delete-effect)`` triples over a
  finite Boolean state space; the canonical classical-planning
  model.
* **Kautz-Selman 1992 — Planning as satisfiability.**  Bounded-length
  plan existence is equivalent to satisfiability of a CNF whose
  size grows linearly in horizon × |actions|.
* **McCain-Turner 1997 — Explanatory frame axioms.**  Reduces the
  encoding by a factor of |fluents| versus the original frame
  axioms; `Planner` ships exactly this form.
* **Blum-Furst 1997 — Graphplan / mutex propagation.**  The layered
  reachability structure underlying the ``h_max`` heuristic and
  the bounded-horizon iteration schedule.
* **Bonet-Geffner 2001 — h_max heuristic.**  Cost-of-cheapest-
  achievement over the delete-relaxed graph; the lower bound on
  optimal plan length used to *start* iterative deepening at the
  earliest feasible horizon.
* **Hoffmann-Nebel 2001 — FF / relaxed plan extraction.**  Greedy
  regression through the layered structure for an upper-bound
  heuristic plan.
* **Rintanen 2012 — Madagascar parallel-action planning.**  Multiple
  non-interfering actions co-fire at the same timestep — the
  encoding ``parallel=True`` uses.

### How it composes with the rest of the runtime

| Question                                                                | Composition                                                                              |
|--------------------------------------------------------------------------|------------------------------------------------------------------------------------------|
| What sequence of discrete actions achieves my goal under hard safety?    | `Planner.solve()` — returns a plan of minimum horizon; the underlying SAT solver       produces a DRAT certificate of no-shorter-plan-exists. |
| What is the lower bound on action cost before the agent commits?         | `Planner.h_max()` — Bonet-Geffner lower bound; the coordinator gates on this.            |
| Refuse to act unless the plan is *optimal*.                              | `Planner.solve_optimal()` ⇒ horizon equals the underlying lower bound.                   |
| Is the goal *provably* infeasible?                                       | `Planner.h_max()` raises `GoalUnreachable` when the relaxed-reachable set excludes it.   |
| What's the maximum-parallelism schedule of the plan?                     | `Planner.solve(parallel=True).parallel_steps` — Rintanen-style mutex-respecting layers.  |
| Replay the plan derivation byte-for-byte for a regulator.                | `AttestationLedger` consumes the SHA-256 chain; the underlying `Solver.check_proof`       re-verifies the bounded-horizon UNSAT witness. |
| Compose with `Synthesizer` to lift each action into a typed tool call.   | `Synthesizer.compile(plan.actions)` → an executable tool-call trace.                      |
| Compose with `Quantilizer` to gate execution on plan optimality.         | `Quantilizer.act` checks `plan.horizon == pl.h_max()` before dispatch.                    |

### Investor framing

When the slide says *"the agent returned an executable 7-step plan
for the warehouse robot, together with a machine-checkable proof
that no 6-step plan exists under the operator's hard-safety
constraints"* — `Planner.solve_optimal` is the primitive doing the
work.  The plan is the deliverable; the UNSAT certificate of every
shorter horizon is the *audit*.

For multi-agent coordination, `Planner` is the *discrete control
plane* of every actuator the runtime drives: the LLM proposes a
goal, `Planner` returns the minimum-cost executable sequence
respecting hard safety clauses, `Coordinator` dispatches the steps
in parallel, and `AttestationLedger` records every state transition
end-to-end.

### What it deliberately doesn't claim

  * Not a numeric / hybrid / temporal planner — fluents are pure
    Boolean; numeric resources, durative actions, or timed initial
    literals require a downstream PDDL+ engine.
  * Not a probabilistic / MDP planner — the underlying solver is
    deterministic Boolean SAT.  For probabilistic planning use
    :class:`~agi.coordinator.Coordinator` over a learnt world model.
  * Not a competitive IPC planner.  The implementation is in pure
    Python and intentionally readable; a coordination engine
    wanting a high-performance back-end swaps in Fast Downward or
    Madagascar behind the same public API.

## Inducer — Levin universal search as a runtime primitive

Every primitive in this runtime *executes* programs; **Inducer searches
the space of all programs.**  Where `Synthesizer` commits to a typed
DSL and `Solver` to a CNF, Inducer commits to *no model class at all* —
only to a small stack-based universal VM — and pays the Levin-search
bound for that generality.  This is the operation that lets the
coordination engine ask *"is there any program at all that explains
this trace?"* without first asking *"what kind of program?"*.

The universal VM has 15 active opcodes packed into a 4-bit alphabet
(`HALT`, `PUSH0/1/2/-1`, `INP`, `DUP/SWP/DRP`, `ADD/SUB/MUL/MOD/NEG`,
`JNZ`).  Programs are enumerated lexicographically; the discovered
program comes with a **Kraft mass** (universal-prior contribution
`2^{-l(p)}`), a **Levin complexity** (`l(p) + log₂ t(p)`, a
constructive upper bound on `Kt(spec)`), an **Occam PAC bound**
(Blumer-Ehrenfeucht-Haussler-Warmuth), and a **SHA-256 certificate**
chaining VM, spec, and program.

::

    >>> from agi.inducer import Inducer, InducerConfig, Spec, induce
    >>> rep = induce([(2, 4), (3, 9), (5, 25), (7, 49)])  # n -> n²
    >>> rep.program.disassemble()           # "INP DUP MUL"
    >>> rep.universal_prior_mass()          # 0.125  (= 2^-3)
    >>> rep.levin_complexity()              # 12.82 bits
    >>> rep.occam_bound(delta=0.05)         # PAC ε for held-out generalisation
    >>> rep.eval([13])                       # 169
    >>> rep.certificate                      # tamper-evident SHA-256

Two search regimes:

  * **Iterative deepening (default)** — enumerate every program of
    length 1, 2, ..., L with a fixed per-program step budget.
    Deterministic, exhaustive, returns the *minimum-length* solver.

  * **Levin universal search** (`mode="levin"`) — Levin's 1973
    dovetail: phase `i` allocates total budget ``T_i``, and a program
    of length ``L`` runs for ``⌊T_i / 2^L⌋`` steps.  If any program
    ``p*`` of length ``L*`` solves the spec in ``T*`` steps, Inducer
    finds *some* solver in total cost ``R(p*) ≤ K_U · 2^{L*} · T*``
    (Levin 1973, Theorem 1).  This is the strongest worst-case bound
    on program search known to be achievable.

A Solomonoff model average — `agi.inducer.kraft_normalised_posterior`
— turns a list of consistent programs into a Kraft-normalised
posterior, the prior a coordination engine should feed into Predictor
or Forecaster for downstream calibration.

Composition with the rest of the runtime:

  * **Predictor** — Predictor's CTW estimates the universal prior over
    *next symbols*; Inducer estimates it over *programs*.  Feed
    Inducer's discovered program back into Predictor's prior to
    accelerate compression on the same source.
  * **Synthesizer** — Synthesizer searches a typed DSL with PAC
    bounds; Inducer searches the *unrestricted* universal VM with the
    Levin bound.  A coordination engine picks Inducer when the DSL is
    unknown, Synthesizer when the DSL is known.
  * **Compressor** — Inducer's program length on a string ``x`` is an
    *upper bound* on ``Kt(x)`` and feeds NCD with tighter (because
    constructive) numerators.
  * **Conjecturer** — every Inducer-discovered program is a candidate
    *generative law* for an observation set; Conjecturer's e-value
    framework lifts the program to a falsifiable law.

What it does *not* do (yet):

  * No neural-guided search (Schmidhuber's *OOPS* prior bias).  The
    enumeration order is pure lexicographic; a learned prior over
    opcodes is the obvious next acceleration.
  * No partial-evaluation pruning across examples.  Each (program,
    example) pair is run independently.
  * No string / list opcodes — the VM operates on integers only.  Add
    typed opcodes to extend the language.

## Verifier — LCF-style proof certificate kernel as a runtime primitive

The other primitives in this runtime each *produce* a certificate
alongside their answer — Reasoner emits a resolution refutation, Solver
emits a DRUP-style unsat trace, Synthesizer emits a CEGIS verification
report, Conjecturer emits a falsifiable derivation. Re-checking those
certificates *inside the primitive that generated them* gains no
independence: a bug in Reasoner's proof reconstruction silently passes
Reasoner's own verifier. The `Verifier` primitive is the runtime's
**independent kernel** for those certificates — the **LCF discipline**
(Milner 1972; Pollack 1998 *de Bruijn criterion*) reduced to a single
runtime call.

The trust model is the classical LCF one (HOL, Isabelle, Coq, Lean):

  * a tiny **kernel** with a fixed enumeration of ~22 inference rules
    (1 resolution + 20 natural-deduction + 1 rewrite-at-position);
  * every certificate is a sequence of kernel calls; the verifier
    only ever applies a kernel call, never a derived rule, never
    "trusts" a cached lemma;
  * the *trusted computing base* is therefore exactly the kernel —
    ~250 lines, auditable in an afternoon — and *nothing else*.

Three proof systems are shipped, each with its own re-derivation path:

  * **Resolution proofs over CNF** (Robinson 1965; Goldberg-Novikov
    2003 *RUP*; Heule-Hunt-Wetzler 2013 *DRAT*).  Every step claims
    that two parent clauses (original or earlier resolvent) resolve
    on a positive-variable pivot to a stated resolvent; the kernel
    re-runs the resolution and matches structurally.  Verification
    succeeds iff every step's resolvent matches and the final
    resolvent is the empty clause (⊥).
  * **Natural-deduction proofs over propositional logic**
    (Gentzen 1935; Prawitz 1965).  Each step names one of the
    twenty kernel rules — ``assumption``, ``premise``,
    ``and_intro/elim_{l,r}``, ``or_intro_{l,r}``, ``or_elim``,
    ``imp_intro/elim`` (modus ponens), ``not_intro/elim``,
    ``bot_elim`` (ex falso quodlibet), ``iff_intro/elim_{l,r}``,
    ``lem`` (classical excluded middle), ``dne`` (double-negation
    elimination), ``top_intro``, ``repeat`` — and the kernel
    re-derives the resulting sequent ``Γ ⊢ φ``.  Verification
    succeeds iff every kernel re-derivation succeeds, the final
    formula matches the goal, and no undischarged assumptions
    remain outside the global premise set.  An
    `enforce_intuitionistic=True` flag rejects ``lem`` and ``dne``
    so a coordination engine can ask the stricter question "does
    this hold constructively?"
  * **Equational rewriting proofs** (Birkhoff 1935; Knuth-Bendix
    1970; Baader-Nipkow 1998).  Axioms are pairs ``ℓ = r`` with
    optionally declared variables; each step rewrites the current
    term at a given position by the chosen axiom in a chosen
    direction (forward or backward) under a given substitution.
    The kernel matches the substituted axiom side against the
    sub-term at the position and replaces it; verification succeeds
    iff the final term equals the target ``rhs``.

Every report carries:

  * `status`: one of ``VERIFIED`` / ``FAILED`` / ``MALFORMED``;
  * `failed_step`: 0-based index of the first failing step (None on
    success);
  * `failure_reason`: human-readable, includes the kernel rule's own
    error message ("pivot variable 5 not present in clause [1, 2, 3]",
    "imp_intro: discharged formula p not in premise context", etc.);
  * `certificate`: HMAC-SHA256 over the canonical serialisation of every
    kernel step in proof order — re-runnable by `AttestationLedger`
    without re-executing the kernel;
  * `kernel_calls`, `tcb_lines`, `elapsed_seconds`: self-describing
    fields a coordinator can quote in an attestation ("this output is
    backed by 22 kernel rules across 250 lines of trusted code, verified
    in 1.4 ms over 47 kernel calls").

Composition is intentional: Reasoner's `last_resolution_proof()` plugs
directly into `Verifier.verify_resolution`; Conjecturer's derivations
plug into `Verifier.verify_natural_deduction`; Synthesizer's CEGIS
counter-examples can be promoted into ND derivations and re-verified;
`Driver` can gate any high-stakes return value behind
``Verifier.verify_*`` returning ``VERIFIED``, giving the coordination
engine a *hard* safety boundary that composes with conformal /
quantile / fuzz gates.

```python
>>> from agi import (
...     Verifier, VerifierConfig, VerifierCNFFormula,
...     VerifierResolutionProof, VerifierResolutionStep,
...     verifier_tcb_summary,
... )
>>> V = Verifier(VerifierConfig(hmac_key=b"runtime-attestation"))
>>> f = VerifierCNFFormula.of([[1, 2], [-1], [-2]])
>>> proof = VerifierResolutionProof((
...     VerifierResolutionStep(parents=(0, 1), pivot=1, resolvent=(2,)),
...     VerifierResolutionStep(parents=(3, 2), pivot=2, resolvent=()),
... ))
>>> rep = V.verify_resolution(f, proof)
>>> rep.status, rep.kernel_calls
('VERIFIED', 2)
>>> verifier_tcb_summary()["kernel_rule_count"]
22
```

The primitive is **stdlib-only**: no Z3, no Lean, no Coq, no
``hypothesis``.  Every kernel rule is one or two Python ``if``
statements; the inner loop is a flat ``for`` over the proof steps with
a small dispatch table.  Verification is linear in proof length —
millions of steps verify in well under a second on commodity hardware.

## Sketcher — bounded-memory streaming sketches as a runtime primitive

Every other primitive in this runtime quietly assumes that someone can
keep the whole stream in memory.  ``Predictor`` keeps every prefix,
``Forecaster`` keeps every prediction-target pair, ``DriftSentinel``
keeps a reference window of arbitrary size, ``Auditor`` keeps every
event, ``Calibration`` keeps every score.  At laboratory scale this
is fine.  At runtime scale — millions of events per second through a
coordination engine over weeks of autonomous operation — it is
fatal.  A production runtime cannot keep everything; it must keep a
**sketch** with provable, finite-sample-valid error bounds.

`Sketcher` is the runtime's **bounded-memory streaming primitive**.
Given a stream of items and a sketch kind, it returns an answer with
an explicit `(ε, δ)` error certificate, the exact byte count of the
state it consumed, and an HMAC over the canonical state for
tamper-evidence.  Eleven sketches ship in one module, every one of
them pure-stdlib:

  * **Misra-Gries (1982)** heavy-hitters with `k` counters —
    deterministic, every item with frequency above `N / (k + 1)`
    survives, additive error ≤ `N / (k + 1)` on every item.
  * **Count-Min Sketch (Cormode-Muthukrishnan 2005)** with optional
    **conservative update (Estan-Varghese 2003)** — ε-additive
    over-estimate with probability ≥ 1 − δ at shape
    `w = ⌈e/ε⌉, d = ⌈ln 1/δ⌉`.
  * **Count Sketch (Charikar-Chen-Farach-Colton 2002)** — signed-hash
    median estimator, *unbiased* point query with ℓ₂-norm error.
  * **AMS / tug-of-war (Alon-Matias-Szegedy 1996)** for the second
    frequency moment `F₂`.
  * **HyperLogLog (Flajolet-Fusy-Gandouet-Meunier 2007)** with
    **HLL++ linear-counting correction (Heule et al. 2013)** —
    cardinality estimation with relative standard error
    ≈ `1.04 / √(2^p)` using `2^p` one-byte registers.
  * **KLL (Karnin-Lang-Liberty 2016)** optimal mergeable quantile
    sketch — `ε ≈ √log(1/δ) / k` simultaneous rank error over every
    quantile, with weight-preserving pair-and-promote compaction
    and random-orphan handling for sorted-stream symmetry.
  * **Greenwald-Khanna (2001)** deterministic quantile sketch —
    additive rank error ≤ ε·N, no randomness.
  * **Vitter (1985) reservoir sampling** (Algorithm R) — uniform
    sample of size `k` from an unbounded stream.
  * **Efraimidis-Spirakis (2006) weighted reservoir** (A-Res) —
    weighted-without-replacement sample with inclusion probability
    proportional to weight.
  * **Bloom (1970) filter** — probabilistic set membership with
    one-sided false-positive rate matching the configured target.
  * **Exponential histogram (Datar-Gionis-Indyk-Motwani 2002)** —
    `ε`-relative-error sliding-window counting in
    `O((1/ε) log²(εN))` space.

The pitch reduced to a runtime call:

```python
>>> from agi.sketcher import Sketcher
>>> # Cardinality of a 200k-distinct-item stream in 4 KB of state
>>> sk = Sketcher.hll(precision=12)
>>> for x in range(200_000):
...     sk.update(f"item_{x}")
>>> sk.cardinality()
204533.0   # rel-error 2.3%; theoretical RSE ≈ 1.6%
>>> sk.report().n_bytes
4111
>>> sk.report().epsilon       # actual relative-standard-error guarantee
0.01625
```

Every sketch is **mergeable** where the underlying algorithm admits a
mergeable summary — Misra-Gries, Count-Min, Count-Sketch, HLL, KLL,
GK, Bloom, AMS-F2 — meaning a distributed coordination engine can
shard a stream across N workers, sketch independently, and combine
the sketches into one answer of the same asymptotic quality as a
serial sketch over the union:

```python
>>> workers = [Sketcher.hll(precision=14, seed=0) for _ in range(8)]
>>> for w in workers:
...     for _ in range(50_000):
...         w.update(some_id())
>>> union = Sketcher.hll(precision=14, seed=0)
>>> for w in workers:
...     union.merge(w)
>>> union.cardinality()
# distinct ids across all 8 shards, identical to a single-sketch run
```

The report carries:

  * `estimate`: kind-specific — an int for cardinality / count, a
    dict for quantiles, a list for samples / heavy-hitters;
  * `epsilon`, `delta`: the *actual* certificate values that follow
    from the sketch's configured shape, not the user's target;
  * `n_items`, `n_bytes`, `capacity`: measured state footprint a
    coordinator can quote to a memory-budget admission gate;
  * `mergeable`: whether this kind admits the distributed-shard merge;
  * `certificate`: HMAC over the canonical state for tamper-evidence
    (replayable by `AttestationLedger`).

Composition with the rest of the runtime is the design intent:

  * **CountMin → DriftSentinel** — low-memory drift detection over
    high-cardinality identifiers (per-id frequencies under
    bounded state).
  * **HyperLogLog → Auditor** — distinct-counts of compliance-
    relevant entities (users, models, datasets) in a long-lived
    audit ledger.
  * **MisraGries → Compressor** — heavy-hitter symbol weights as
    an empirical prior for MDL code-book construction.
  * **KLL → Forecaster** — streaming quantile-binned calibration
    histograms updated on every observation.
  * **Reservoir → ExperimentDesigner** — unbiased eval-pool
    sampling from a vastly larger candidate stream.
  * **Bloom → ToolSynth** — dedup of candidate synthesised programs
    without storing every hash.
  * **F2Sketch → CausalDiscoverer** — streaming approximation of
    second-moment-based mutual information.
  * **ExpHistogram → Forecaster** — sliding-window event counts for
    short-horizon predictions.

The primitive is **stdlib-only**: no NumPy, no probabilistic-counters
library, no fast hash dependency.  Hashes go through `hashlib.blake2b`
keyed by a per-salt 8-byte integer (cheap, pairwise-independent in
practice); PRNG seeds are scrambled through SplitMix64 before
xorshift so small consecutive seeds give strongly decorrelated streams
(important for federated sketching with worker-id-derived seeds).

## Analogist — structure-mapping analogical reasoning as a runtime primitive

Every other primitive in this runtime treats reasoning **within** a
domain.  ``Predictor`` predicts the next symbol of a single stream.
``Scientist`` recovers a closed-form law from a single table.
``Conjecturer`` proposes a single proposition.  ``Inducer`` searches
for one program that fits one specification.  But the core operation a
coordination engine performs when it lifts a lesson learned in one
ticket into the policy that handles the **next** ticket — the
operation a debugger performs when it recognises that the bug in
front of it has the same shape as a bug it has seen before, the
operation a researcher performs when she carries the structure of an
argument from fluid dynamics into traffic flow — is **analogy**.

`Analogist` is the runtime primitive that performs that operation.
Given two relational descriptions — a **base** (well-known, richly
structured) and a **target** (unfamiliar, possibly incomplete) — it
returns a small set of *global mappings*, each a one-to-one,
parallel-connected alignment between base and target objects, ranked
by a Structural Evaluation Score that rewards **systematicity**
(Gentner 1983): deep, interconnected relational structure beats
isolated attributes.  Each global mapping comes with a list of
**candidate inferences** — expressions present in the base whose
entities have already been mapped to the target, projected as
predictions about what *should* be true in the target if the analogy
is sound.

### Algorithms shipped

  * **SME (Falkenhainer-Forbus-Gentner 1989)**: the canonical
    structure-mapping engine.  Three stages: (1) enumerate local
    match hypotheses under tiered identicality, (2) score them with
    a Structural Evaluation Score that propagates *parental support*
    down the relation tree (`SES(child) += λ · SES(parent)`),
    (3) greedy best-first search over consistent unions of match
    hypotheses under the one-to-one and parallel-connectivity
    constraints.
  * **MAC/FAC (Forbus-Gentner-Law 1995)** retrieval: a fast content-
    vector dot-product (Many Are Called) selects a short-list from a
    long-term memory of cases; full SME (Few Are Chosen) ranks the
    short-list by structural similarity.  The cost profile that lets
    the runtime keep a large case base and still answer in bounded
    time.
  * **ACME (Holyoak-Thagard 1989)** as an alternative engine: a
    constraint-satisfaction network that relaxes structural,
    semantic, and pragmatic constraints simultaneously.  Selected
    via `analogist_acme()`.
  * **Copycat-style proportional analogy (Hofstadter 1985;
    Mitchell 1993)**: the small `ProportionalAnalogy` sub-primitive
    that solves letter-string `a:b :: c:?` problems by rule
    enumeration — the runtime's symbol-stream pattern-transfer
    operator.

The pitch reduced to a runtime call:

```python
>>> from agi.analogist import sme
>>> analogist = sme(hmac_key=b"secret")
>>> analogist.add_description("solar", [
...     ("cause",
...        ("attracts", "sun", "planet"),
...        ("revolves_around", "planet", "sun")),
...     ("greater", ("mass", "sun"), ("mass", "planet")),
...     ("greater", ("temperature", "sun"), ("temperature", "planet")),
...     ("yellow", "sun"),
... ])
>>> analogist.add_description("atom", [
...     ("cause",
...        ("attracts", "nucleus", "electron"),
...        ("revolves_around", "electron", "nucleus")),
...     ("greater", ("mass", "nucleus"), ("mass", "electron")),
... ])
>>> report = analogist.match("solar", "atom")
>>> dict(report.mappings[0].entity_map)
{'sun': 'nucleus', 'planet': 'electron'}
>>> report.mappings[0].inferences
((('greater', ('temperature', 'nucleus'), ('temperature', 'electron')),
  ('greater', ('temperature', 'sun'), ('temperature', 'planet'))),
 (('yellow', 'nucleus'), ('yellow', 'sun')))
```

The mapping is sound under two structural constraints — **one-to-one**
(no base object maps to two target objects and vice versa) and
**parallel connectivity** (matched relations have matched arguments,
recursively) — both of which are certified by the report.  A
coordinator that wants to admit the analogy into its policy has a
verifier; a coordinator that wants to reject it has a counter-example.

### How it composes with the rest of the runtime

  * The candidate inferences are *predictions*.  Hand them to
    `Refuter` for falsification — a coordinator that has refuted
    "yellow(nucleus)" on the canonical solar-atom analogy has
    demonstrated that mere-appearance transfer is unsound; the
    surviving inference "greater(temperature, nucleus, electron)"
    is the one to operationalise.
  * `Conformal` can wrap any single transferred prediction in a
    distribution-free coverage interval — turning "the analogy
    suggests X" into "the analogy suggests X with probability ≥ 95%
    that the true value lies in [lo, hi]".
  * `MAC/FAC` retrieval is the long-term-memory side of the
    coordinator's case base.  Combined with `Skills` /
    `SelfEvalBank`, the runtime can lift a successful skill from one
    ticket into a candidate skill for a structurally analogous one
    — *cross-ticket learning by analogy*.
  * The `score` decomposes by predicate kind (`higher_order`,
    `relation`, `function`, `attribute`).  A `Strategist` that
    weighs analogical evidence against direct evidence has the
    per-kind contribution to threshold on.
  * Every report carries an HMAC `certificate` over the canonical
    mapping; the `AttestationLedger` can replay an entire match
    byte-for-byte from the certificate alone, so a coordinator
    publishing a transferred lesson has a tamper-evident record
    that the analogy it acted on is the analogy it explains.

### Investor framing

Today's frontier LLMs are notoriously brittle on analogy benchmarks
(Raven's Progressive Matrices, ARC, Mitchell's letter-string
problems): they pattern-match on surface tokens but fail to align
*relational structure*.  Lovett & Forbus 2017 showed that classical
structure-mapping accounts for human performance on exactly the
problems on which LLMs collapse.  `Analogist` is the runtime's
**structural-alignment co-processor** — a deterministic, certificate-
producing primitive a coordination engine can call whenever a
language model needs a sound way to *transfer* a lesson across a
domain boundary.  It is the operational rendering of Hofstadter &
Sander's (2013) claim that analogy is not a peripheral cognitive
trick but the *core* of cognition, made callable at the same tier
as `Solver` and `Planner`.

### What it deliberately doesn't claim

  * `Analogist` does not perform *induction* over expressions — that
    is the job of `Inducer` (Levin universal search) and `Scientist`
    (sparse symbolic-law recovery).  Analogist transports an existing
    structure across a domain boundary; it does not invent the
    structure.
  * It is not a similarity metric.  Two descriptions can score
    arbitrarily low on cosine similarity and still admit a perfect
    SME mapping (Markman & Gentner 1993); the runtime exposes both
    answers separately so a coordinator can pick the right one.
  * The candidate inferences are *hypotheses*.  They are not
    asserted true; they are emitted into the runtime's verification
    pipeline (`Refuter`, `Conformal`, `Verifier`) so the
    coordination engine has an explicit checkpoint between "the
    analogy proposes X" and "the runtime claims X".

## Searcher — bounded-anytime certified tree search as a runtime primitive

Every other primitive in this runtime **consumes** a question.  Predictor
gets a stream and returns the next symbol; Solver gets a CNF and returns
SAT/UNSAT; Inducer gets a spec and returns a program.  But the operation
that decides **which question to ask next**, given a state, a set of
admissible actions, and a means of evaluating their consequences, is
*search*.  Search is the canonical primitive of every agent that acts
under uncertainty over a tree of options — **AlphaZero is search,
MuZero is search, Stockfish is search**, the A\* planner inside a
self-driving stack is search, and the move-list a debugger considers
in front of a bug is search.  `Searcher` is the runtime's *bounded,
anytime, certified* version of that operation, exposed as a single
primitive a coordination engine can drive with budgets it must respect.

The pitch reduced to a runtime call:

```python
>>> from agi import Searcher, SearcherConfig
>>> sv = Searcher(SearcherConfig(algorithm="puct", max_iterations=4096))
>>> report = sv.search(
...     root_state,
...     actions=lambda s: s.legal_moves(),
...     apply=lambda s, a: s.play(a),
...     terminal=lambda s: s.is_terminal(),
...     reward=lambda s: s.reward(),
...     policy_prior=lambda s, A: {a: 1/len(A) for a in A},
...     value=lambda s: 0.0,
... )
>>> report.best_action          # canonical recommended action at the root
>>> report.best_value           # search's value estimate
>>> report.principal_variation  # deepest sequence the search agreed on
>>> report.certificate          # SHA-256 chain of (parent, action, child) events
>>> report.budget_used          # nodes, time, peak depth — what the search consumed
>>> report.regret_bound         # algorithm-specific finite-time regret bound
```

### Algorithms shipped

Six families of search under a single `algorithm=` switch.  The default
is `"auto"` — pick the family that respects the supplied evaluator
signatures.

  * **A\*** (Hart-Nilsson-Raphael 1968) — best-first over `f = g + h`.
    Optimally efficient under a consistent heuristic.  `weighted=w`
    switches to **weighted A\*** (Pohl 1970) with a worst-case `w`-
    suboptimality bound.
  * **IDA\*** (Korf 1985) — iterative-deepening A\* with linear space.
  * **UCT** (Kocsis-Szepesvári 2006) — MCTS with the **UCB1** rule
    (Auer-Cesa-Bianchi-Fischer 2002).  Finite-time regret
    `O(K log T / Δ_min)`.
  * **PUCT** (Silver et al. 2017, AlphaGo Zero) — UCT with a *policy
    prior* `P(s,a)` added to the exploration term.  Reduces to UCT
    under a uniform prior; the AlphaZero recommendation `c_puct=1.25`
    is the default.  Optional **Dirichlet root noise** for self-play.
  * **Alpha-Beta** (McCarthy 1956 / Knuth-Moore 1975) with iterative
    deepening (Slate-Atkin 1977), transposition table (Greenblatt
    1967), history heuristic (Schaeffer 1989), and aspiration windows.
  * **Beam search** (Reddy 1977) with configurable width and score
    direction (`"value"` or `"cost"`).
  * **Branch-and-Bound** (Land-Doig 1960) — best-first with incumbent
    pruning.

### What "bounded, anytime, certified" means

  * **Bounded** — every algorithm exposes a uniform stop predicate
    over (wall-clock seconds, expansion count, node count, peak
    memory, deadline timestamp).  A coordinator with 30 ms left in
    its SLO budget passes that 30 ms and gets back the best decision
    the searcher could compute within it.  `report.budget_used` and
    `report.bound_hit` record exactly what was consumed and which
    bound (if any) fired.
  * **Anytime** — at every iteration the current best action and
    value are well-defined.  `report.history` records the
    `(iteration, best_action, best_value)` trajectory so a
    coordinator can detect convergence.
  * **Certified** — every report carries a SHA-256 chain over the
    canonical sequence of `(parent_key, action, child_key, evaluation,
    selected)` decisions.  Replaying the search against the same
    config, root, evaluators, and RNG seed reproduces the chain
    byte-for-byte.  Two processes that agree on the certificate
    agree on the search.  Optional HMAC under a `secret_key` for
    authenticated chains.
  * **Pure stdlib** — no NumPy, no Torch, no SciPy.  The same module
    runs inside a sandboxed coordinator, inside a CI worker, inside
    a 256 MB Lambda.

### How it composes with the rest of the runtime

Every evaluator (`actions`, `apply`, `terminal`, `reward`, `heuristic`,
`policy_prior`, `value`) is a Python callable the coordinator supplies
— *including other primitives in this runtime*.  Composition is the
whole point:

  * **`Predictor`** as `value` — CTW gives a calibrated leaf estimate
    over a symbol stream; PUCT then chooses where to spend the next
    sampling step.
  * **`Verifier`** as `terminal` — a proof-checker terminates the
    branch the moment the lemma is discharged; the certificate
    composes with `AttestationLedger`.
  * **`Solver`** as `apply` — for a SAT-encoded transition system,
    CDCL is the move generator; `Searcher` then drives the high-level
    plan search.
  * **`Analogist`** as `policy_prior` — retrieved structural mappings
    bias the prior toward actions that worked in analogous past
    cases.
  * **`Conformal`** wraps the leaf `value` in a distribution-free
    coverage interval — a coordinator can SLO on
    `P(true_value ∈ [lo, hi]) ≥ 1−α` per leaf.
  * **`Cartographer`** can use `Searcher`'s `report.regret_bound`
    as the "ZPD" signal: a task whose search produces a regret
    bound that just barely shrinks under more iterations is in the
    zone of proximal development.

### Investor framing

PUCT is the algorithmic core of every milestone-grade AI of the last
decade: AlphaGo Zero, AlphaZero, MuZero, Stockfish-NNUE, and modern
LLM-based code-search agents.  `Searcher` is the runtime's
**deterministic, certificate-producing rendering** of that core,
callable as a single in-process primitive that respects an SLO
budget and composes with every other primitive in the runtime.  It
turns "give the agent more thinking time" from a vague request into
a *measurable budget knob* a coordination engine can dial.

### What it deliberately doesn't claim

  * `Searcher` does not learn its own value or policy network — that
    requires gradient compute, which is the job of the learner track
    (LoRA SFT) and out of scope for the in-process runtime.  It will
    use a learned model the moment a coordinator passes one in as a
    callable.
  * It does not solve continuous-action MDPs out of the box; progressive
    widening (`Coulom 2007`) is shipped behind `progressive_widening=
    True` for the discrete-projection case.  True continuous-control
    integration (cross-entropy method, iLQR, MPPI) is a separate
    primitive.
  * The certificate proves *reproducibility*, not *correctness of the
    evaluators*.  A buggy `reward` produces a tamper-evident search
    over a buggy reward.  Compose with `Verifier` / `Refuter` to
    establish that the *evaluators* themselves are sound.

## Distiller — amortized policy/value distillation as a runtime primitive

Every other primitive in this runtime **computes** an answer.
`Searcher` runs PUCT on a fresh tree.  `Solver` decides a fresh CNF.
`Inducer` enumerates fresh programs.  The operation that takes the
*outputs* of those primitives — visit distributions over actions,
value estimates at states, accepted decisions on inputs — and
**compiles them into a cheap, callable model** so the next instance of
the same kind of question is answered in *amortized constant time* is
**distillation**.

Distillation is the operational mechanism behind every milestone-grade
self-improving AI of the last decade: **AlphaGo Zero distils
800-rollout PUCT into a single forward pass; MuZero distils a learned-
dynamics search into the same; expert iteration / DAgger distils a
slow expert into a fast student; algorithmic distillation distils a
*learning algorithm* into a frozen Transformer's activations**.  In
every case the search-or-oracle is the *teacher*, the parametric model
is the *student*, and the expected behaviour of the teacher under the
student's own distribution is the target.

`Distiller` is the runtime's *bounded, anytime, certified, stdlib*
version of that operation.  Composed with `Searcher`, it closes the
AlphaZero-style self-improvement loop **inside one Python process**,
without a GPU, without a deep-learning framework, without a tokenizer.

The pitch reduced to a runtime call:

```python
>>> from agi import (Searcher, SearcherConfig, Distiller, DistillerConfig,
...                  expert_iteration_step)
>>> teacher = Searcher(SearcherConfig(algorithm="puct", max_iterations=512))
>>> student = Distiller(DistillerConfig(model="linear", n_features=4096))
>>>
>>> for ep in range(100):
...     state = root_state()
...     while not is_terminal(state):
...         rep = teacher.search(state, actions=..., apply=...,
...                              terminal=..., reward=...,
...                              policy_prior=student.as_policy_prior(),
...                              value=student.as_value())
...         student.observe(state=state,
...                          action_distribution=rep.root_visits_by_action,
...                          value=rep.best_value)
...         state = apply(state, rep.best_action)
...     student.fit()  # eval-gated swap of the deployed model
>>>
>>> # the student is now usable as a *standalone* fast policy/value:
>>> p_at_s   = student.policy(s, [...actions...])  # dict action → prob
>>> v_at_s   = student.value(s)                    # scalar
```

### Model families shipped

All pure stdlib — no NumPy, no PyTorch, no SciPy.

  * **`"knn"`** — exact :math:`k`-nearest-neighbour over a feature-
    hashed Euclidean distance (Cover & Hart 1967): `O(N)` per query;
    consistent under the Cover-Hart bound for any
    state-feature space.
  * **`"linear"`** — per-action linear softmax policy + linear value
    head with feature hashing (Weinberger et al. 2009); trained by
    epoch-shuffled batch gradient descent with `L2` + per-coordinate
    clip for numerical safety; `O(d)` per query.
  * **`"locally_weighted"`** — locally-weighted regression (Atkeson,
    Moore & Schaal 1997): Gaussian-kernel-weighted average over the
    demonstration set.
  * **`"ucb_table"`** — exact tabular memoization; the right answer for
    small finite state spaces.
  * **`"ensemble"`** — log-linear opinion pool (Gneiting & Raftery 2007)
    over any subset of the above, with optionally externally-fit weights.

### Calibration

  * **Temperature scaling** (Guo et al. 2017) on the policy logits,
    fit by Brier-minimisation on a held-out slice.
  * **Isotonic value calibration** (Brunk et al. 1972) via PAV.

### Eval-gated deployment

Every `.fit()` produces a *candidate* model.  Deployment is gated by
a held-out cross-entropy + value-MSE drop ≥ `min_improvement`.  A
candidate that regresses the incumbent **cannot be deployed** — the
rollback story is enforced inside the primitive.  AlphaZero ladder
discipline, inside one process.

### Reservoir replay buffer

Vitter (1985) Algorithm R: bounded-memory uniform sample over the
entire demonstration stream.  No rolling windows, no batch hand-picking,
no oldest-data bias.

### Certificate chain

SHA-256 chain over the canonical sequence of `(epoch, mini-batch hash,
parameter delta hash, eval result)` events.  Two distillers in two
processes fed the same demonstrations under the same seed agree on the
certificate byte-for-byte.  Optional HMAC under `secret_key` for
authenticated chains.

### How it composes with the rest of the runtime

`Distiller` is the *amortising co-processor* for the rest of the
runtime:

  * **`Searcher` ↔ `Distiller`** — the AlphaGo Zero loop:
    Searcher generates training distributions (`root_visits_by_action`)
    and value targets (`best_value`); Distiller fits a student;
    Searcher uses the student as `policy_prior` and `value`; repeat.
    `expert_iteration_step()` ships this as a one-line helper.
  * **`Predictor` (CTW)** — Distiller's `value` head can stand in for
    Predictor's calibrated next-symbol estimate when the state space
    is fixed.  Wire one into the other.
  * **`Conformal`** wraps `student.value(s)` in a distribution-free
    coverage interval, turning "the student predicts +0.42" into
    "the student predicts +0.42, with 95% coverage of the true value
    in [+0.21, +0.58]".
  * **`Cartographer`** uses `DistillerReport.improvement_over_baseline`
    as the per-skill learning-progress signal: a task whose
    distillation step *just barely* improves the incumbent is in the
    zone of proximal development.
  * **`AttestationLedger`** consumes the certificate chain.  A
    regulator can replay every parameter update from the certificate
    alone.

### Investor framing

The AlphaZero ladder — *search produces targets, network distils
targets, network biases next search, repeat* — is the proven path
from "useful agent" to "Elo-saturated specialist" in every domain it
has been tried (Go, chess, shogi, Atari, protein folding, code).
`Distiller` is the runtime's **in-process, certificate-producing,
GPU-free realisation of that ladder**.  Composed with `Searcher`, it
turns "give the agent more training time" from a vague request into a
*measurable cost-per-decision curve* a coordination engine can dial.
A coordinator's investor dashboard is upstream of the field: the
`improvement_over_baseline` is a measured drop, not a claim.

### What it deliberately doesn't claim

  * `Distiller` is not a deep network.  Its model families are
    deliberately simple — kNN, hashed linear, LWR, exact table — so
    the runtime stays stdlib-only and reproducible byte-for-byte.
    For deep-network distillation, swap the `Distiller` model for the
    learner-track LoRA loop (see `ARCHITECTURE.md`).
  * It does not solve the *credit assignment* problem inside the
    teacher — that is the teacher's responsibility (Searcher's
    `reward`).  Distiller is a *function approximator* for whatever
    targets the teacher emits.
  * The certificate proves *reproducibility of the fit*, not
    *correctness of the teacher's targets*.  Garbage targets in,
    tamper-evident garbage student out.  Compose with `Verifier`
    on the upstream targets.

## Curator — automated curriculum *generation* as a runtime primitive

Every long-running runtime that learns eventually faces a problem
that neither `Cartographer` nor `Arbiter` can answer alone.
Cartographer selects *from a given pool* of tasks the one with the
highest expected learning progress.  Arbiter commits to the best of
*a finite arm set* with a fixed-confidence bound.  But the problem
upstream of both is: **where does the pool come from?**  AlphaZero
is not AlphaZero because it picks well from a fixed library of
board positions — it is AlphaZero because **self-play generates the
positions, at a difficulty just beyond current capability,
forever**.  The same is true of every self-improving system: a
curriculum is *built*, not given.

`Curator` is the runtime primitive that builds it.  Given a
parameterised task generator (a function from a difficulty vector
`θ ∈ Θ` to a concrete task) and a competence oracle (a function that
runs the agent against a task and returns 0/1 success), `Curator`
maintains an online estimate of the agent's competence across `Θ`
and proposes new tasks drawn from the **frontier of proximal
development** (Vygotsky 1934, Oudeyer & Kaplan 2007): tasks the
agent solves with probability that is neither too low (no signal)
nor too high (no progress), and whose learning progress is empirically
the highest.

### The four-primitive self-improvement loop

`Curator` is the missing fourth leg of the in-process AlphaGo-style
loop the runtime now ships end-to-end:

```
  Curator    → proposes new tasks at the ZPD frontier
   ↓
  Searcher   → solves them (PUCT / A* / alpha-beta / …)
   ↓
  Distiller  → compiles solutions into a fast student
   ↓                                  ↑
   └── student becomes Searcher's     │
       policy_prior + value for the   │
       next round ────────────────────┘
```

Composed with `Cartographer` (which picks *among* the Curator's
proposals by learning progress), the runtime has the complete
chain:

  * **"where do new tasks come from?"** → `Curator`
  * **"which of the proposed tasks should I attempt next?"** →
    `Cartographer`
  * **"given this task, what's the answer?"** → `Searcher`
  * **"compile the answer into a callable student"** → `Distiller`
  * **"use the student as the prior for the next Searcher call"**
    → loop closed

### Strategies shipped

  * **`"zpd"`** — Vygotsky's zone of proximal development.  Sample θ
    such that the posterior on competence is closest to
    `target_competence` (default 0.6).  Uses the Beta-Binomial
    conjugate (Jeffreys prior) and the Wilson score interval.
  * **`"learning_progress"`** — Oudeyer-Kaplan IAC (2007).  Track
    recent vs. older competence per cell; sample θ proportional to
    `|μ̂_recent − μ̂_prev|`.
  * **`"thompson_lp"`** — Thompson sampling over learning progress.
    Draw posterior LP per cell from Beta(s+½, n−s+½) and pick the
    argmax.  Russo et al. (2018) regret bounds apply.

### Calibration

`Curator.brier_score()` reports the Brier score of the predicted-vs-
realised success rate over the last `brier_window` proposals
(Gneiting & Raftery 2007).  A coordinator can SLO-gate on calibration.

### Certificate chain

SHA-256 chained over the canonical (proposal, observation) event
sequence.  Replay-verifiable byte-for-byte under the same config and
seed.  Optional HMAC under `secret_key` for authenticated chains.

### Investor framing

AlphaGo Zero's *self-play* is the single most important reason it
crossed superhuman play: it generated its own training data, at
the difficulty just beyond what it could do.  `Curator` is the
runtime's **generic, in-process renderer of that idea** —
domain-agnostic, certificate-producing, stdlib-only.  Together with
`Searcher` and `Distiller` it closes the in-process self-improvement
loop the rest of the architecture needs to *compound* over time;
a coordinator's investor dashboard can watch the
`improvement_over_baseline` curve fall and the frontier difficulty
rise as the agent learns.

### What it deliberately doesn't claim

  * `Curator` does not *invent* the difficulty parameterisation —
    the user supplies `param_lo` / `param_hi` / `n_buckets` and a
    `generator(theta) → task` callable.  Auto-discovering a useful
    parameterisation is an open research problem (UED, POET,
    Open-Ended Learning).
  * The Beta-Binomial competence posterior assumes binary outcomes;
    fine-grained quality scores are passed in via the `success`
    binarisation (e.g. `success = quality >= threshold`).
  * The cell discretisation is uniform.  Adaptive (KD-tree, BSP)
    discretisation is a natural extension and not yet shipped.

## Mentalist — Bayesian theory-of-mind as a runtime primitive

The multi-agent primitives in this runtime — `Negotiator`, `Coalition`,
`Mechanism`, `Persuader`, `Diplomat`, `Equilibrator` — all assume the
*other* parties have beliefs, desires, and intentions that can be
reasoned about.  None of them, by design, maintains the actual
probabilistic *model* of those mental states.  `Mentalist` is the
runtime primitive that does.

Theory of mind (Premack & Woodruff 1978) is the operation a debugger
performs when guessing the test author's intent, the operation a
negotiator performs when modelling the counterparty's reservation
price, and the operation a coordination engine must perform whenever
another agent's behaviour deviates from the prior.  Modern AI has
rediscovered it under three names — *inverse RL* (Ziebart-Maas-
Bagnell-Dey 2008), *Bayesian theory of mind* (Baker-Saxe-Tenenbaum
2009; Baker-Jara-Ettinger-Saxe-Tenenbaum 2017), and *opponent
modelling* (Foerster et al. 2018).

```python
>>> from agi import Mentalist, MentalistConfig
>>> m = Mentalist(MentalistConfig(rng_seed=1))
>>> m.register_agent("alice",
...                  states=("low", "mid", "high"),
...                  actions=("pass", "bid"),
...                  outcomes=("win", "lose"))
>>> for _ in range(20):
...     m.observe("alice", state="high", action="bid", reward=1.0, outcome="win")
...     m.observe("alice", state="low",  action="pass", reward=0.0, outcome="lose")
>>> m.predict("alice", state="high")
{'pass': 0.04, 'bid': 0.96}
>>> m.infer_desire("alice")
{'win': 7.31, 'lose': -7.31}
>>> bound = m.pac_bayes_bound("alice")
>>> bound.upper_bound
1.34  # Catoni-style PAC-Bayes upper bound on log-loss
```

What `Mentalist` ships:

  * **Bayesian belief tracking** — Dirichlet posteriors over latent
    state distributions; online conjugate updates.
  * **MaxEnt inverse RL** (Ziebart 2010 §3.4) — closed-form gradient
    descent on per-outcome utility weights, with full-action-space
    policy evaluation so single-action histories still produce signal.
  * **Bayesian rationality estimation** — Gamma posterior on the
    Boltzmann inverse-temperature ``β``; an agent that always picks
    the utility-maximising action drives ``β → ∞``, one that picks
    uniformly drives ``β → 0``.
  * **Capability posteriors** — Beta-Bernoulli on per-(state, action)
    success rates; ``confidence()`` returns Clopper-Pearson (1934)
    exact credible intervals.
  * **Four prediction methods** — `map`, `softmax` (Boltzmann-rational),
    `thompson` (sample-and-greedy with O(√T log T) regret), and
    `bayes_avg` (posterior-weighted mixture over Thompson samples).
  * **Simulation rollouts** — anytime forecasts of the agent's
    expected (state, action) trajectory under the posterior-mean policy.
  * **Nested theory of mind** — `nested_belief(observer, target, state)`
    returns the observer's posterior over the target's next action;
    the recursive ``ToM_k`` of Gmytrasiewicz-Doshi 2005.
  * **PAC-Bayes prediction certificate** (Catoni 2007) — closed-form
    upper bound on the policy's expected log-loss, with explicit
    KL-to-prior and sample-size dependence.
  * **Identifiability report** — equivalence classes of outcomes that
    are *empirically indistinguishable* on the observed data; the
    runtime knows when its IRL is underdetermined.
  * **SHA-256 chain certificate** — every registration, observation,
    inference and prediction is folded into a tamper-evident chain;
    replaying the same observations against the same RNG seed
    reproduces the certificate byte-for-byte.
  * **Pure stdlib** — no NumPy, no Torch, no SciPy.

Composition with the rest of the runtime:

  * Pass `mentalist.infer_desire(id)` as the `preferences` argument
    to `Negotiator` / `Mechanism` — the bargaining now runs against
    the runtime's *recovered* model of the counterparty.
  * `Persuader` consumes `mentalist.predict` distributions to score
    persuasive messages by expected belief-shift.
  * `Bandit` queries about a *known agent* can be routed through
    `Mentalist.predict(..., method=THOMPSON)` for a calibrated
    explanation along with the recommendation.
  * `Abductor` picks the model *family*; `Mentalist` picks the model
    *parameters* — natural staged inference.

Limitations honestly stated:

  * Default state/action/outcome spaces are *finite and discrete*.
    Continuous spaces require discretisation via `Sketcher` /
    `Topologist` before being fed in.
  * The agent is assumed to be ε-Boltzmann-rational.  An adversary that
    deliberately randomises against the recovered utility is identified
    as ``β → 0`` but not exploited beyond that.
  * Nested ToM beyond depth 2 is currently expensive (``O(|A|^k)``) and
    requires composition with `Sampler` for posterior marginalisation.

## Reconciler — Aumann agreement as a runtime primitive

Every primitive in this runtime that emits a posterior — `Bandit` over arms,
`BayesOpt` over the optimum location, `Imaginator` over future returns,
`Forecaster` over the next observation, `Mentalist` over a counterparty's
utility — eventually produces output that the coordination engine has to
*combine* with the output of some other primitive that estimates a related
quantity. Without a principled aggregation step the coordinator picks one
(loses the information in the others), or averages naively (loses the
calibration of the most-certain one), or hand-tunes a weighted vote (loses
any guarantee of optimality).

`Reconciler` is the runtime's *bounded, anytime, certified, stdlib*
version of that aggregation. It implements four pooling rules that
together cover the literature on consensus belief aggregation.

```python
from agi.reconciler import Reconciler, ReconcilerConfig

rec = Reconciler(ReconcilerConfig(method="aumann"))
rec.register_topic("arm_a_wins", outcomes=("yes", "no"))
rec.contribute("arm_a_wins", source="bandit",   belief={"yes": 0.70, "no": 0.30})
rec.contribute("arm_a_wins", source="bayesopt", belief={"yes": 0.60, "no": 0.40})
rec.contribute("arm_a_wins", source="psrl",     belief={"yes": 0.65, "no": 0.35})

report = rec.consensus("arm_a_wins")
print(report.consensus)            # {"yes": 0.65, "no": 0.35}
print(report.outlier)              # ("bandit", 0.0056)
print(report.converged, report.rounds)  # True, 17
```

### Aggregation methods shipped

  * **`"linear"`** — Stone 1961 linear opinion pool
    ``q(·) = Σ_i w_i p_i(·)``. Genest-McConway 1990 show this is the
    *only* externally-Bayesian pool when the weights do not depend on
    the experts' beliefs.
  * **`"logarithmic"`** — Bordley 1982 log-linear pool
    ``q(·) ∝ Π_i p_i(·)^{w_i}``. The maximum-entropy combination
    subject to matching each expert's KL-projection of the consensus.
  * **`"kl_barycenter"`** — Bregman 1967 right-KL barycenter that
    minimises ``Σ_i w_i · KL(q ‖ p_i)``; the closed-form solution
    coincides with the logarithmic pool.
  * **`"aumann"`** — Geanakoplos-Polemarchakis 1982 iterative
    Bayesian agreement. Each round every expert updates by averaging
    with the current pool; on finite state spaces the iteration
    provably reaches consensus in finitely many rounds. The
    round-cap returns the closest-to-consensus KL-barycenter with
    ``converged=False`` when fired.

### What it ships

  * `consensus(topic, method, weights)` — the consensus pmf plus
    `per_source_kl`, `outlier`, `confidence_interval` (HRMS anytime-
    valid CI per outcome), `effective_n_sources` (inverse Herfindahl-
    Hirschman), `converged`/`rounds`, and `fingerprint_hash`.
  * `calibration(topic, source)` — per-source Massey 1951 KS test on
    PIT of realised outcomes plus closed-form average log-loss (for
    binary topics where PIT is uninformative).
  * `identifiability_report(topic)` — flags topics where every source
    assigns zero mass to some outcome and reports the effective number
    of independent contributors.
  * Tamper-evident SHA-256 fingerprint chain with optional HMAC-SHA-256
    over every register / contribute / consensus / calibration event so
    `AttestationLedger.verify` replays the consensus byte-for-byte.
  * `export_state()` / `import_state(snap)` round-trip byte-identical
    chain head.
  * Pure stdlib — list-of-lists arithmetic, log-sum-exp softmax, hashlib
    SHA-256. No NumPy, no Torch.

### How it composes with the rest of the runtime

  * **`Bandit` / `BayesOpt` / `Imaginator` / `Forecaster` /
    `Predictor`** — each contributes its posterior to a Reconciler
    topic and the coordinator sees the calibrated consensus instead of
    any single primitive's belief.
  * **`Auditor`** — Reconciler's per-source outlier KL is a candidate
    test statistic; Auditor BH-controls FDR across simultaneous topics.
  * **`DriftSentinel`** — running consensus stability is a
    martingale-difference under common knowledge; CUSUM flags
    contributor drift.
  * **`Aligner`** — preferences over (topic, consensus) pairs become
    training data for the system's reward model.
  * **`Mentalist`** — supplies the rationality posterior the
    coordinator weights each Mentalist-modelled counterparty's
    contribution by.
  * **`Conformal`** — wraps the consensus pmf with a finite-sample
    prediction set.
  * **`Coordinator`** — every Goal whose execution depends on more
    than one primitive's posterior routes through Reconciler — the
    coordination engine sees one calibrated belief plus the outlier
    name plus the anytime-valid CI plus the audit-chain head, instead
    of K conflicting posteriors.

### Investor framing

Reconciler is the **consensus-belief kernel**. Every primitive
processes one signal; Reconciler processes the ensemble.  Pair every
Goal whose execution depends on more than one source through
Reconciler and the coordination engine gets *one* calibrated belief
plus *one* outlier name plus *one* receipt the compliance officer can
sign, instead of K conflicting posteriors. This is the line between
*"we run K AI primitives"* and *"we run K AI primitives and reconcile
their beliefs with provable Bayesian agreement before action."*

### What it deliberately doesn't claim

  * The Aumann iteration shipped here is the *cognitive-economy
    approximation* (Hanson 2003) — each expert broadcasts the
    posterior pmf rather than the partition cell containing the truth.
    The convergence + agreement properties Aumann 1976 proved on the
    full formulation hold in finite-step approximation here, but the
    common-prior assumption is not enforced.
  * Calibration via the PIT KS test is uninformative for binary
    outcomes; for binary topics use the log-loss field instead.
  * The primitive aggregates pmfs over a finite outcome set. For
    continuous posteriors (densities) discretise first via
    `Conformal` or use the `Hedger` universal-experts primitive for
    online sequence prediction.

## Imaginator — learned-world-model rollouts as a runtime primitive

Every primitive in this runtime that *plans* eventually faces the same
problem: it has to reason about *what happens next* before it commits real
cost. `Searcher` runs anytime certified tree search over a *caller-supplied*
successor enumerator. `ActiveInferencer` reduces expected free energy over
a *caller-supplied* generative model. `Planner` compiles a *caller-supplied*
PDDL operator schema. In every case the user hands over a dynamics function
and trusts it.

`Imaginator` is the primitive that **learns** that dynamics function from
observed transitions, **bounds** the error of every imagined trajectory, and
emits a **tamper-evident receipt** for every imagined rollout. It is the
**model-based-RL inner loop** as a runtime primitive a coordination engine
can register, drive, and audit.

```python
from agi.imaginator import Imaginator, ImaginatorConfig

im = Imaginator(ImaginatorConfig(family="categorical", discount=0.9))
im.register_env("supply", states=("ok","stockout"), actions=("ship","wait"))

for s, a, s_next, r in observed_transitions():
    im.observe("supply", s, a, s_next, r)

plan = im.value_iteration("supply", horizon=30, discount=0.9)
roll = im.imagine("supply", state="ok",
                  policy=lambda s: plan.policy[s],
                  horizon=20, samples=128, method="thompson")
print(roll.expected_return, roll.value_lcb, roll.value_ucb)  # Maurer-Pontil 95% CI
print(roll.hrms_lcb,        roll.hrms_ucb)                   # anytime-valid CI

pac = im.pac_value_bound("supply", policy=plan.policy, delta=0.05, horizon=20)
print(pac.epsilon, pac.min_observations)                     # Kearns-Singh PAC
```

### Dynamics families shipped

  * **`"categorical"`** — discrete-state, discrete-action MDP with
    Dirichlet-multinomial conjugate posterior on per-(state, action)
    successor distributions and Normal-Gamma conjugate posterior on
    per-(state, action) rewards. Closed-form Bayesian updates; the
    posterior predictive over next state is the Dirichlet-Multinomial
    mean ``α[s'] / Σ α[s']``. This is the **Bayesian R-MAX** family
    (Strehl-Littman-Wiewiora 2009; Auer-Jaksch-Ortner 2010 UCRL2), with
    optimism-under-uncertainty delivered via Thompson-sampled transition
    matrices (Strens 2000 / Osband-Russo-Van Roy 2013 *PSRL*) with
    Bayesian-regret bound ``O(τ √(SAT log T))`` where τ is the diameter.

  * **`"linear_gaussian"`** — continuous-state linear dynamics
    ``s_{t+1} = A s_t + B a_t + ε_t`` with matrix-normal-inverse-Wishart
    conjugate prior on ``[A | B]`` and ``Σ``. Closed-form online
    sufficient-statistic updates; closed-form moment-matching predictive
    ``N(μ, σ²)`` at every horizon. This is the **PILCO** family
    (Deisenroth-Rasmussen 2011) with the closed-form moment propagation
    PILCO required a Gaussian process for, here delivered by a Bayesian
    linear model whose epistemic uncertainty closes in expectation as
    ``n → ∞``.

### Mathematical and algorithmic roots

  * **Sutton 1990 *Dyna*** — real and imagined transitions update the
    same value function; one observation buys both a planning step and a
    learning step.

  * **Kearns-Singh 2002** — the *simulation lemma*
    ``|V^π_M̂ − V^π_M| ≤ (γ/(1−γ)²) ε`` whenever ``M̂`` is ``ε``-accurate
    in transition + reward. Backbone of `pac_value_bound`.

  * **Strehl-Littman-Wiewiora 2009 PAC-MDP** — sample complexity
    ``O((SA / ε²(1−γ)⁴) · log(SAδ⁻¹))`` exactly the formula
    `required_samples_for_pac` returns.

  * **Janner-Fu-Zhang-Levine 2019 *When to Trust Your Model*** — the
    short-horizon-rollout argument: imagined trajectories are accurate
    at small ``h``, biased at large ``h``; `trajectory_quantiles` expose
    the growing predictive variance that justifies the horizon cap.

  * **Hafner-Lillicrap-Ba-Norouzi 2020 *Dream to Control*** — DreamerV3
    treats imagined trajectories as the policy-optimisation surface;
    Imaginator delivers the trajectories with calibrated bounds the
    coordinator reads *before* committing.

  * **Maurer-Pontil 2009** empirical-Bernstein bound on Monte Carlo
    return; **Howard-Ramdas-McAuliffe-Sekhon 2021** anytime-valid
    confidence sequences; **Massey 1951** Kolmogorov-Smirnov test for
    PIT calibration (asymptotic distribution via the Stephens 1970
    correction).

### What it ships

  * `imagine(env, state, policy, horizon, samples, method)` — Monte
    Carlo trajectories from the posterior predictive; bundles
    `expected_return`, `return_std`, Maurer-Pontil `value_lcb/ucb`,
    HRMS anytime-valid `hrms_lcb/ucb`, return quantiles, full
    trajectories, per-step state quantiles, and a chain `fingerprint_hash`.
  * `value_iteration(env, horizon, discount, tol)` — closed-form DP
    planning on the posterior-mean transition / reward.
  * `thompson_policy(env, horizon, discount)` — PSRL: draw one
    transition matrix from the Dirichlet posterior, return the
    value-iteration policy for that draw.
  * `pac_value_bound(env, policy, delta, horizon)` — Kearns-Singh
    simulation-lemma PAC bound on the policy-value estimation error.
  * `required_samples_for_pac(env, epsilon, delta)` — invert the PAC
    bound: minimum ``min_n`` per reachable (s, a) to achieve a target ε.
  * `bayes_average_value(env, policy, horizon, samples, n_models)` —
    Bayesian Model Averaging (Madigan-Raftery 1994) value estimate.
  * `moment_rollout(env, state, policy, horizon, noise)` — closed-form
    linear-Gaussian moment propagation (PILCO).
  * `identifiability_report(env, min_observations)` — flags
    under-observed (s, a) pairs and reports effective Dirichlet
    concentration per pair (Cao-Cohen-Szepesvári 2021).
  * `pit_calibration(env)` — one-sample KS test on the PIT of one-step
    rewards under the Student-t reward predictive.
  * Tamper-evident SHA-256 fingerprint chain (genesis seed
    `"agi.imaginator.v1\x00" + secret_key`) with optional HMAC-SHA-256
    over every register / observe / imagine / value / certify event so
    `AttestationLedger.verify` replays the imagined trajectory
    byte-for-byte from the same observation stream + RNG seed.
  * `export_state()` / `import_state(snap)` — full posterior snapshot
    round-trips byte-identical chain head.
  * Pure stdlib — list-of-lists matrices, Cholesky-via-Lentz solver,
    Marsaglia-Tsang gamma draws, hashlib SHA-256. No NumPy, no SciPy,
    no PyTorch.

### How it composes with the rest of the runtime

  * **`Searcher`** — Imaginator is *the* successor enumerator Searcher
    accepts. Searcher's tree search runs over imagined transitions; the
    cost-of-evaluation Searcher trades off is Imaginator's per-sample
    rollout time.
  * **`ActiveInferencer`** — Imaginator supplies the generative model
    (state-transition + observation likelihood) that the Active Inference
    primitive's expected-free-energy minimisation requires.
  * **`Quantilizer`** — Imaginator's `return_quantiles` *is* the
    distribution Quantilizer thresholds on. "Deploy the policy whose
    imagined return is in the top ``q`` quantile of the posterior over
    dynamics."
  * **`Distiller`** — distil the value-iteration policy returned by
    `Imaginator.value_iteration` into an amortised neural / linear policy.
  * **`Planner`** — Imaginator's posterior-mean transition matrix is a
    PDDL-compilable operator schema; Planner reads it and solves SAT with
    the deterministic mode of Imaginator's MAP transitions.
  * **`DriftSentinel`** — per-step log-loss of one-step predictions is a
    martingale-difference under correct dynamics; DriftSentinel runs a
    CUSUM and flags world drift in real time.
  * **`Bandit` / `BayesOpt`** — Thompson-sampled value from Imaginator is
    a cheap proxy oracle for hyperparameter / arm selection.
  * **`Curator`** — Imaginator's identifiability report identifies (state,
    action) pairs that are still under-observed; Curator targets them in
    its next curriculum batch.
  * **`AttestationLedger`** — every register / observe / imagine / value /
    certify event chain-hashes into the ledger; a compliance officer
    replays byte-for-byte from the observation stream + RNG seed.
  * **`Coordinator`** — every Goal whose execution requires reasoning
    over future world states routes through Imaginator. The coordination
    engine no longer hand-writes the dynamics function; it observes a few
    real transitions, registers them with Imaginator, and queries imagined
    value with calibrated uncertainty bounds the compliance officer can
    sign before action.

### Investor framing

Imaginator is the runtime's **imagination kernel**. Every prior primitive
processes the present. Imaginator processes the future with calibrated
uncertainty and a receipt the compliance officer can sign before money
moves. This is the line between *"we run AI"* and *"we run AI that reasons
about consequences before committing them."* Pair with `Quantilizer` for
safety-bounded deployment, `Searcher` for tree-search over imagined
futures, and `AttestationLedger` for cryptographic replay — the
**model-based-RL inner loop**, delivered as a runtime primitive a
coordination engine can drive.

### What it deliberately doesn't claim

  * Not a frontier dynamics learner — no neural networks, no transformer
    world model. The two model families shipped (Dirichlet-multinomial
    categorical and matrix-normal-inverse-Wishart linear-Gaussian) are
    the *conjugate* ones that admit closed-form Bayesian updates and
    provable PAC bounds.
  * Not online closed-loop control — Imaginator imagines on demand, but
    the coordinator decides when to act. Composition with a real-time
    controller is the caller's responsibility.
  * Not partial-observability — the categorical family assumes the
    observed state *is* the latent state. Composition with `Filterer` for
    the POMDP case is the recommended pattern; the linear-Gaussian family
    natively supports observation noise but the latent-state dimensionality
    must match the action space.

## Flower — Generative Flow Networks as a runtime primitive

Every prior primitive in this runtime that *selects* returns *one* answer
— `BayesOpt`'s argmax, `Searcher`'s best leaf, `Solver`'s witness,
`Planner`'s plan. Every prior primitive that *samples* — `Sampler`, the
posterior-predictive in `Imaginator` — samples from a fixed target that
is **not** the reward distribution. Neither is what a real product needs.
A drug-discovery pipeline ships a *panel*. A program-synthesis loop ships
a *panel*. A negotiation engine ships a *panel*.

`Flower` is the runtime's **diversification kernel**: the bounded,
anytime, certified, stdlib-only implementation of Generative Flow
Networks (Bengio-Lahlou-Deleu-Hu-Tiwari-Bengio 2021; Malkin-Jain-Everett-
Sun-Bengio 2022 *Trajectory Balance*; Madan et al. 2023 *Sub-trajectory
Balance*). A GFlowNet learns a forward policy ``P_F(s' | s)`` on a DAG
such that the marginal probability of terminating at object ``x`` is
**proportional to the reward**:

```
P_T(x)  =  R(x) / Z       where     Z = Σ_x R(x).
```

This is the fundamentally different generative regime: not *"argmax R"*
(RL), not *"sample from a fixed π"* (MCMC), but **sample objects with
probability proportional to reward** — the panel the coordinator
ships, with a calibrated mode-coverage receipt.

```python
from agi.flower import Flower, FlowerConfig

flow = Flower(FlowerConfig(loss="trajectory_balance"))
flow.register_env("molecules", initial="",
                  successors=lambda s: [(c, s + c) for c in "01"] if len(s) < 4 else [],
                  terminal=lambda s: len(s) == 4,
                  reward=reward_fn)

for _ in range(300):
    flow.train_step("molecules", n_trajectories=16, epsilon=0.1)

batch = flow.sample("molecules", n=200)
print(batch.unique_terminals, batch.mean_reward, batch.mean_reward_lcb)

cov = flow.mode_coverage("molecules", n_samples=500, top_k=3)
print(cov.tv_to_target, cov.modes_found, cov.mode_coverage_lcb)
# AttestationLedger.verify(batch.fingerprint)
```

### Loss families shipped

  * **`"flow_matching"`** — Bengio-Bengio 2021 detailed flow balance at
    every non-initial non-terminal state: `Σ_in F  =  Σ_out F  +  R·1[terminal]`.
    Closed-form gradient on log-flow logits.
  * **`"detailed_balance"`** — Bengio et al. 2021 *GFlowNet Foundations*
    edge constraint `F(s) P_F(s'|s) = F(s') P_B(s|s')`. Single-edge
    residual → low gradient variance.
  * **`"trajectory_balance"`** — Malkin-Jain-Everett-Sun-Bengio 2022.
    `logZ + Σ log P_F = log R(x) + Σ log P_B`. Lowest-variance
    GFlowNet loss on small-to-mid DAGs.
  * **`"subtrajectory_balance"`** — Madan-Rector-Brooks-Korablyov-
    Bengio-Liu-Chen-Hu-Bengio 2023. Geometric-weighted sub-trajectory
    residuals; combines FM's local credit-assignment with TB's global
    signal.

### What the primitive ships

  * **Bounded mode-coverage diagnostics.** Closed-form total-variation
    distance between the empirical terminal distribution and the
    reward-proportional target; Hoeffding 1963 UCB on TV; Howard-Ramdas-
    McAuliffe-Sekhon 2021 anytime-valid LCB on the *coverage probability*
    over the top-K reward modes.
  * **Maurer-Pontil + HRMS bounds** on `E[R]` under the learned sampler,
    written into every `SampleBatch`.
  * **Top-K Pareto extraction** — distinct terminals with highest
    reward, paired with observed visit count; the panel the coordinator
    ships.
  * **Identifiability report** — under-sampled edges + unreachable
    rewarded modes; the next-curriculum-batch hand-off to `Curator`.
  * **PIT calibration** — KS test of the reward PIT vs Uniform(0, 1);
    the baseline statistic the `DriftSentinel` CUSUM consumes.
  * **Tamper-evident attestation.** Every register / observe /
    train_step / sample / certify event chain-hashes into an HMAC-secured
    fingerprint; a compliance officer replays the candidate batch
    byte-for-byte from the observation stream + RNG seed.

### Composes with the rest of the runtime

  * `Quantilizer` — Flower's `(terminals, rewards)` *is* the empirical
    distribution the Quantilizer thresholds on. Ship the K candidates
    whose reward is in the top-`q` quantile of the GFlowNet's posterior.
  * `Imaginator` — Imaginator's posterior over rewards is a drop-in
    `reward_fn` for the Flower when the real reward is delayed or noisy.
  * `Reconciler` — Flower's terminal distribution is one expert in the
    pool; Reconciler aggregates GFlowNet + BayesOpt + Searcher into a
    common-prior consensus.
  * `Curator` — Flower's identifiability report names the
    under-sampled (state, action) edges Curator targets next.
  * `Aligner` — Flower's K-tuple of high-reward terminals is one half
    of the preference pair the Aligner trains on; the other half is
    the user's chosen winner. *Generate-then-rank-then-align*, closed
    loop.
  * `Distiller` — distil the learned forward policy `P_F` into an
    amortised classifier that runs at inference time without rollouts.
  * `DriftSentinel` — PIT-of-rewards is the live uniformity statistic
    CUSUM trips on when sampling drifts.
  * `AttestationLedger` — every event chain-hashes; cryptographic
    replay from observation stream + seed.
  * `Coordinator` — every Goal that benefits from a *panel* (drug
    design, code search, negotiation playbooks, hyperparameter sweeps)
    routes through Flower.

### What it deliberately doesn't claim

  * Not a frontier neural GFlowNet — no transformers, no graph neural
    nets. The tabular per-edge logit + per-env `logZ` is the
    **convergent, identifiable** core that admits closed-form gradients
    and provable mode-coverage bounds.
  * Not a continuous-action generator — the discrete-DAG core covers
    every combinatorial-generation use case (molecules-as-strings,
    programs-as-tokens, layouts-as-grids); continuous extensions
    (Lahlou-Deleu-Hu-Bengio 2023) are reserved for a follow-up
    primitive.
  * Not a guaranteed exploration mechanism — the GFlowNet samples
    proportional to reward, not to information gain. Pair with
    `Curator` for active-learning-style exploration of the
    identifiability gap.

## HTTP / SSE surface

`python -m agi.server` exposes the Runtime over HTTP for out-of-process
coordinators:

| Method | Path                              | Purpose                                       |
|--------|-----------------------------------|-----------------------------------------------|
| GET    | `/healthz`                        | Liveness check                                |
| GET    | `/capabilities`                   | What the runtime offers right now             |
| GET    | `/metrics`                        | Counters + totals for SLO/observability       |
| GET    | `/sessions`                       | List sessions                                 |
| POST   | `/sessions`                       | Create a session (body = SessionConfig + `namespace`) |
| GET    | `/sessions/{id}`                  | Inspect state                                 |
| POST   | `/sessions/{id}/chat`             | One turn; returns `{final_text, session}`     |
| POST   | `/sessions/{id}/cancel`           | Cancel between turns                          |
| POST   | `/sessions/{id}/reset`            | Clear conversation, keep session              |
| POST   | `/sessions/{id}/checkpoint`       | Persist session to the session store          |
| POST   | `/sessions/restore`               | `{session_id}` → reload from store            |
| DELETE | `/sessions/{id}`                  | End                                           |
| GET    | `/events`                         | SSE stream of all events                      |
| GET    | `/events?session_id=…&kind=…`     | Filtered SSE                                  |
| GET    | `/events/history`                 | Replay past events                            |
| GET    | `/tasks`                          | List queued/running/done tasks                |
| POST   | `/tasks`                          | Submit a task (prompt + budget + deadline)    |
| GET    | `/tasks/{id}`                     | Inspect a task                                |
| POST   | `/tasks/drain`                    | Run queued tasks (synchronous; `max_ticks`)   |
| POST   | `/skills`                         | Save a skill                                  |
| POST   | `/tools`                          | Synthesize a sandboxed tool                   |

Optional bearer-token auth via `AGI_AUTH_TOKEN` env var or `--auth-token`.

## stdio JSON-RPC surface

For coordinators that prefer spawning the runtime as a subprocess
(MCP-style), `CoordinationProtocol` exposes the same surface over
newline-delimited JSON-RPC 2.0 on stdin/stdout:

```python
from agi.runtime import Runtime
from agi.protocol import CoordinationProtocol
CoordinationProtocol(Runtime()).serve_stdio()
```

Methods: `ping`, `version`, `runtime.capabilities`, `runtime.metrics`,
`session.create/chat/cancel/end/get/list`, `tasks.submit/get/drain`,
`plans.submit/run/get/list/cancel`, `skills.save`, `tools.synthesize`,
`events.subscribe/unsubscribe/history`.
Notifications: `ready` (banner on connect), `event` (one per bus event
while subscribed).

### Parallel DAG plans — `ParallelScheduler`

`agi.scheduler.ParallelScheduler` is the coordination primitive when the
work has *shape*. Hand it a `Plan` (steps + dependencies) and it
dispatches independent steps in parallel up to `max_concurrent_steps`,
retries transient failures with exponential backoff, enforces per-plan
budget and deadline, and streams `plan.step.*` / `plan.completed` events.
The same surface is exposed over JSON-RPC as `plans.submit` / `plans.run`
for out-of-process coordinators.

```python
from agi.scheduler import ParallelScheduler, SchedulerConfig, RetryPolicy
from agi.coordinator import Plan, PlanStep

sched = ParallelScheduler(runtime, config=SchedulerConfig(
    max_concurrent_steps=4,
    retry_policy=RetryPolicy(max_attempts=3, backoff_seconds=0.5),
))
result = sched.run(Plan(steps=[
    PlanStep(id="a", prompt="research X"),
    PlanStep(id="b", prompt="research Y"),
    PlanStep(id="c", prompt="synthesize", depends_on=["a", "b"]),
]), budget_usd=5.0)
```

See `examples/parallel_plan_demo.py` for a fan-out / fan-in walkthrough.

## Event kinds (the coordination contract)

The bus emits typed events. Coordinators pattern-match on `kind`:

- `session.created` / `session.ended`
- `chat.started` / `chat.completed`
- `usage.updated` — running token + cost totals
- `skill.loaded` — a skill was injected into a prompt
- `subagent.started` / `subagent.completed` — delegation
- `tool.synthesized` — agent extended itself
- `critic.scored` — gate fired with a confidence score
- `autoloop.iteration_started` / `autoloop.iteration_completed`
- `autoloop.completed` / `autoloop.failed` / `autoloop.budget_exhausted`
- `autoloop.skill_promoted` — a winning trajectory graduated into the library
- `fork.race_started` / `fork.race_completed`
- `pool.node_added` / `pool.node_removed` / `pool.node_unhealthy`
- `pool.dispatch_started` / `pool.dispatch_completed` / `pool.dispatch_failed`
- `plan.scheduled` / `plan.step.ready` / `plan.step.running`
- `plan.step.completed` / `plan.step.failed` / `plan.step.retry`
- `plan.completed` / `plan.failed` / `plan.budget_exhausted` / `plan.cancelled`
- `drift.started` / `drift.observation` / `drift.detected` / `drift.reset` / `drift.cleared`
- `error` — including `CostCeilingExceeded` when budget runs out

Subagent token usage rolls up into the parent session for honest accounting.

## What it can do

- Read/write files, run shell commands
- Search the live web (`web_search_20260209`) and fetch URLs (`web_fetch_20260209`)
- Remember things across sessions (`~/.agi/memory.jsonl`)
- Load **relevant skills automatically** before answering (procedural memory)
- **Synthesize new tools mid-session** in a sandboxed subprocess (AST scan +
  banned imports + smoke test + per-call timeout)
- **Delegate subtasks to specialist subagents** with cost roll-up
- Plan with adaptive thinking on hard tasks (`effort: high`)
- Stream output and emit a typed event for every state transition
- Enforce per-session **cost ceilings** at the runtime layer
- Critic gate: scores final responses and annotates low-confidence ones
- **Learn which (role, model, effort) wins** on which prompts via a
  contextual Thompson-sampling bandit (`PolicyRouter`)
- **Federate over many runtimes** with skill- and load-aware dispatch
  (`RuntimePool`) — one coordinator, N runtime nodes
- **Speak JSON-RPC over stdio** so any external coordinator drives the
  runtime as a subprocess (`CoordinationProtocol`)
- **Mine its own regression suite** from successful traces and refuse
  promotions that regress it (`SelfEvalBank`)
- **Auto-decompose Goals** into multi-step DAGs via heuristic patterns
  or an LLM planner (`agi.goalc`)

## What it can't do (yet)

Everything that makes AGI AGI: open-ended self-improvement on the base
model, robust out-of-distribution transfer, grounded world models, durable
goal pursuit across weeks of autonomy. The Opus reasoning core is frozen.
The `learner/` track is the path toward durable improvement via LoRA on a
small open base. See `ARCHITECTURE.md` for the design and what's open
research vs. tractable engineering.

## Testing

```sh
python -m unittest discover tests
# 230+ tests across events, skills, toolsynth, runtime, server, persistence,
# tasks, coordinator, autoloop, fork, capabilities, skillmine, agent, learner,
# policy, pool, protocol, selfeval, goalc
```

All tests run without an API key; they exercise the runtime, sandbox, and
HTTP server via a `FakeAgent` factory so CI doesn't burn budget.
