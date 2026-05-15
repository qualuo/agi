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
  scheduler.py      # ParallelScheduler — DAG-aware parallel plan execution
  skillmine.py      # mine reusable skills from successful trace patterns
  skills.py         # markdown skill library with retrieval (procedural memory)
  reflection.py     # per-task lessons-to-memory loop (medium-timescale learning)
  world_model.py    # observed-entity tracker (file/url/command + outcomes)
  toolsynth.py      # sandboxed Python tool synthesis (subprocess isolated)
  tasks.py          # Task / TaskQueue / TaskRunner — scheduled work
  persistence.py    # checkpoint sessions to disk and rehydrate
  memory.py         # persistent JSONL memory store + namespacing (multi-tenant)
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
