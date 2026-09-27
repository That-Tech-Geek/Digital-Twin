# Architecture
Ingestion -> quality gates -> canonical 5-minute timeline -> causal features -> patient-level split -> baselines/forecasters -> personalized twin -> probabilistic simulator -> counterfactuals -> evaluation/dashboard.
Layer A initializes constrained physiological proxies from static context. They are model parameters, not clinical measurements.
Layer B estimates dynamic latent state with GRU; TCN is a comparison model.
Layer C produces probabilistic future glucose trajectories and threshold-crossing probability.
Counterfactuals are simulations only and must never be rendered as treatment recommendations.
Algorithmic validation is on insulin-treated/open longitudinal cohorts; Indian T2D deployment requires future appropriate validation.
