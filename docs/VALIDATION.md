# Validation protocol
Primary split: OhioT1DM patient-level LOSO.
External validation: DiaTrend and HUPA-UCM.
Hyperparameters and thresholds are selected only inside training patients.
Forecast: MAE, RMSE, MARD, Clarke A%.
Risk: AUROC, PR-AUC, recall, precision, Brier, lead time, FP/day.
Report patient-level means and dispersion. Clarke Error Grid is for glucose concentration forecasts only.
No target metric is a result until measured.
