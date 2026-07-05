# Assumptions Log
This file records every relevant assumption made during the construction of the thesis dataset and modeling pipeline.

## Project Assumptions

1. The project is developed in Python.
2. The project is developed and run from VS Code.
3. The initial dataset will use free/public data sources.
4. The data ingestion layer will later be replaceable with Refinitiv/LSEG data.
5. The first version of the analysis uses daily data.
6. The project should remain modular and simple.
7. Raw data, cleaned data, model outputs, figures, and tables are stored separately.
8. The project is installed locally in editable mode using `pip install -e .` so that modules inside `src/` can be imported from scripts.


## Data Assumptions

1. The initial free-data universe uses 9 listed euro-area banks.
2. The initial bank universe includes Italy, France, Germany, Spain, and the Netherlands.
3. UK and Swiss banks are excluded from the first version to avoid currency complications.
4. The first market benchmark is the Euro Stoxx 50.
5. The first banking-sector benchmark is a free ETF proxy rather than an official STOXX banking index.
6. The first stress variable is the VIX from FRED.
7. Refinitiv/LSEG data may later replace or enrich the free-data inputs.
8. The first raw data downloader uses `yfinance` for bank and market price data.
9. The first raw data downloader uses FRED through `pandas-datareader` for the VIX series.
10. Raw downloaded data are saved before any return or loss construction.
11. Yahoo tickers are treated as provisional and may later be replaced by Refinitiv identifiers.
12. Missing or failed ticker downloads will be reported but not silently ignored.
13. Yahoo price downloads are saved with two header rows: price field and ticker.
14. For the first processed price panel, adjusted close is preferred over close.
15. If adjusted close is missing for a ticker, close may be used as a fallback.
16. Raw Yahoo files are not manually edited; cleaning is performed through code.
17. Processed price panels are saved in wide format with dates as the index and tickers as columns.
18. Stress variables are aligned by date but not forward-filled at this stage.
19. For the first returns dataset, returns will be computed from cleaned adjusted price panels rather than raw Yahoo files.
20. Log returns are computed as first differences of log prices.
21. Losses are defined as negative log returns.
22. Dates with missing prices will initially be dropped after return construction.
23. Stress variables will be aligned to the return panel by date.


## Data Notes from First Build

1. The first bank return panel contains 5,389 daily observations and 9 euro-area banks.
2. The bank return sample runs from 2005-01-04 to 2026-07-03.
3. Building the balanced bank return panel required dropping 124 rows with missing bank returns.
4. The market return panel was aligned to the bank return calendar.
5. After alignment, the market return panel contains 760 missing observations for EXX1.DE and 708 missing observations for ^STOXX50E.
6. After alignment, the VIX stress-variable panel contains 113 missing observations.
7. The first equal-weighted system loss series was computed as the cross-sectional mean of bank losses.
8. No forward-filling has been applied yet to prices, returns, market variables, or VIX.
10. After excluding UCG.MI, the first modeling universe contains 8 euro-area banks: BBVA.MC, BNP.PA, CBK.DE, DBK.DE, GLE.PA, INGA.AS, ISP.MI, and SAN.MC.
11. The updated rolling-volatility and return plots no longer show UCG.MI, confirming that the processed modeling panel uses the 8-bank universe.
12. The remaining return series still show large crisis-period movements, which are treated as genuine extreme observations unless later audits suggest otherwise.

## Data Quality Assumptions

1. Extreme daily returns are audited before fitting GARCH or EVT models.
2. Extreme observations are not removed automatically.
3. Potentially suspicious observations are flagged for inspection rather than deleted silently.
4. The first audit reports the largest positive and negative daily returns for each bank.

## Data Quality Notes

1. The extreme-return audit suggests that most ING extreme returns occur during recognized crisis periods.
2. UCG.MI displays repeated early-sample positive and negative jumps of approximately 17%, often on adjacent trading days.
3. UCG.MI early-sample observations require further inspection before fitting GARCH or EVT models.
4. No observations are removed at this stage.
5. The price-window audit confirms repeated early-sample UCG.MI adjusted-price jumps followed by immediate reversals.
6. These UCG.MI jumps are treated as likely data-adjustment artifacts in the free Yahoo dataset.
7. UCG.MI is excluded from the first free-data modeling sample.
8. UCG.MI may be reintroduced later if Refinitiv/LSEG total return data are available and pass the same audit checks.
9. No other bank is excluded at this stage.

## Modeling Assumptions

1. Returns are computed as daily log returns.
2. Losses are defined as negative log returns.
3. The first system loss measure uses equal weights across banks.
4. The first stress-regime definition uses the 80th percentile of VIX.
5. The initial EVT threshold candidate set is 90%, 95%, and 97.5%.
6. The main EVT threshold is initially set to 95%.
7. The initial volatility model is GJR-GARCH(1,1) with Student-t innovations.
8. The initial tail probability levels for VaR and ES are 99% and 99.5%.
9. Each bank's marginal return dynamics are filtered separately using a univariate GJR-GARCH(1,1) model.
10. The conditional mean is modeled as a constant.
11. The conditional innovation distribution is initially Student-t.
12. GJR-GARCH is used because it captures volatility clustering and asymmetric volatility responses to negative shocks.
13. EVT will be applied to standardized residual losses, not directly to raw returns.
14. Standardized residual losses are defined as negative standardized residuals.

## Modeling Notes

1. The first GJR-GARCH(1,1)-Student-t models converged successfully for all 8 banks.
2. All estimated leverage terms gamma[1] are positive, supporting the use of an asymmetric volatility model.
3. Estimated Student-t degrees of freedom are finite and approximately between 5 and 7, indicating heavy-tailed standardized innovations.
4. The standardized residual panel contains no missing values.
5. Estimated GJR-GARCH beta[1] coefficients are high, between approximately 0.91 and 0.94, indicating persistent conditional volatility.
6. Estimated gamma[1] coefficients are positive for all banks, supporting the presence of asymmetric volatility responses.
7. Estimated Student-t degrees of freedom range from approximately 5 to 7, indicating heavy-tailed standardized innovations.
8. These results support the use of GJR-GARCH filtering before applying EVT to standardized residual losses.
9. The approximate GJR-GARCH volatility persistence measure alpha[1] + 0.5 gamma[1] + beta[1] is below one for all banks.
10. Estimated persistence values range from approximately 0.988 to 0.996, indicating highly persistent but stationary conditional volatility dynamics.

## GARCH Diagnostic Assumptions

1. GARCH diagnostics are performed on standardized residuals from the GJR-GARCH(1,1)-Student-t models.
2. Standardized residual losses are defined as negative standardized residuals.
3. The first diagnostic plots are visual diagnostics rather than formal specification tests.
4. A 60-day rolling mean of squared standardized residuals is used as a visual check for remaining volatility clustering.
5. EVT modeling proceeds only after checking that standardized residuals are reasonable.

## GARCH Diagnostic Notes

1. Standardized residual means are close to zero for all banks.
2. Standardized residual standard deviations are close to one for all banks.
3. Excess kurtosis remains positive after GJR-GARCH filtering, ranging approximately from 2.7 to 7.6.
4. Residual losses remain heavy-tailed after volatility filtering.
5. The GJR-GARCH filtering step is considered acceptable for the first EVT implementation.
6. The top standardized residual losses will be audited before final EVT estimation.

## EVT Strategy

1. The first EVT implementation uses a baseline stationary POT-GPD model on GJR-GARCH standardized residual losses.
2. More advanced EVT specifications are considered only after diagnostic evidence justifies them.
3. Exceedance clustering diagnostics are used to decide whether declustering is needed.
4. Regime-specific EVT is considered only if residual tail behavior appears materially different between calm and stress regimes.
5. The initial thesis pipeline prioritizes interpretability and diagnostic discipline over unnecessary model complexity.
6. EVT model comparison will be performed across threshold choices, declustering choices, and possibly regime-specific specifications.
7. The baseline model is a static POT-GPD model at the 95% threshold using all exceedances.
8. Robustness specifications include thresholds at 90% and 97.5%.
9. Declustering robustness specifications use runs rules with run lengths 3 and 5.
10. Regime-specific EVT is considered as the first advanced extension, using stress regimes defined by VIX or market drawdowns.
11. Covariate-dependent GPD scale models may be considered as a more advanced extension after the static and regime-specific models are implemented.

## EVT Assumptions

1. EVT is applied to standardized residual losses, not raw losses.
2. Standardized residual losses are defined as \(Y_{i,t}=-\hat z_{i,t}\).
3. The baseline POT-GPD model assumes that standardized residual losses are approximately stationary over the estimation sample.
4. The baseline POT-GPD model assumes that GJR-GARCH filtering substantially reduces volatility-driven dependence.
5. The baseline POT-GPD model initially uses all threshold exceedances.
6. Exceedance clustering will be diagnosed before final EVT estimation.
7. Runs declustering may be used as a robustness check if exceedance clustering is material.
8. Regime-specific EVT may be considered as an extension if residual tail behavior appears unstable across stress and calm regimes.

## EVT Diagnostic Notes

1. EVT threshold diagnostics are computed on standardized residual losses.
2. At the 90% threshold, each bank has approximately 539 exceedances, but exceedance clustering is moderate.
3. At the 95% threshold, each bank has approximately 270 exceedances and cluster ratios around 0.80, suggesting moderate but acceptable clustering.
4. At the 97.5% threshold, each bank has approximately 135 exceedances and clustering is mild, but parameter estimates may be noisier due to fewer observations.
5. The 95% threshold is selected as the baseline POT-GPD threshold.
6. The 90% and 97.5% thresholds are retained for robustness checks.
7. The baseline EVT model uses all exceedances.
8. Runs declustering with run length 3 is retained as a robustness check if needed.

## Baseline EVT Estimation Assumptions

1. The baseline EVT model is a static POT-GPD model.
2. The baseline threshold is the 95th percentile of standardized residual losses for each bank.
3. The baseline model uses all threshold exceedances without declustering.
4. GPD parameters are estimated by maximum likelihood.
5. Residual-level VaR and ES are computed at probability levels 99% and 99.5%.
6. Expected Shortfall is reported only when the estimated shape parameter is below one.
7. Threshold robustness and declustering robustness are performed after the baseline model is working.

## Baseline EVT Notes

1. The baseline POT-GPD model converged successfully for all 8 banks.
2. Estimated GPD shape parameters are positive for all banks, indicating heavy-tailed standardized residual losses.
3. Estimated shape parameters range approximately from 0.07 to 0.23.
4. All estimated shape parameters are below one, so residual Expected Shortfall estimates are finite.
5. The highest estimated shape parameters are observed for CBK.DE, GLE.PA, and DBK.DE.
6. Baseline 99.5% residual VaR estimates lie roughly between 3.11 and 3.38 standardized residual loss units.
7. Baseline 99.5% residual ES estimates lie roughly between 3.88 and 4.48 standardized residual loss units.

## EVT Threshold Robustness Notes

1. Threshold robustness was assessed at the 90%, 95%, and 97.5% thresholds.
2. Residual VaR estimates at the 99.5% level are broadly stable across thresholds.
3. Residual ES estimates at the 99.5% level are also broadly stable across thresholds.
4. GPD shape estimates are more sensitive than VaR and ES estimates, especially at the 97.5% threshold.
5. CBK.DE shows the strongest shape-parameter instability, with the 97.5% estimate close to zero.
6. The 95% threshold remains the baseline because it balances exceedance count, clustering, and threshold bias.
7. The 97.5% threshold is retained as a high-threshold robustness check but is not used as the main specification.

## Declustering Robustness Assumptions

1. Declustering robustness is performed at the 95% threshold.
2. Runs declustering is tested using run lengths 3 and 5.
3. For declustered specifications, only the maximum residual loss in each cluster is retained.
4. Declustered VaR and ES estimates are interpreted as robustness diagnostics rather than the main baseline forecasts.
5. A fully rigorous declustered return-level interpretation may require extremal-index adjustment.

## Declustering Robustness Notes

1. Declustering robustness was performed at the baseline 95% threshold.
2. Runs declustering with run lengths 3 and 5 reduces exceedance counts from 270 to roughly 200-225 cluster maxima per bank.
3. Estimated GPD shape parameters generally decrease after declustering.
4. CBK.DE shows the largest shape-parameter sensitivity to declustering.
5. Residual VaR estimates at the 99.5% level remain stable across no-declustering, runs-3, and runs-5 specifications.
6. Residual ES estimates at the 99.5% level are also broadly stable, though somewhat more sensitive than VaR.
7. The baseline specification remains the all-exceedance POT-GPD model at the 95% threshold.
8. Runs declustering is retained as a robustness check rather than imposed in the main model.

## EVT Diagnostic Plot Assumptions

1. Mean excess plots are used as visual diagnostics for threshold suitability.
2. GPD QQ plots are used to assess fitted tail quantiles against empirical excess quantiles.
3. GPD probability plots are used to assess fitted exceedance probabilities.
4. Diagnostic plots are interpreted qualitatively and used alongside threshold and declustering robustness checks.
5. EVT diagnostic plots are first produced for the baseline 95% threshold model.

## EVT Diagnostic Plot Notes

1. Baseline GPD probability plots are close to the diagonal for most banks, supporting the fitted POT-GPD specification.
2. Baseline GPD QQ plots show reasonable fit for the bulk of threshold exceedances, with some deviations among the largest observations.
3. BNP.PA, GLE.PA, ISP.MI, and SAN.MC show some upper-tail QQ deviations.
4. Mean excess plots are broadly consistent with heavy-tailed behavior, though not perfectly linear.
5. The 95% threshold remains acceptable when diagnostic plots are interpreted alongside threshold and declustering robustness checks.
6. EVT diagnostics support proceeding to conditional VaR and ES estimation on the original loss scale.

## Conditional VaR/ES Assumptions

1. Conditional VaR and ES are computed on the original daily loss scale.
2. Residual-level EVT VaR and ES estimates are transformed using GJR-GARCH conditional volatility.
3. The GARCH mean parameter is divided by 100 because GARCH estimation was performed on returns scaled by 100.
4. Conditional loss VaR is computed as \(-\hat\mu_i + \hat\sigma_{i,t}\widehat{VaR}_{Y_i,p}\).
5. Conditional loss ES is computed as \(-\hat\mu_i + \hat\sigma_{i,t}\widehat{ES}_{Y_i,p}\).
6. Forecasts are initially computed in-sample using fitted conditional volatility; rolling out-of-sample forecasts are reserved for the backtesting stage.

## Conditional VaR/ES Notes

1. Conditional VaR and ES forecasts were successfully computed for all 8 banks.
2. Forecasts are expressed in daily log-loss units.
3. Conditional ES exceeds conditional VaR for every bank, date, and probability level.
4. Conditional risk forecasts at 99.5% exceed corresponding forecasts at 99%.
5. Mean 99.5% conditional VaR ranges approximately from 6.2% to 8.3% across banks.
6. Mean 99.5% conditional ES ranges approximately from 7.7% to 11.3% across banks.
7. The highest average conditional risk estimates are observed for CBK.DE, GLE.PA, DBK.DE, and INGA.AS.
8. These are fitted conditional risk series; rolling out-of-sample forecasting and backtesting are separate next steps.

## Conditional Risk Plot Assumptions

1. Conditional risk plots initially focus on the 99.5% probability level.
2. Conditional VaR and ES time series are plotted in daily log-loss units.
3. Actual losses are compared against conditional VaR for selected banks.
4. VaR violations are defined as dates where actual loss exceeds conditional VaR.
5. Selected-bank plots are diagnostic and illustrative before formal backtes
## Conditional Risk Plot Notes

1. Conditional VaR and ES forecasts spike during major stress periods, especially 2008-2009 and 2020.
2. Conditional ES remains above conditional VaR throughout the sample.
3. At the 99.5% level, selected-bank violation counts are close to the expected count of approximately 27 violations over 5,389 observations.
4. CBK.DE, DBK.DE, and INGA.AS show slightly more 99.5% VaR violations than expected, while GLE.PA shows slightly fewer.
5. VaR violations visually cluster around stress periods, motivating formal unconditional and conditional coverage tests.

## VaR Backtesting Assumptions

1. VaR violations are defined as actual losses exceeding conditional VaR forecasts.
2. Backtesting is performed at the 99% and 99.5% probability levels.
3. Kupiec unconditional coverage tests whether the observed violation rate equals the nominal violation probability.
4. Christoffersen independence tests whether VaR violations are serially independent.
5. Christoffersen conditional coverage combines unconditional coverage and independence.
6. The first backtesting implementation uses fitted conditional risk series; rolling out-of-sample backtesting is a later extension.

## VaR Backtesting Notes

1. Baseline VaR backtesting was performed at the 99% and 99.5% probability levels.
2. Kupiec unconditional coverage is not rejected for any bank at either probability level.
3. At the 99.5% level, observed violation counts are close to the expected count of approximately 27 violations.
4. Christoffersen independence is not rejected for any bank at the 99.5% level.
5. At the 99% level, BNP.PA shows evidence of violation clustering, with an independence-test p-value below 5%.
6. Conditional coverage is not rejected for any bank or probability level at the 5% significance level.
7. These backtests are interpreted as fitted-sample diagnostics; rolling out-of-sample backtesting remains a later extension.


## ES Scoring Assumptions

1. ES evaluation is performed using actual losses, conditional VaR, and conditional ES forecasts.
2. The first ES evaluation uses a simple joint VaR/ES tail score.
3. Lower average ES scores indicate better tail-risk forecast performance.
4. ES scoring is used as a diagnostic and model-comparison tool rather than a standalone formal regulatory backtest.
5. More formal Fissler-Ziegel scoring functions may be added later for model comparison.

## ES Scoring Notes

1. Simple ES scoring was computed for all banks at the 99% and 99.5% levels.
2. At the 99.5% level, BNP.PA, SAN.MC, and BBVA.MC have the lowest mean ES scores.
3. At the 99.5% level, CBK.DE, GLE.PA, DBK.DE, and INGA.AS have the highest mean ES scores.
4. The ES score ranking is broadly consistent with estimated conditional ES levels and residual tail-risk estimates.
5. Maximum ES scores are large for some banks because rare VaR exceedances receive high penalties when the tail probability is small.
6. ES scoring is retained as a model-comparison diagnostic for future EVT specifications.

## Regime-Specific EVT Assumptions

1. The first regime-specific EVT model defines stress using VIX.
2. The stress regime is defined as days where VIX exceeds its 80th percentile over the aligned sample.
3. Calm and stress regimes are estimated separately.
4. Within each regime, the POT threshold is the 95th percentile of standardized residual losses in that regime.
5. Regime-specific GPD parameters are estimated by maximum likelihood.
6. Regime-specific EVT is interpreted as an advanced extension to the baseline static POT-GPD model.
7. VIX is used as a free-data stress proxy; VSTOXX or European market drawdowns may be used in later versions.
8. Rows with missing VIX are dropped for regime-specific EVT estimation.

## Regime-Specific EVT Notes

1. The VIX-based regime split produced 4,221 calm observations and 1,055 stress observations.
2. With a 95% within-regime threshold, each bank has 211 calm exceedances and 53 stress exceedances.
3. Stress-regime POT thresholds are higher than calm-regime thresholds for all banks.
4. Stress-regime GPD scale estimates exceed calm-regime scale estimates for all banks.
5. Stress-regime GPD shape estimates exceed calm-regime shape estimates for seven out of eight banks.
6. GLE.PA has a lower stress-regime shape estimate than calm-regime shape estimate, but a substantially higher stress-regime scale estimate.
7. At the 99.5% level, stress-regime residual VaR and ES exceed calm-regime residual VaR and ES for every bank.
8. Stress-regime parameter estimates are based on approximately 53 exceedances per bank and should be interpreted cautiously.

## Regime-Specific Conditional VaR/ES Assumptions

1. Regime-specific conditional VaR and ES forecasts use the VIX-based stress-regime indicator.
2. On calm days, conditional forecasts use calm-regime residual EVT estimates.
3. On stress days, conditional forecasts use stress-regime residual EVT estimates.
4. GARCH conditional volatility remains the same as in the baseline model.
5. The GARCH mean parameter is converted to raw return units by dividing by 100.
6. Regime-specific conditional forecasts are compared against the baseline static EVT forecasts using the same VaR backtests and ES scores.

## Regime-Specific Conditional VaR/ES Notes

1. Regime-specific conditional VaR and ES forecasts were successfully computed for all banks.
2. At the 99.5% level, mean stress-regime conditional VaR exceeds mean calm-regime conditional VaR for every bank.
3. At the 99.5% level, mean stress-regime conditional ES exceeds mean calm-regime conditional ES for every bank.
4. Stress-regime mean conditional VaR is more than twice calm-regime mean conditional VaR for every bank.
5. Stress-regime mean conditional ES is more than twice calm-regime mean conditional ES for every bank.
6. INGA.AS shows the largest stress-to-calm amplification in both VaR and ES.
7. Regime-specific conditional forecasts remain fitted-sample diagnostics until rolling out-of-sample validation is implemented.

## Baseline vs Regime-Specific Risk Comparison Assumptions

1. Baseline conditional VaR and ES use static residual EVT estimates.
2. Regime-specific conditional VaR and ES use calm or stress residual EVT estimates depending on the VIX-based regime indicator.
3. Baseline and regime-specific forecasts are compared over their common date range.
4. Comparisons are computed separately for calm and stress days.
5. A useful regime-specific model should generally reduce risk forecasts in calm periods and increase risk forecasts in stress periods relative to the static baseline.

## Baseline vs Regime-Specific Risk Comparison Notes

1. The regime-specific model produces lower conditional VaR and ES than the baseline model on calm days for every bank and probability level.
2. The regime-specific model produces higher conditional VaR and ES than the baseline model on stress days for every bank and probability level.
3. At the 99.5% level, regime-specific stress VaR exceeds baseline stress VaR by approximately 1.9 to 3.9 percentage points.
4. At the 99.5% level, regime-specific stress ES exceeds baseline stress ES by approximately 2.8 to 5.6 percentage points.
5. The comparison supports the interpretation that static EVT averages across calm and stress tail regimes.
6. Predictive superiority is not inferred from this comparison alone and must be evaluated using backtesting and scoring.

## Regime-Specific VaR Backtesting Assumptions

1. Regime-specific VaR backtesting uses the same loss series as the baseline backtesting exercise.
2. VaR violations are defined as actual losses exceeding regime-specific conditional VaR.
3. Backtesting is performed at the 99% and 99.5% probability levels.
4. Results are compared against baseline VaR backtesting results.
5. The regime-specific model is evaluated as a fitted-sample diagnostic before any rolling out-of-sample implementation.

## Regime-Specific VaR Backtesting Notes

1. Regime-specific VaR backtesting was performed on 5,276 aligned observations.
2. Kupiec unconditional coverage is not rejected for any bank at the 99% or 99.5% probability levels.
3. Christoffersen independence is not rejected for any bank at either probability level.
4. Conditional coverage is not rejected for any bank at either probability level.
5. At the 99.5% level, no bank has consecutive VaR violations.
6. Compared with the baseline model, the regime-specific model removes the BNP.PA 99% independence rejection.
7. Results are interpreted as fitted-sample diagnostics pending rolling out-of-sample validation.

## Regime-Specific ES Scoring Assumptions

1. Regime-specific ES scoring uses the same simple joint VaR/ES score as the baseline model.
2. The score is computed using actual losses, regime-specific conditional VaR, and regime-specific conditional ES.
3. Lower average scores indicate better tail-risk forecast performance.
4. Scores are compared against baseline ES scores over the common aligned sample.
5. Results are interpreted as fitted-sample diagnostics before rolling out-of-sample validation.

## Regime-Specific ES Scoring Notes

1. Regime-specific ES scoring was computed for all banks at the 99% and 99.5% levels.
2. At the 99.5% level, the regime-specific model lowers the mean ES score for every bank relative to the static baseline.
3. The largest 99.5% score improvements occur for banks with stronger stress-tail behavior, especially CBK.DE, GLE.PA, DBK.DE, and ISP.MI.
4. Violation rates remain close to nominal levels under the regime-specific model.
5. The regime-specific model improves fitted-sample ES scoring while preserving VaR calibration.
6. Results remain fitted-sample diagnostics until rolling out-of-sample validation is implemented.

## Baseline vs Regime Model Evaluation Assumptions

1. Baseline and regime-specific models are compared using VaR backtesting and simple ES scoring.
2. VaR comparison uses Kupiec unconditional coverage, Christoffersen independence, and conditional coverage p-values.
3. ES scoring comparison uses mean simple ES scores by asset and probability level.
4. Lower ES scores indicate better tail-risk forecast performance.
5. ES score improvement is computed as the percentage reduction in mean ES score from baseline to regime-specific EVT.
6. Comparisons are interpreted as fitted-sample diagnostics over the available aligned samples.

## Baseline vs Regime Model Evaluation Notes

1. The regime-specific model improves the mean ES score for every bank at both the 99% and 99.5% probability levels.
2. At the 99.5% level, ES score improvements range from approximately 3.6% to 13.8%.
3. The largest 99.5% ES score improvements are observed for CBK.DE, GLE.PA, DBK.DE, and ISP.MI.
4. Both baseline and regime-specific models pass Kupiec unconditional coverage tests for all banks and probability levels.
5. The baseline model has one Christoffersen independence rejection at the 99% level for BNP.PA.
6. The regime-specific model has no Christoffersen independence rejection at either probability level.
7. The regime-specific model preserves VaR calibration while improving fitted-sample ES scoring and violation independence diagnostics.
8. Results remain fitted-sample diagnostics until rolling out-of-sample validation is implemented.

## Baseline vs Regime Risk Plot Assumptions

1. Baseline and regime-specific risk forecasts are plotted over their common date range.
2. Plots focus first on the 99.5% probability level.
3. Selected-bank plots focus on banks with relatively high tail-risk estimates.
4. Difference plots are computed as regime-specific risk minus baseline risk.
5. Positive differences indicate higher regime-specific risk forecasts than the static baseline.
6. Negative differences indicate lower regime-specific risk forecasts than the static baseline.

## Baseline vs Regime Risk Plot Notes

1. Baseline and regime-specific VaR/ES forecasts are visually close during many normal periods.
2. Regime-specific forecasts are generally lower than baseline forecasts in calm regimes.
3. Regime-specific forecasts rise above baseline forecasts during VIX-defined stress regimes.
4. Difference plots confirm that regime-specific minus baseline risk is mostly negative in calm periods and positive during stress periods.
5. The largest positive differences occur around major stress episodes, especially 2008-2009 and 2020.
6. ES differences are larger than VaR differences, consistent with ES being more sensitive to tail severity.

## Tail Dependence Assumptions

1. Tail dependence is initially estimated nonparametrically from bank loss exceedance indicators.
2. Losses are defined as negative log returns.
3. A bank is in tail distress at level q when its loss exceeds its empirical q-quantile.
4. Pairwise tail dependence is estimated as the conditional probability that one bank is distressed given that another bank is distressed.
5. Tail dependence is computed for q = 0.90, 0.95, and 0.975.
6. The 0.95 threshold is used as the main tail-dependence level.
7. Estimates at q = 0.975 are interpreted cautiously because fewer joint exceedances are available.

## Tail Dependence Notes

1. Pairwise tail-dependence estimates were computed successfully for q = 0.90, 0.95, and 0.975.
2. At q = 0.95, the strongest pairwise links are BBVA.MC-SAN.MC and BNP.PA-GLE.PA.
3. At q = 0.95, the highest average tail connectedness is observed for GLE.PA, BNP.PA, SAN.MC, INGA.AS, and BBVA.MC.
4. CBK.DE has the lowest average tail connectedness at q = 0.95.
5. Tail-connectedness rankings differ from marginal tail-risk rankings, showing that standalone risk and systemic co-tail exposure are distinct.
6. Incoming and outgoing connectedness are identical in this first implementation because banks have equal empirical exceedance counts at common quantile thresholds.
7. The q = 0.975 estimates are interpreted cautiously because fewer exceedance observations are available.

## Tail Dependence Heatmap Assumptions

1. The heatmap visualizes pairwise nonparametric tail dependence at the 95% loss threshold.
2. Rows represent the bank becoming extreme.
3. Columns represent the conditioning bank already in an extreme-loss state.
4. Darker or larger heatmap values indicate stronger joint downside tail dependence.
5. The diagonal equals one by construction and is included for reference.

## Tail Dependence Plot Notes

1. The q = 0.95 tail-dependence heatmap shows strong within-country tail links, especially BBVA.MC-SAN.MC and BNP.PA-GLE.PA.
2. Cross-country tail dependence is also visible, especially among BNP.PA, GLE.PA, and INGA.AS.
3. The average connectedness ranking places GLE.PA and BNP.PA at the top.
4. CBK.DE has the lowest average tail connectedness at q = 0.95.
5. Tail connectedness differs from marginal tail risk, showing that individual tail severity and systemic co-tail exposure should be interpreted separately.

## Tail Dependence Network Assumptions

1. The tail-dependence network is built from pairwise nonparametric tail-dependence estimates at q = 0.95.
2. Banks are represented as nodes.
3. Edges are included when pairwise tail dependence is at least 0.55.
4. Edge weights equal the pairwise tail-dependence estimate.
5. The first network visualization treats links as undirected because the empirical matrix is symmetric under common quantile thresholds.
6. The network is descriptive and does not imply causal contagion.

## Tail Dependence Network Notes

1. The q = 0.95 tail-dependence network uses an edge threshold of 0.55.
2. The strongest retained edge is BBVA.MC-SAN.MC, followed by BNP.PA-GLE.PA.
3. The connected core includes BBVA.MC, SAN.MC, BNP.PA, GLE.PA, INGA.AS, and ISP.MI.
4. DBK.DE and CBK.DE are isolated at the 0.55 threshold.
5. GLE.PA acts as a central bridge in the retained network, linking the French pair to Spanish and Dutch/Italian nodes.
6. The network is descriptive and does not imply causal contagion.

## Tail Dependence Network Robustness Assumptions

1. The main tail-dependence network uses an edge threshold of 0.55.
2. A robustness network is also computed using an edge threshold of 0.50.
3. The robustness network is used to assess whether isolated nodes in the main network remain weakly connected under a less restrictive cutoff.
4. The network remains descriptive and does not imply causal contagion.

## Tail Dependence Network Robustness Notes

1. Lowering the network threshold from 0.55 to 0.50 preserves the main euro-area co-tail core.
2. The strongest edges remain BBVA.MC-SAN.MC and BNP.PA-GLE.PA.
3. DBK.DE enters the connected component at the 0.50 threshold through several moderate links.
4. CBK.DE remains peripheral and has only one retained edge at the 0.50 threshold.
5. INGA.AS becomes the highest weighted-degree node at the 0.50 threshold, suggesting broad moderate co-tail connectedness.
6. The robustness network supports the distinction between strong tail links and broader moderate tail connectedness.

## Empirical CoVaR Assumptions

1. System loss is defined as the equal-weighted average loss across included banks.
2. Bank distress is defined as the bank loss exceeding its empirical q-level loss quantile.
3. Normal bank state is defined as the bank loss being less than or equal to its empirical median loss.
4. Distress CoVaR is estimated as the empirical q-level quantile of system loss conditional on bank distress.
5. Normal CoVaR is estimated as the empirical q-level quantile of system loss conditional on the bank being in a normal state.
6. Delta CoVaR is defined as distress CoVaR minus normal CoVaR.
7. CoVaR estimates are interpreted as conditional association measures, not causal contagion estimates.
8. CoVaR is computed for q = 0.90, 0.95, and 0.975, with q = 0.95 as the main level.

## Empirical CoVaR Notes

1. Empirical CoVaR was computed successfully for q = 0.90, 0.95, and 0.975.
2. At q = 0.95, all banks have positive Delta CoVaR estimates.
3. BNP.PA, DBK.DE, and GLE.PA have the highest Delta CoVaR estimates at q = 0.95.
4. Distress CoVaR values are very similar across banks at q = 0.95, indicating strong overlap in bank distress days during system-wide stress episodes.
5. Delta CoVaR rankings are partly driven by differences in normal-state CoVaR.
6. BNP.PA, GLE.PA, and DBK.DE remain near the top across multiple CoVaR thresholds.
7. CoVaR is interpreted as conditional systemic-risk association, not causal contagion.

## Empirical CoVaR Plot Assumptions

1. Delta CoVaR plots use empirical CoVaR estimates based on equal-weighted system loss.
2. The main ranking plot uses q = 0.95.
3. The robustness plot compares Delta CoVaR across q = 0.90, 0.95, and 0.975.
4. Higher Delta CoVaR indicates a larger increase in system tail risk conditional on bank distress.
5. CoVaR plots are interpreted as conditional systemic-risk association, not causal contagion.

## Empirical CoVaR Plot Notes

1. The q = 0.95 Delta CoVaR ranking places BNP.PA, DBK.DE, and GLE.PA at the top.
2. Cross-bank Delta CoVaR differences are small at q = 0.95, indicating a compressed ranking.
3. Distress CoVaR is much higher than normal CoVaR for every bank.
4. Distress CoVaR values are very similar across banks, suggesting strong overlap in bank distress days during systemic stress periods.
5. Delta CoVaR increases monotonically across q = 0.90, 0.95, and 0.975 for all banks.
6. Empirical CoVaR is useful for measuring broad conditional system stress association, while pairwise tail dependence gives more granular cross-bank information.

## Integrated Systemic-Risk Summary Assumptions

1. The systemic-risk summary combines marginal tail severity, tail connectedness, Delta CoVaR, and tail-network centrality.
2. Marginal tail severity is measured using mean conditional ES at the 99.5% level from the regime-specific model.
3. Tail connectedness is measured using average total tail connectedness at q = 0.95.
4. Systemic contribution is measured using empirical Delta CoVaR at q = 0.95.
5. Network centrality is measured using weighted degree from the q = 0.95 tail-dependence network with a 0.50 edge threshold.
6. Lower numerical ranks indicate higher systemic relevance.
7. The average rank is an equal-weighted summary of the four component ranks.
8. The integrated score is descriptive and does not imply causal systemic importance.

## Integrated Systemic-Risk Summary Notes

1. GLE.PA has the strongest overall systemic-risk profile, with the lowest average rank.
2. BNP.PA and INGA.AS rank second jointly by average rank, but for different reasons.
3. BNP.PA ranks highest in Delta CoVaR and second in average tail connectedness.
4. INGA.AS has the highest network weighted degree in the 0.50-threshold robustness network.
5. CBK.DE has the highest mean 99.5% ES but weak tail connectedness and low network centrality.
6. The integrated results show that marginal tail severity and systemic co-tail relevance are distinct dimensions of bank risk.
7. The composite ranking is descriptive and equal-weighted across the selected systemic-risk metrics.

## Integrated Systemic-Risk Plot Assumptions

1. The integrated systemic-risk plots use the equal-weighted composite ranking from the systemic-risk summary table.
2. Lower average rank indicates higher systemic relevance across the selected metrics.
3. Component rank plots show the separate rankings for marginal ES, tail connectedness, Delta CoVaR, and network weighted degree.
4. The plots are descriptive and do not imply causal systemic importance.

## Integrated Systemic-Risk Plot Notes

1. The average-rank plot identifies GLE.PA as the strongest integrated systemic-risk name.
2. BNP.PA and INGA.AS form the next-highest systemic-risk group, with equal average ranks.
3. GLE.PA ranks near the top across all four components, making it the most balanced systemic-risk profile.
4. BNP.PA is especially important through Delta CoVaR and tail connectedness.
5. INGA.AS is especially important through network weighted degree.
6. CBK.DE ranks first in marginal ES but last in tail connectedness and network weighted degree.
7. The plots reinforce that marginal tail risk and systemic connectedness are distinct dimensions.

## Regime-Specific Tail Dependence Assumptions

1. Regime-specific tail dependence is estimated separately on calm and stress days.
2. The stress regime is defined using the previously constructed VIX-based stress indicator.
3. Within each regime, bank-specific empirical loss thresholds are computed at q = 0.90, 0.95, and 0.975.
4. Pairwise tail dependence is estimated as the conditional probability that one bank is extreme given that another bank is extreme within the same regime.
5. Stress-minus-calm differences are used to measure whether pairwise tail dependence strengthens during stress periods.
6. Estimates in the stress regime are based on fewer observations and are interpreted cautiously.
7. Regime-specific tail dependence is descriptive and does not imply causal contagion.

## Regime-Specific Tail Dependence Notes

1. Average tail connectedness is higher in the stress regime than in the calm regime for every bank at q = 0.95.
2. The largest stress-minus-calm connectedness increases are observed for CBK.DE, DBK.DE, BBVA.MC, and INGA.AS.
3. CBK.DE is relatively peripheral in the static network but exhibits the largest connectedness increase during stress.
4. The largest pairwise stress-minus-calm increase occurs for CBK.DE-DBK.DE.
5. Other large stress-amplified links include CBK.DE-INGA.AS and DBK.DE-INGA.AS.
6. The BNP.PA-GLE.PA link changes little and slightly declines, suggesting persistent high co-tail dependence across regimes rather than stress activation.
7. These results support the interpretation that systemic tail connectedness is regime-dependent.

## Regime-Specific Tail Dependence Plot Assumptions

1. Calm and stress tail-dependence heatmaps are plotted at q = 0.95.
2. The stress-minus-calm heatmap reports the change in pairwise tail dependence from calm to stress regimes.
3. Positive stress-minus-calm values indicate stronger tail dependence during stress periods.
4. Connectedness bar plots compare average total tail connectedness across calm and stress regimes.
5. Regime-specific plots are descriptive and do not imply causal contagion.

## Regime-Specific Tail Dependence Plot Notes

1. Stress-regime tail dependence is stronger and more widespread than calm-regime tail dependence.
2. Average tail connectedness is higher in stress than calm for every bank.
3. The largest pairwise stress-minus-calm increase is CBK.DE-DBK.DE.
4. Large stress-amplified links also involve CBK.DE-INGA.AS and DBK.DE-INGA.AS.
5. BNP.PA-GLE.PA remains strong across regimes and changes little, indicating persistent rather than stress-activated dependence.
6. The results support the main thesis claim that systemic tail connectedness is regime-dependent.

## Student-t Copula Benchmark Assumptions

1. The Student-t copula benchmark is fitted to empirical pseudo-observations of bank losses.
2. Each bank loss series is transformed to pseudo-uniform margins using ranks divided by n + 1.
3. The Student-t copula models cross-sectional dependence through a correlation matrix and a common degrees-of-freedom parameter.
4. The copula benchmark is parametric and symmetric in lower and upper tails.
5. Copula-implied tail dependence is compared with empirical tail-dependence estimates.
6. The Student-t copula benchmark is used as a robustness and comparison model, not as the main thesis contribution.
7. The first implementation uses static dependence over the full aligned sample; regime-specific copulas may be considered later.

## Student-t Copula Plot Assumptions

1. Copula comparison plots use simulated finite-threshold Student-t copula tail dependence at q = 0.95.
2. The empirical comparison uses the nonparametric q = 0.95 tail-dependence matrix.
3. Positive empirical-minus-copula differences indicate empirical links stronger than the static copula benchmark.
4. Negative differences indicate links where the copula benchmark exceeds empirical tail dependence.
5. The plots evaluate the static Student-t copula as a parametric benchmark, not as the main dependence model.

## Student-t Copula Plot Notes

1. The Student-t copula finite-threshold heatmap broadly captures the main static tail-dependence structure.
2. The empirical-minus-copula heatmap is positive for most off-diagonal links.
3. The largest empirical-minus-copula gaps involve INGA.AS, GLE.PA, BNP.PA, and ISP.MI.
4. The largest-error plot shows that the static Student-t copula mostly underestimates the strongest empirical pairwise tail-dependence links.
5. The copula benchmark supports the interpretation that static parametric dependence captures average heavy-tailed dependence but smooths over pair-specific and regime-dependent structure.
 
 ## Stress Activation Index Assumptions

1. The Stress Activation Index is defined as stress-regime average tail connectedness minus calm-regime average tail connectedness.
2. Persistent Connectedness is defined as the average of calm- and stress-regime average tail connectedness.
3. Both measures are computed at the main tail-dependence threshold q = 0.95.
4. Banks are classified using sample median splits of Persistent Connectedness and Stress Activation.
5. The classification is descriptive and intended to distinguish persistent systemic connectedness from stress-activated systemic connectedness.
6. The Stress Activation Index does not imply causal contagion.

## Stress Activation Index Notes

1. CBK.DE has the highest Stress Activation Index at q = 0.95.
2. CBK.DE also has the lowest Persistent Connectedness, indicating that it is not persistently central but becomes much more connected during stress.
3. DBK.DE and INGA.AS are also classified as stress activators.
4. BBVA.MC is classified as a stress-amplified core bank, combining high Persistent Connectedness with high Stress Activation.
5. SAN.MC, GLE.PA, and BNP.PA are classified as persistent core banks.
6. ISP.MI is classified as peripheral under the median-split classification.
7. The largest pairwise stress activation is CBK.DE-DBK.DE, increasing from approximately 0.384 in calm regimes to 0.698 in stress regimes.
8. The Stress Activation Index supports the distinction between persistent systemic connectedness and stress-activated systemic connectedness.

## Stress Activation Index Plot Assumptions

1. Stress Activation Index plots are computed at q = 0.95.
2. The scatter plot uses Persistent Connectedness on the horizontal axis and Stress Activation Index on the vertical axis.
3. Median lines separate banks into four descriptive groups: persistent core, stress activator, stress-amplified core, and peripheral.
4. The bar plots rank banks by Stress Activation Index and Persistent Connectedness.
5. The classification is descriptive and does not imply causal contagion.

## Stress Activation Index Plot Notes

1. The scatter plot separates banks into four groups using median splits of Persistent Connectedness and Stress Activation.
2. CBK.DE is the clearest stress activator, with the highest Stress Activation Index and the lowest Persistent Connectedness.
3. DBK.DE and INGA.AS are also classified as stress activators.
4. BBVA.MC is the only stress-amplified core bank.
5. SAN.MC, GLE.PA, and BNP.PA are classified as persistent core banks.
6. ISP.MI is classified as peripheral.
7. The classification supports the thesis distinction between persistent systemic connectedness and stress-activated systemic connectedness.

## Estimation Assumptions

1. Descriptive statistics are computed using the balanced bank return panel.
2. Descriptive statistics are computed on log returns, not simple returns.
3. Skewness and kurtosis are included to document non-Gaussian behavior.
4. The descriptive table is exported both as CSV and LaTeX.

## Visualization Assumptions

1. Initial exploratory plots are produced from the balanced bank return and loss panels.
2. Rolling volatility is computed using a 60-trading-day rolling standard deviation.
3. The first plots are diagnostic and descriptive; they are not model outputs.
4. Plots are saved as PNG files for quick inspection.

## Backtesting Assumptions

_To be added._

## Open Questions

2. The free-data bank panel contains small numbers of missing adjusted prices across tickers. We need to decide whether to drop dates with missing values or use limited forward-filling before return construction.
3. The market price panel starts on 2007-03-30 because the banking-sector ETF proxy has a shorter history.
4. VIX contains missing values, likely due to holidays and calendar mismatches. We need to align stress variables to the bank trading calendar.