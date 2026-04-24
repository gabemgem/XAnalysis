# v2 — Collateralized Auction Optimization

This directory contains code for simulating and optimizing a *collateralized* second-price auction mechanism, where a threshold function τ (tau) gates advertiser participation based on the externality their content produces. The goal is to find the optimal τ that maximizes social welfare relative to a standard VCG auction counterfactual.

Advertiser values (`v`) are proxied by tweet engagement (actions per month) and externalities (`e`) are derived from Community Notes ratings.

---

## Files

### Data Pipeline

**[get_tweets.ipynb](get_tweets.ipynb)**
Fetches tweets from the Twitter/X API using Tweepy and stores them in a MySQL database. Reads tweet IDs from a pickle file, batches requests in groups of 100, and respects rate limits with a 70-second sleep between batches.

**[create_distribution.ipynb](create_distribution.ipynb)**
Queries the MySQL database for tweets, Community Notes, and note ratings. Computes an externality score for each tweet by combining note classifications (misleading vs. not) with helpfulness ratings from raters. Normalizes scores by impression count and exports the final dataset as `full_tweets.csv`, which is the primary real-data input for the auction optimizer.

**[full_tweets.csv](full_tweets.csv)**
Processed tweet dataset produced by `create_distribution.ipynb`. Each row is a tweet with columns including `v_score` (advertiser value, proportional to actions per month) and `e_score` (externality, normalized Community Notes signal per 1000 impressions per month).

### Synthetic Distributions

**[alternate_distributions.ipynb](alternate_distributions.ipynb)**
Generates synthetic 2D advertiser (e, v) distributions for controlled experiments. Produces six named distributions saved to `data/samples/`:
- `a` — single-modal normal
- `b` — two-modal side by side
- `c` / `d` — two-modal diagonal (/ and \\)
- `e` — two-modal stacked
- `f` — four-modal

Scatter and KDE density plots are saved to `data/figures/`.

### Auction Optimizer

**[Collateralized_Auction_genetic.py](Collateralized_Auction_genetic.py)**
Exploratory, cell-structured script (percent-cell format) for running the genetic algorithm auction optimizer interactively. Defines:
- `run_auction` — simulates one VCG auction and one collateralized auction for a set of advertisers
- `tau` — polynomial threshold function evaluated as `v ≥ τ(e, coefficients)`
- `run_ga` — wraps PyGAD to search for optimal τ coefficients by maximizing average collateralized welfare across many auction draws
- Plotting utilities (`plot_auctions`, `plot_advertisers`)

Coefficients are searched in log-space (`ln_coeffs_to_coeffs`) to allow the GA to explore a wider dynamic range.

**[Collateralized_Auction_genetic_script.py](Collateralized_Auction_genetic_script.py)**
Production CLI version of the optimizer, designed for submission to an HPC cluster. Key differences from the exploratory script:
- Accepts command-line arguments (`--k`, `--externality-cost`, `--polynomial-degree`, `--data`, etc.)
- Normalizes `v` and `e` jointly to `[-1, 1]` before running the GA, then converts the best τ coefficients back to original space using `NegOneOneScaler.poly_from_normalized`
- Saves results to `data/output/ga_results_{id}.pkl`
- Includes a `NegOneOneScaler` class with full forward/inverse polynomial coefficient transforms via Horner's rule

### Cluster Scripts

**[cluster_run.sh](cluster_run.sh)**
SGE array job script that submits one distribution file and one polynomial degree to the cluster. Runs 7 parallel tasks (one per externality cost variation: 0.001, 0.005, 0.0075, 0.01, 0.025, 0.05, 0.1). Uses more generations (4000) for polynomial degrees > 1.

Usage: `qsub cluster_run.sh <data_file.csv> <polynomial_degree>`

**[cluster_run_all.sh](cluster_run_all.sh)**
Convenience script that submits all combinations of distributions (0, a–f) × polynomial degrees (1, 2, 3) by calling `cluster_run.sh` repeatedly. Run this to kick off a full sweep.

### Analysis & Plots

**[Create_Plots.ipynb](Create_Plots.ipynb)**
Loads GA result pickle files from `data/output/` and generates all figures. Produces:
- Scatter plots of advertiser vs. externality welfare across externality cost and polynomial degree
- Line plots of welfare change (advertiser, externality, total) vs. externality cost
- Side-by-side visualizations of optimal affine/quadratic/cubic τ functions overlaid on advertiser scatter plots, for each distribution
- Plots of the full space of τ functions explored during GA search
- Welfare distribution histograms comparing collateralized vs. VCG outcomes

**[ev_density_plot.png](ev_density_plot.png)**
KDE density contour plot of the empirical (externality, advertiser value) joint distribution from real tweet data, saved by `Create_Plots.ipynb`.

### Reference

**[Audited_Auctions_EC_25_Supplemental-1.pdf](Audited_Auctions_EC_25_Supplemental-1.pdf)**
Supplemental document for the associated research paper submitted to EC 2025.

---

## Workflow

```
get_tweets.ipynb          → MySQL DB
create_distribution.ipynb → full_tweets.csv
alternate_distributions.ipynb → data/samples/*.csv

cluster_run_all.sh
  └─ cluster_run.sh (×21 jobs)
       └─ Collateralized_Auction_genetic_script.py → data/output/ga_results_*.pkl

Create_Plots.ipynb        → data/figures/*.png
```
