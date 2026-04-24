import random
import pandas as pd
import numpy as np
import time
import pickle as pkl
import sys
import itertools

from numpy.polynomial import Polynomial


def _compose_with_linear(coeffs, a, b):
    """
    Return Polynomial p(ax + b) given p(x) with increasing-order coeffs.
    Uses Horner's rule; works on old NumPy w/o Polynomial.compose.
    """
    L = Polynomial([b, a])
    q = Polynomial([0.0])
    for c in reversed(coeffs):
        q = q * L + c
    return q


class NegOneOneScaler:
    """
    Normalize numeric data to [-1, 1] and invert later.
    Works with pandas Series or DataFrame. Includes general polynomial coefficient transforms.
    """
    def __init__(self, feature_range=(-1.0, 1.0), joint=False):
        self.feature_range = feature_range
        self.joint = joint
        self.data_min_ = None
        self.data_max_ = None
        self.columns_ = None
        self._fitted = False

    def fit(self, X):
        if isinstance(X, pd.Series):
            self.columns_ = [X.name] if X.name is not None else [0]
            self.data_min_ = pd.Series([X.min()], index=self.columns_)
            self.data_max_ = pd.Series([X.max()], index=self.columns_)
        elif isinstance(X, pd.DataFrame):
            self.columns_ = list(X.columns)
            if self.joint:
                gmin, gmax = X.min().min(), X.max().max()
                self.data_min_ = pd.Series([gmin]*len(self.columns_), index=self.columns_)
                self.data_max_ = pd.Series([gmax]*len(self.columns_), index=self.columns_)
            else:
                self.data_min_ = X.min()
                self.data_max_ = X.max()
        else:
            raise TypeError("X must be a pandas Series or DataFrame")
        self._fitted = True
        return self

    def transform(self, X):
        self._check_is_fitted()
        a, b = self.feature_range
        rng = self.data_max_ - self.data_min_

        def _scale(s, col):
            denom = rng[col]
            if denom == 0 or np.isclose(denom, 0):
                return pd.Series(np.full(len(s), (a + b) / 2.0), index=s.index, name=s.name)
            return ((s - self.data_min_[col]) / denom) * (b - a) + a

        if isinstance(X, pd.Series):
            col = self.columns_[0]
            return _scale(X, col)
        elif isinstance(X, pd.DataFrame):
            missing = set(self.columns_) - set(X.columns)
            if missing:
                raise ValueError(f"X is missing columns seen during fit: {missing}")
            return X[self.columns_].apply(lambda s: _scale(s, s.name))
        else:
            raise TypeError("X must be a pandas Series or DataFrame")

    def inverse_transform(self, X):
        self._check_is_fitted()
        a, b = self.feature_range
        scale = (b - a)
        rng = self.data_max_ - self.data_min_

        def _inv(s, col):
            denom = rng[col]
            if denom == 0 or np.isclose(denom, 0):
                return pd.Series(np.full(len(s), self.data_min_[col]), index=s.index, name=s.name)
            return ((s - a) / scale) * denom + self.data_min_[col]

        if isinstance(X, pd.Series):
            col = self.columns_[0]
            return _inv(X, col)
        elif isinstance(X, pd.DataFrame):
            missing = set(self.columns_) - set(X.columns)
            if missing:
                raise ValueError(f"X is missing columns seen during fit: {missing}")
            return X[self.columns_].apply(lambda s: _inv(s, s.name))
        else:
            raise TypeError("X must be a pandas Series or DataFrame")

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def _affine_params(self, col):
        a, b = self.feature_range
        min_v = self.data_min_[col]
        max_v = self.data_max_[col]
        rng = max_v - min_v
        if np.isclose(rng, 0):
            raise ValueError(f"Column '{col}' is constant; polynomial transforms are ill-defined.")
        alpha = (b - a) / rng
        beta = a - alpha * min_v
        return alpha, beta

    def poly_from_normalized(self, coeffs_n, x_col, y_col):
        self._check_is_fitted()
        if not (2 <= len(coeffs_n) <= 4):
            raise ValueError("Expected polynomial length 2-4 (linear/quadratic/cubic).")
        ax, bx = self._affine_params(x_col)
        ay, by = self._affine_params(y_col)
        p_in_x = _compose_with_linear(coeffs_n, ax, bx)
        p_y = (p_in_x - by) / ay
        coefs = p_y.coef
        coefs[np.isclose(coefs, 0, atol=1e-15)] = 0.0
        return coefs.tolist()

    def _check_is_fitted(self):
        if not self._fitted:
            raise RuntimeError("Scaler is not fitted. Call fit(X) or fit_transform(X) first.")


def tau(externality, coeffs):
    tau_val = 0
    for i, coeff in enumerate(coeffs):
        tau_val += coeff * (externality**i)
    return tau_val


def auction_objective(w_results, ad_scaler=1.0, ext_scaler=1.0):
    return (w_results[0] * ad_scaler) + (w_results[1] * ext_scaler)


def run_auction(advertisers, tau_coeffs, k=1):
    counterfactual_bids = sorted(advertisers, key=lambda x: x[0])
    top_bids = counterfactual_bids[-k:]
    w_vcg = (sum(b[0] for b in top_bids), sum(b[1] for b in top_bids))

    collateralized_bids = sorted(
        [ad for ad in advertisers if ad[0] >= tau(ad[1], tau_coeffs)],
        key=lambda x: x[0]
    )
    top_bids = collateralized_bids[-k:]
    w_coll = (sum(b[0] for b in top_bids), sum(b[1] for b in top_bids))

    return w_vcg, w_coll


def run_auction_set(tau_coeffs, advertisers, k=1):
    w_vcg, w_coll = [], []
    for advertiser_set in advertisers:
        w_vcg_i, w_coll_i = run_auction(advertiser_set, tau_coeffs, k)
        w_vcg.append(w_vcg_i)
        w_coll.append(w_coll_i)

    w_vcg_ad  = [w[0] for w in w_vcg]
    w_coll_ad = [w[0] for w in w_coll]
    w_vcg_ex  = [w[1] for w in w_vcg]
    w_coll_ex = [w[1] for w in w_coll]
    individual_welfares = {
        'vcg_ad': w_vcg_ad, 'coll_ad': w_coll_ad,
        'vcg_ex': w_vcg_ex, 'coll_ex': w_coll_ex,
    }
    return (
        tau_coeffs,
        np.mean([auction_objective(w) for w in w_coll]),
        np.mean([auction_objective(w) for w in w_vcg]),
        [auction_objective(w) for w in w_coll],
        [auction_objective(w) for w in w_vcg],
        individual_welfares,
    )


def _eval_grid_on_auction(grid, advertiser_set, k, ext_scaler):
    """Vectorized welfare for every grid coefficient vector on one auction draw.

    Computes tau(e_i) for all G grid points and all n advertisers at once via a
    single matrix multiply, then selects top-k admitted winners per grid point.

    Parameters
    ----------
    grid : (G, degree+1) float array
    advertiser_set : list of (v, e) tuples
    Returns (G,) array of collateralized welfare values.
    """
    v = np.array([a[0] for a in advertiser_set])  # (n,)
    e = np.array([a[1] for a in advertiser_set])  # (n,)
    degree = grid.shape[1]

    # e_powers[i, p] = e[i]^p  →  shape (n, degree)
    e_powers = np.column_stack([e**p for p in range(degree)])

    # tau_values[g, i] = tau(e[i]) under coefficient vector grid[g]  →  (G, n)
    tau_values = grid @ e_powers.T

    # admitted[g, i] = True when advertiser i passes the threshold under grid[g]
    admitted = v[np.newaxis, :] >= tau_values  # (G, n)

    # Sort advertisers by v descending so top-k selection is a prefix operation
    order = np.argsort(-v)
    admitted_sorted = admitted[:, order]  # (G, n)
    v_sorted = v[order]                   # (n,)
    e_sorted = e[order]                   # (n,)

    # cumsum[g, j] = number of admitted advertisers in positions 0..j for grid[g]
    # top_k[g, j] = True when position j is among the top-k admitted
    cumsum = np.cumsum(admitted_sorted, axis=1)
    top_k = (cumsum <= k) & admitted_sorted  # (G, n)

    # Welfare = sum of v + ext_scaler*e for top-k winners  →  (G,)
    return top_k @ v_sorted + (top_k @ e_sorted) * ext_scaler


def run_grid_search(advertisers, polynomial_degree, k=1, num_points=20, ext_scaler=1.0):
    """
    Exhaustive grid search over tau polynomial coefficients in normalized space.

    Evaluates all grid points simultaneously using vectorized numpy operations:
    for each auction draw, a single matrix multiply computes tau(e_i) for every
    grid point and every advertiser at once, then top-k selection is done via a
    cumsum mask. This uses all CPU cores through numpy's internal BLAS threading
    without pickling overhead.

    Parameters
    ----------
    advertisers : list of lists of (v, e) tuples
        Normalized advertiser pairs, one list per simulated auction.
    polynomial_degree : int
        Degree of the tau polynomial (1 = affine, 2 = quadratic, 3 = cubic).
    k : int
        Number of allocation slots per auction.
    num_points : int
        Grid points per coefficient dimension. Grid size = num_points^(degree+1).
    ext_scaler : float
        Weight for the externality welfare component. Pass R_e / R_v to preserve
        the original (v + e) objective scale across independent normalization.

    Returns
    -------
    best_coeffs : list of float
        Coefficients of the best tau, in normalized space.
    best_welfare : float
        Average collateralized welfare of the best tau.
    all_results : list of (coeffs, welfare) tuples
        All (coeffs, welfare) pairs, sorted descending by welfare.
    """
    beta0_range = np.linspace(-2, 2, num_points)
    higher_range = np.linspace(-5, 5, num_points)
    ranges = [beta0_range] + [higher_range] * polynomial_degree

    grid = np.array(list(itertools.product(*ranges)))  # (G, degree+1)
    G = len(grid)
    print(f"Grid search: {polynomial_degree+1} coefficients × {num_points} points = {G:,} combinations")

    start_time = time.time()
    total_welfare = np.zeros(G)
    for advertiser_set in advertisers:
        total_welfare += _eval_grid_on_auction(grid, advertiser_set, k, ext_scaler)
    welfares = total_welfare / len(advertisers)
    elapsed = time.time() - start_time
    print(f"Grid search complete in {elapsed:.1f}s")

    order = np.argsort(-welfares)
    all_results = [(grid[i].tolist(), float(welfares[i])) for i in order]
    best_idx = int(order[0])
    best_coeffs: list[float] = [float(x) for x in grid[best_idx]]
    best_welfare: float = float(welfares[best_idx])
    return best_coeffs, best_welfare, all_results


def ad_distribution(data, data_n, num_advertisers, rng):
    sampled_indices = rng.choice(data.index, size=num_advertisers, replace=False)
    sampled_data   = data.loc[sampled_indices]
    sampled_data_n = data_n.loc[sampled_indices]
    advertisers   = [(row['v'], row['e']) for _, row in sampled_data.iterrows()]
    advertisers_n = [(row['v'], row['e']) for _, row in sampled_data_n.iterrows()]
    return (advertisers, advertisers_n)


def print_stats(auction_output):
    iw = auction_output['individual_welfares']
    print('\n========================================')
    print('Welfare Averages')
    print(f"Avg. VCG Advertiser Welfare:          {np.mean(iw['vcg_ad']):.4f}")
    print(f"Avg. Collateralized Advertiser Welfare: {np.mean(iw['coll_ad']):.4f}")
    print(f"Avg. VCG Externality Welfare:          {np.mean(iw['vcg_ex']):.4f}")
    print(f"Avg. Collateralized Externality Welfare: {np.mean(iw['coll_ex']):.4f}")
    vcg_tot  = [ad+ex for ad, ex in zip(iw['vcg_ad'],  iw['vcg_ex'])]
    coll_tot = [ad+ex for ad, ex in zip(iw['coll_ad'], iw['coll_ex'])]
    print(f"\nAvg. VCG Total Welfare:               {np.mean(vcg_tot):.4f}")
    print(f"Avg. Collateralized Total Welfare:    {np.mean(coll_tot):.4f}")
    print(f"Avg. Change in Total Welfare:         {np.mean(coll_tot) - np.mean(vcg_tot):.4f}")
    print('========================================\n')


def compile_results(externality_cost_per_impression, num_advertisers, num_auctions,
                    random_seed, k, polynomial_degree, auction_output, scaler, advertisers):
    tau_coeffs = scaler.poly_from_normalized(
        auction_output['tau'], x_col='e', y_col='v'
    )
    best_results = run_auction_set(tau_coeffs, advertisers, k)
    _, _, _, best_coll_welfare, best_vcg_welfare, best_individual_welfares = best_results

    w_vcg_ad  = list(best_individual_welfares['vcg_ad'])
    w_coll_ad = list(best_individual_welfares['coll_ad'])
    w_vcg_ex  = list(best_individual_welfares['vcg_ex'])
    w_coll_ex = list(best_individual_welfares['coll_ex'])

    return {
        'externality_cost_per_impression': externality_cost_per_impression,
        'num_advertisers': num_advertisers,
        'num_auctions': num_auctions,
        'random_seed': random_seed,
        'k': k,
        'polynomial_degree': polynomial_degree,
        'tau': tau_coeffs,
        'advertisers': advertisers,
        'w_vcg_adv':  w_vcg_ad,
        'w_coll_adv': w_coll_ad,
        'w_vcg_ext':  w_vcg_ex,
        'w_coll_ext': w_coll_ex,
        'w_vcg_tot':  [ad+ex for ad, ex in zip(w_vcg_ad,  w_vcg_ex)],
        'w_coll_tot': [ad+ex for ad, ex in zip(w_coll_ad, w_coll_ex)],
        'tested_functions': auction_output['tested_functions'],
    }


def main(args):
    data = pd.read_csv(args['data'])

    k                    = args['k']
    externality_cost     = args['externality_cost']
    action_cost          = args['action_cost']
    polynomial_degree    = args['polynomial_degree']
    random_seed          = args['seed']
    num_items            = args['num_items']
    num_auctions         = args['num_auctions']
    num_points           = args['num_points']
    random_number_of_items = args['random_number_of_items']
    random_k             = args['random_k']
    run_id               = args['id']

    data['v'] = data['v_score'] * action_cost
    data['e'] = data['e_score'] * externality_cost

    rng = np.random.default_rng(random_seed)
    if random_number_of_items:
        num_items = rng.integers(1, int(len(data) * 0.1))
    if random_k:
        k = rng.integers(1, int(num_items * 0.1))

    # Normalize v and e independently so each spans [-1, 1] in the search space.
    # joint=True would compress e to near 0 when externality_cost << action_cost,
    # preventing the search from using the externality dimension meaningfully.
    scaler = NegOneOneScaler(joint=False)
    scaler.fit(data[['v', 'e']])
    data_n = scaler.transform(data[['v', 'e']])

    # Weight the normalized e component so the fitness function is equivalent to
    # optimizing the original (v + e) welfare. With independent normalization,
    # tilde_v and tilde_e both span [-1, 1], but original v and e differ in scale
    # by a factor of R_e / R_v, which encodes externality_cost.
    R_v = float(data['v'].max() - data['v'].min())
    R_e = float(data['e'].max() - data['e'].min())
    ext_scaler_n = R_e / R_v

    all_advertisers = [ad_distribution(data, data_n, num_items, rng) for _ in range(num_auctions)]
    advertisers_original, advertisers = zip(*all_advertisers)

    best_coeffs_n, best_welfare, all_results = run_grid_search(
        advertisers=advertisers,
        polynomial_degree=polynomial_degree,
        k=k,
        num_points=num_points,
        ext_scaler=ext_scaler_n,
    )

    polynomial_string = ' + '.join(
        [f'{c:.6f}' if i == 0 else f'{c:.6f}*x^{i}' for i, c in enumerate(best_coeffs_n)]
    )
    print(f"Best tau (normalized): y = {polynomial_string}")
    print(f"Best welfare (normalized): {best_welfare:.6f}")

    best_results = run_auction_set(best_coeffs_n, advertisers, k)
    best_coeffs_n_2, best_avg_coll, best_avg_vcg, best_coll_welfares, best_vcg_welfares, best_iw = best_results

    auction_output = {
        'tau': best_coeffs_n_2,
        'avg_coll_welfare': best_avg_coll,
        'avg_vcg_welfare':  best_avg_vcg,
        'coll_welfare':     best_coll_welfares,
        'vcg_welfare':      best_vcg_welfares,
        'individual_welfares': best_iw,
        'tested_functions': [list(c) for c, _ in all_results],
    }
    print_stats(auction_output)

    result = compile_results(
        externality_cost_per_impression=externality_cost,
        num_advertisers=num_items,
        num_auctions=num_auctions,
        random_seed=random_seed,
        k=k,
        polynomial_degree=polynomial_degree,
        auction_output=auction_output,
        scaler=scaler,
        advertisers=advertisers_original,
    )

    out_path = f'./output/results/gs_results_{run_id}.pkl'
    with open(out_path, 'wb') as f:
        pkl.dump([result], f)
    print(f"Saved results to {out_path}")


def print_usage():
    print("Usage: python Collateralized_Auction_grid_search.py [OPTIONS]")
    print("Options:")
    print("  --k INT                    Required. Number of auction slots.")
    print("  --externality-cost FLOAT   Required. Externality cost parameter.")
    print("  --polynomial-degree INT    Required. Tau polynomial degree (1-3).")
    print("  --data FILENAME            Required. Input CSV file.")
    print("  --action-cost FLOAT        Optional. Defaults to 1.0.")
    print("  --seed INT                 Optional. Random seed.")
    print("  --num-items INT            Optional. Advertisers per auction. Defaults to 20.")
    print("  --num-auctions INT         Optional. Number of auctions. Defaults to 500.")
    print("  --num-points INT           Optional. Grid points per dimension. Defaults to 20.")
    print("  --n-jobs INT               Optional. Parallel workers (-1 = all cores). Defaults to -1.")
    print("  --random-number-of-items   Optional flag.")
    print("  --random-k                 Optional flag.")
    print("  --id STRING                Optional. Run identifier for output filename.")
    print("  --scc-it INT               Optional. Maps to k and externality cost via preset table.")
    print("  --help                     Show this message and exit.")


def parse_args(argv):
    args = {
        'random_number_of_items': False,
        'random_k': False,
        'action_cost': 1.0,
        'num_items': 20,
        'num_auctions': 500,
        'num_points': 20,
        'n_jobs': -1,
        'seed': None,
    }

    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == '--help':
            print_usage()
            sys.exit(0)
        elif arg == '--k':
            i += 1; args['k'] = int(argv[i])
        elif arg == '--externality-cost':
            i += 1; args['externality_cost'] = float(argv[i])
        elif arg == '--action-cost':
            i += 1; args['action_cost'] = float(argv[i])
        elif arg == '--polynomial-degree':
            i += 1; args['polynomial_degree'] = int(argv[i])
        elif arg == '--seed':
            i += 1; args['seed'] = int(argv[i])
        elif arg == '--data':
            i += 1; args['data'] = argv[i]
        elif arg == '--num-items':
            i += 1; args['num_items'] = int(argv[i])
        elif arg == '--num-auctions':
            i += 1; args['num_auctions'] = int(argv[i])
        elif arg == '--num-points':
            i += 1; args['num_points'] = int(argv[i])
        elif arg == '--n-jobs':
            i += 1; args['n_jobs'] = int(argv[i])
        elif arg == '--random-number-of-items':
            args['random_number_of_items'] = True
        elif arg == '--random-k':
            args['random_k'] = True
        elif arg == '--id':
            i += 1; args['id'] = str(argv[i])
        elif arg == '--scc-it':
            i += 1
            scc_it = int(argv[i])
            k_variations = [1]
            ext_variations = [0.001, 0.005, 0.0075, 0.01, 0.025, 0.05, 0.1]
            args['k'] = k_variations[scc_it % len(k_variations)]
            args['externality_cost'] = ext_variations[(scc_it // len(k_variations)) % len(ext_variations)]
            print(f"scc-it {scc_it}: k={args['k']}, externality_cost={args['externality_cost']}")
        else:
            print(f"Unknown argument: {arg}")
            print_usage()
            sys.exit(1)
        i += 1

    if 'id' not in args:
        args['id'] = str(np.random.randint(1, 100000000))

    required = ['k', 'externality_cost', 'polynomial_degree', 'data']
    missing = [key for key in required if key not in args]
    if missing:
        print(f"Missing required arguments: {', '.join('--' + m.replace('_', '-') for m in missing)}")
        print_usage()
        sys.exit(1)

    return args


if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    main(args)
