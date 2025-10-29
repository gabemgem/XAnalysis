#%%
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm
from joblib import Parallel, delayed
import pygad
import time
import pickle as pkl
#%% md
# # Collateralized Ad Auction
# 
# An ad auction is an auction where advertisers are bidding for a user's attention through an ad distributor (Google, Facebook, etc.). The advertiser pays the distributor some amount of money for their ad to show up on the user's browser.
# 
# Ads carry with them some value for the advertiser and, crucially, some value for the general social welfare. The social welfare value often comes through the ad's intended effect and truthfulness. For example, an ad that denies climate change would have a negative social welfare value, while one that encourages using clean energy might have a positive social welfare value.
# 
# Let's assume the auctioneer knows the true social welfare value for any ad. A penalized ad auction is one in which the advertiser who wins the auction pays an additional cost as a function of the ad's social welfare value. (this can equivalently be implemented through a collateralization mechanism with probabalistic auditing and loss of the collateral as a penalty).
# 
# This code implements a collateralized second-price ad auction. Advertisers place a bid only if their value is greater than the expected payment value plus the known penalty (or expected collateralization) payment for their ad. Their payment is the second highest bid. This ensures that advertisers bid their true values.
# 
# The advertiser's value is denoted $v$ where $v_1, v_2, ...$ are the highest value, second highest value, etc. The social welfare value is denoted $e$ where $e_1$ is the social welfare value of the ad with the highest advertiser value. Given a collateralization function ($C$) that returns the collateralization value for a given $e$, $\tau$ is a function that returns $C(e) + \mathbb{E}(v_2)$. An advertiser will only bid if their $v_i > \tau(v_i, e_i)$.
# 
# ## Data
# 
# X data was collected from the X community notes files as well as from the X API. Posts were pulled from the API if they had a community note with at least 400 "HELPFUL" ratings. Post externality cost per 1000 impressions (CPM) was calculated by giving note types values between -3 and 3, and then subtracting the average rating where 'HELPFUL' = 1, 'SOMEWHAT_HELPFUL' = 0, and 'NOT_HELPFUL' = -1.
# 
# $$e = (\text{note score} - \text{rating score}) * \text{impression count per month} * 0.00153$$
# 
# Average advertiser cost per first action data was pulled from [this website](https://www.webfx.com/social-media/pricing/how-much-does-it-cost-to-advertise-on-twitter/).
# 
# Post 'advertiser value' will be calculated during a simulation using this formula:
# $$v = \text{random cost per action} * \frac{retweets+replies+likes+quotes+bookmarks}{impressions/1000}$$
# 
# 
# 
# 
#%%
tweet_data = pd.read_csv('full_tweets.csv')

print(tweet_data[['externality_cpm', 'action_count_per_1000_impressions']].describe())

average_ad_costs = [0.13,0.38,0.63,0.88,1.255,1.755,2.255,2.755,3.255]
# the 'not applicable' 2.5% was devided and distributed evenly to the other
# categories to create a pmf that sums to 1
ad_cost_probabilities = [0.1527777778,
                         0.1777777778,
                         0.1527777778,
                         0.1277777778,
                         0.09447777778,
                         0.08607777778,
                         0.06947777778,
                         0.08607777778,
                         0.05277777778]
# example usage: rng.choice(average_ad_costs, num_draws, p=ad_cost_probabilities) 
#%%
tweet_data['created_at_dt'] = pd.to_datetime(tweet_data['created_at'])
date_pulled = pd.to_datetime('02/02/2025')
tweet_data['days_old'] = (date_pulled - tweet_data['created_at_dt']).dt.days
tweet_data['months_old'] = tweet_data['days_old']/30

# plot histogram of impressions per month
plt.hist(tweet_data['impression_count']/(tweet_data['days_old']/30), bins=30, log=True)
plt.xlabel('Impressions per Month')
plt.ylabel('Frequency')
plt.title('Impressions per Month Histogram')
plt.show()

tweet_data['impressions_per_month'] = tweet_data['impression_count']/(tweet_data['months_old'])

print(tweet_data['impressions_per_month'].describe())
#%%
# user per month
externality_cost_per_impression = 0.00153

externality_max = tweet_data['externality'].abs().max()
tweet_data['ext_dollars_per_month'] = ((tweet_data['externality']/externality_max) * 
                                       tweet_data['impressions_per_month'] * 
                                       externality_cost_per_impression)
tweet_data['ext_per_month'] = ((tweet_data['externality']/externality_max) * 
                                       tweet_data['impressions_per_month'])

# producer actions per month
tweet_data['action_per_month'] = tweet_data['action_count'] / tweet_data['months_old']

print(tweet_data[['ext_dollars_per_month', 'action_per_month']].describe())

# plot histograms of externality and action per month
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].hist(tweet_data['ext_dollars_per_month'], bins=30, log=True)
axs[0].set_xlabel('Externality Dollars per Month')
axs[0].set_ylabel('Frequency')
axs[0].set_title('Externality Dollars per Month Histogram')

axs[1].hist(tweet_data['action_per_month'], bins=30, log=True)
axs[1].set_xlabel('Action Count per Month')
axs[1].set_ylabel('Frequency')
axs[1].set_title('Action Count per Month Histogram')

fig.show()
#%% md
# ## Parameters
# 
# - `num_advertisers`: The number of advertisers participating in the auction
# - `tau`: A function that returns the expected payment given an ad
# - `ad_distribution`: A function that initializes $(v_i, e_i)$ pairs according to a given distribution
# - `value_ratio`: The ratio of the advertiser value to the social welfare value
# - `k`: number of allocation spots per auction (in results look at addition of e and v values for all winners)
#%%
random_seed = 123
random.seed(random_seed)
rng = np.random.default_rng(random_seed)

num_auctions = 100
num_advertisers = 100
tau = lambda v, e: -0.5*e + 0.5
k = 1

def ad_distribution(data, num_advertisers, rng):
    sample = data.sample(n=num_advertisers, random_state=rng)
    advertisers = [(row['v'], row['e']) for index, row in sample.iterrows()]
    # v = rng.choice(average_ad_costs, 1, p=ad_cost_probabilities) * tweet_data['action_count_per_1000_impressions'].iloc[random_i]
    # e = tweet_data['externality_cpm'][random_i]
    return advertisers
#%% md
# ## Auction
# 1. Initialize advertiser ad values according to parameters
# 2. Run counterfactual second-price auction
#  - Advertisers decide on bid ($v$, honest)
#  - Everyone bids
#  - Record $(v_1, e_1, v_1+e_1) = w_{vcg}$
# 3. Run collateralized second-price auction
#  - Advertisers decide on bid ($v$, honest)
#  - Advertisers with $v_i > \tau(v_i, e_i)$ place bid
#  - Record $(v_1, e_1, v_1+e_1) = w_{coll}$
# 
#  Make change to allow for any number of accepted bids (k). If no advertisers pass the tau check, output welfare of 0. Output v-vector that is v1, v1+v2, v1+...+vk etc. Visualize:
#  - histogram of welfare for given k
#  - line (x=k, y=avg welfare for k)
#%%
def auction_objective(w_results, ad_scaler=1, ext_scaler=1):
    return (w_results[0]*ad_scaler)+(w_results[1]*ext_scaler)

def tau(e, coeffs):
    tau_val = 0
    for i, coeff in enumerate(coeffs):
        tau_val += coeff * (e**i)
    return tau_val

def run_auction(advertisers, tau_coeffs, k=1):

    # run counterfactual auction
    counterfactual_bids = []
    for ad in advertisers:
        counterfactual_bids.append(ad)
    ## sort bids according to advertiser value
    counterfactual_bids.sort(key=lambda x: x[0])
    ## get the top k bids
    top_bids = counterfactual_bids[-k:]
    w_vcg = (sum([b[0] for b in top_bids]), sum([b[1] for b in top_bids]))

    # run collateralized auction
    collateralized_bids = []
    for ad in advertisers:
        if ad[0] >= tau(ad[1], tau_coeffs):
            collateralized_bids.append(ad)
    ## sort bids according to advertiser value
    collateralized_bids.sort(key=lambda x: x[0])
    top_bids = collateralized_bids[-k:]
    w_coll = (sum([b[0] for b in top_bids]), sum([b[1] for b in top_bids]))

    return w_vcg, w_coll

def run_auction_set(tau_coeffs, advertisers, k=1):
    w_vcg = []
    w_coll = []
        
    for advertiser_set in advertisers:
            w_vcg_i, w_coll_i = run_auction(advertiser_set, tau_coeffs, k)
            w_vcg.append(w_vcg_i)
            w_coll.append(w_coll_i)
            
    w_vcg_objective = [auction_objective(w) for w in w_vcg]
    w_coll_objective = [auction_objective(w) for w in w_coll]
    
    w_vcg_ad = [w[0] for w in w_vcg]
    w_coll_ad = [w[0] for w in w_coll]
    w_vcg_ex = [w[1] for w in w_vcg]
    w_coll_ex = [w[1] for w in w_coll]
    individual_welfares = {
        'vcg_ad': w_vcg_ad,
        'coll_ad': w_coll_ad,
        'vcg_ex': w_vcg_ex,
        'coll_ex': w_coll_ex
    }
    
    avg_vcg_objective = np.mean(w_vcg_objective)
    avg_coll_objective = np.mean(w_coll_objective)
    
    return (tau_coeffs, avg_coll_objective, avg_vcg_objective, w_coll_objective, w_vcg_objective, individual_welfares)

# def run_auctions(num_auctions, 
#                  num_advertisers, 
#                  slope_magnitude_max, 
#                  slope_step_size,
#                  intercept_min,
#                  intercept_max,
#                  intercept_step_size,
#                  data, 
#                  k=1, 
#                  rng=None):
#     # output variables
#     w_vcg = []
#     w_coll = []
#     advertisers = [ad_distribution(data, num_advertisers, rng) for _ in range(num_auctions)]
#     
#     grid = [
#         (m, b) 
#         for m in np.arange(-slope_magnitude_max, 0, slope_step_size) 
#         for b in np.arange(intercept_min, intercept_max, intercept_step_size)
#     ]
#     
#     results = Parallel(n_jobs=-1)(
#         delayed(run_auction_set)(m, b, advertisers, k) 
#         for (m, b) in tqdm(grid, desc='Auction Simulations')
#     )
#     # results[i] = (m, b, avg_coll_objective, avg_vcg_objective, w_coll_objective, w_vcg_objective)
#     best_result = max(results, key=lambda x: x[2])
#     best_m, best_b, best_avg_coll_welfare, best_avg_vcg_welfare, best_coll_welfare, best_vcg_welfare, best_individual_welfares = best_result
#     best_line = (best_m, best_b)
#             
#     print('========================================')
#     print('Finished Running Simulations')
#     print(f'Best Tau: ad_value = {best_line[0]}*social_welfare + {best_line[1]}')
#     print(f'Avg Collateralized Objective: {best_avg_coll_welfare}')
#     # print(f'Avg VCG Welfare: {best_avg_vcg_welfare}')
#     # print(f'\nChange in Total Welfare: {best_avg_coll_welfare - best_avg_vcg_welfare}')
#     print('========================================')
# 
#     return {
#         'advertisers': advertisers,
#         'tau': best_line,
#         'avg_coll_welfare': best_avg_coll_welfare,
#         'avg_vcg_welfare': best_avg_vcg_welfare,
#         'coll_welfare': best_coll_welfare,
#         'vcg_welfare': best_vcg_welfare,
#         'individual_welfares': best_individual_welfares
#     }

def print_stats(auction_output, ad_scaler=1, ext_scaler=1):
    
    individual_welfares = auction_output['individual_welfares']
    w_vcg_ad = [v*ad_scaler for v in individual_welfares['vcg_ad']]
    w_coll_ad = [v*ad_scaler for v in individual_welfares['coll_ad']]
    w_vcg_ex = [e*ext_scaler for e in individual_welfares['vcg_ex']]
    w_coll_ex = [e*ext_scaler for e in individual_welfares['coll_ex']]
    
    print('\n========================================')
    print('Welfare Averages')
    print(f'Avg. VCG Advertiser Welfare: {np.mean(w_vcg_ad):.2f}')
    print(f'Avg. Collateralized Advertiser Welfare: {np.mean(w_coll_ad):.2f}')
    print(f'Avg. VCG Social Welfare: {np.mean(w_vcg_ex):.2f}')
    print(f'Avg. Collateralized Social Welfare: {np.mean(w_coll_ex):.2f}')
    print(f'\nAvg. Change in Advertiser Welfare: {np.mean(w_coll_ad) - np.mean(w_vcg_ad):.2f}')
    print(f'Avg. Change in Social Welfare: {np.mean(w_coll_ex) - np.mean(w_vcg_ex):.2f}')
    print(f'\nAvg. VCG Total Welfare: {np.mean([ad+ex for ad, ex in zip(w_vcg_ad, w_vcg_ex)]):.2f}')
    print(f'Avg. Collateralized Total Welfare: {np.mean([ad+ex for ad, ex in zip(w_coll_ad, w_coll_ex)]):.2f}')
    print(f'Avg. Change in Total Welfare: {np.mean([ad+ex for ad, ex in zip(w_coll_ad, w_coll_ex)]) - np.mean([ad+ex for ad, ex in zip(w_vcg_ad, w_vcg_ex)]):.2f}')
    print('========================================\n')

def plot_auctions(auction_output, ad_scaler=1, ext_scaler=1):
    counterfactual_welfare = auction_output['vcg_welfare']
    collateralized_welfare = auction_output['coll_welfare']
    
    individual_welfares = auction_output['individual_welfares']
    w_vcg_ad = [v*ad_scaler for v in individual_welfares['vcg_ad']]
    w_coll_ad = [v*ad_scaler for v in individual_welfares['coll_ad']]
    w_vcg_ex = [e*ext_scaler for e in individual_welfares['vcg_ex']]
    w_coll_ex = [e*ext_scaler for e in individual_welfares['coll_ex']]
    
    # plot 3 welfare histograms in one plot
    fig = plt.figure(figsize=(8,7))
    gs = gridspec.GridSpec(2,2,figure=fig)

    # plot individual advertiser and externality welfares
    ax1 = fig.add_subplot(gs[0,0])
    ax1.hist([w_vcg_ad, w_coll_ad], bins=30, alpha=0.5, color=['red', 'blue'],
                label=['Counterfactual', 'Collateralized'])
    ax1.legend(loc='upper right')
    ax1.set_xlabel('Advertiser Welfare')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Advertiser Welfare Distribution (ζ=0.01)')

    ax2 = fig.add_subplot(gs[0,1])
    ax2.hist([w_vcg_ex, w_coll_ex], bins=30, alpha=0.5, color=['red', 'blue'],
                label=['Counterfactual', 'Collateralized'])
    ax2.legend(loc='upper right')
    ax2.set_xlabel('Externality Welfare')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Externality Welfare Distribution (ζ=0.01)')
    ax2.tick_params(axis='x', rotation=45)

    ax3 = fig.add_subplot(gs[1,:])
    ax3.hist([counterfactual_welfare, collateralized_welfare], bins=30, alpha=0.5, color=['red', 'blue'], label=['Counterfactual', 'Collateralized'])
    ax3.legend(loc='upper right')
    ax3.set_xlabel('Total Welfare')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Total Welfare Distribution (ζ=0.01)')
    plt.tight_layout()
    plt.show()
    
    
def plot_advertisers(auction_output, ad_scaler=1, ext_scaler=1):
    # plot all advertisers and best line
    advertisers = auction_output['advertisers']
    tau_coeffs = auction_output['tau']
    v = [ad[0]*ad_scaler for advertisers_set in advertisers for ad in advertisers_set]
    e = [ad[1]*ext_scaler for advertisers_set in advertisers for ad in advertisers_set]
    min_e, max_e = min(e)/3, max(e)
    division_size = int((max_e - min_e) / 100)
    tau_x = [min_e + i*division_size for i in range(100)]
    tau_y = [tau(t, tau_coeffs) for t in tau_x]
    # tau_min = sum([tau_coeffs[i]*(min_e**i) for i in range(len(tau_coeffs))])
    # tau_max = sum([tau_coeffs[i]*(max_e**i) for i in range(len(tau_coeffs))])
    
    # %matplotlib notebook
    fig, axs = plt.subplots(1, 1, figsize=(10, 5))
    
    axs.scatter(e, v, alpha=0.5)
    axs.plot(tau_x, tau_y, color='red')
    axs.set_ylabel('Advertiser Value')
    axs.set_xlabel('Externality Value')
    axs.set_title('Social Welfare Values vs Advertiser Values with Best Tau Line')
    
    # set x and y limits
    axs.set_xlim(min_e, max_e)
    axs.set_ylim(0, max(v)*1.1)
    fig.show()
    
def compile_results(externality_cost_per_impression, 
                    num_advertisers, 
                    num_auctions, 
                    random_seed, 
                    k, 
                    polynomial_degree, 
                    auction_output):
    individual_welfares = auction_output['individual_welfares']
    w_vcg_ad = [v for v in individual_welfares['vcg_ad']]
    w_coll_ad = [v for v in individual_welfares['coll_ad']]
    w_vcg_ex = [e for e in individual_welfares['vcg_ex']]
    w_coll_ex = [e for e in individual_welfares['coll_ex']]

    result = {
        'externality_cost_per_impression': externality_cost_per_impression,
        'num_advertisers': num_advertisers,
        'num_auctions': num_auctions,
        'random_seed': random_seed,
        'k': k,
        'polynomial_degree': polynomial_degree,
        'tau': auction_output['tau'],
        'advertisers': auction_output['advertisers'],
        'w_vcg_adv': np.mean(w_vcg_ad),
        'w_coll_adv': np.mean(w_coll_ad),
        'w_vcg_ext': np.mean(w_vcg_ex),
        'w_coll_ext': np.mean(w_coll_ex),
        'w_vcg_tot': np.mean([ad+ex for ad, ex in zip(w_vcg_ad, w_vcg_ex)]),
        'w_coll_tot': np.mean([ad+ex for ad, ex in zip(w_coll_ad, w_coll_ex)]),
    }
    return result
    
#%%
# num_auctions = 500
# num_advertisers = 50
# slope_magnitude_max = 15
# slope_step_size = 0.1
# intercept_min = -700
# intercept_max = -500
# intercept_step_size = 10
# k=1
# 
# advertiser_scaler = tweet_data['action_per_month'].abs().max()
# externality_scaler = tweet_data['ext_dollars_per_month'].abs().max()
# 
# tweet_data['v'] = tweet_data['action_per_month']# / advertiser_scaler
# tweet_data['e'] = tweet_data['ext_dollars_per_month']# / externality_scaler
# 
# random_seed = 1234
# rng = np.random.default_rng(random_seed)
# 
# auction_output = run_auctions(
#     num_auctions, 
#     num_advertisers, 
#     slope_magnitude_max, 
#     slope_step_size,
#     intercept_min,
#     intercept_max,
#     intercept_step_size,
#     data=tweet_data, 
#     k=k, 
#     rng=rng)
# print_stats(auction_output)#, advertiser_scaler, externality_scaler)
# plot_auctions(auction_output)#, advertiser_scaler, externality_scaler)
# plot_advertisers(auction_output)#, advertiser_scaler, externality_scaler)
#%%
last_fitness = 0
def on_generation(ga_instance):
    global last_fitness
    # print(f"Generation = {ga_instance.generations_completed}")
    # print(f"Fitness    = {ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]}")
    # print(f"Change     = {ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1] - last_fitness}")
    print(f"Solution {ga_instance.generations_completed} : {ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[0]} - Fitness : {ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]} - Change : {ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1] - last_fitness}")
    last_fitness = ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]
    
def ln_coeffs_to_coeffs(ln_coeffs):
    coeffs = [np.sign(c)*(np.exp(abs(c))-1) for c in ln_coeffs]
    return coeffs
    

def run_ga(data,
           advertisers,
           externality_cost_per_impression,
           num_advertisers,
           num_auctions,
           random_seed,
           gene_space=None,
           k=1,
           polynomial_degree=1,
           initial_genes=None
           ):
    externality_cost = externality_cost_per_impression

    

    
    def run_auction_set_fitness(ga_instance, tau_coeffs, solution_idx):
        w_coll = []
        # convert tau_coeffs from ln to exp
        coeffs = ln_coeffs_to_coeffs(tau_coeffs)
            
        for advertiser_set in advertisers:
                _, w_coll_i = run_auction(advertiser_set, coeffs, k)
                w_coll.append(w_coll_i)
                
        w_coll_objective = [auction_objective(w) for w in w_coll]
        
        avg_coll_objective = np.mean(w_coll_objective)
        
        return avg_coll_objective

    initial_population = None
    if initial_genes is not None:
        initial_population = [[initial_genes[i]+random.uniform(-0.0001, 0.0001) for i in range(len(initial_genes))] for _ in range(19)]
        initial_population.append(initial_genes)

    ga_instance = pygad.GA(
        num_generations=600 if polynomial_degree == 1 else 5000,
        initial_population=initial_population,
        num_parents_mating=10,
        fitness_func=run_auction_set_fitness,
        sol_per_pop=20,
        num_genes=polynomial_degree+1,
        gene_space=gene_space,
        parent_selection_type='tournament',
        mutation_type="random",
        mutation_probability=1.0,#(0.5, 0.25),
        # mutation_by_replacement=False,
        # random_mutation_max_val=100,
        # random_mutation_min_val=-100,
        crossover_type='single_point',
        on_generation=None,
        random_seed=random_seed,
        parallel_processing=10
    )

    # save start time
    start_time = time.time()
    ga_instance.run()
    # save end time
    end_time = time.time()

    print("Time taken to run GA:", end_time - start_time)
    
    # ga_instance.plot_fitness()
    best_solution, best_fitness, _ = ga_instance.best_solution()
    best_tau_coeffs_ln = best_solution
    best_tau_coeffs = ln_coeffs_to_coeffs(best_tau_coeffs_ln)
    polynomial_string = ' + '.join([f'{c:.6f}*x^{i}' if i != 0 else f'{c:.6f}' for i, c in enumerate(best_tau_coeffs)])
    print(f"Best line found: y = {polynomial_string}")
    # print("Fitness:", best_fitness)
    
    best_results = run_auction_set(best_tau_coeffs, advertisers, k)
    best_tau_coeffs_2, best_avg_coll_welfare, best_avg_vcg_welfare, best_coll_welfare, best_vcg_welfare, best_individual_welfares = best_results
    
    # print(best_tau_coeffs)
    # print(best_tau_coeffs_2)
    
    auction_output = {
            'advertisers': advertisers,
            'tau': best_tau_coeffs_2,
            'avg_coll_welfare': best_avg_coll_welfare,
            'avg_vcg_welfare': best_avg_vcg_welfare,
            'coll_welfare': best_coll_welfare,
            'vcg_welfare': best_vcg_welfare,
            'individual_welfares': best_individual_welfares
        }
    print_stats(auction_output)
    plot_auctions(auction_output)
    plot_advertisers(auction_output)
    return ga_instance, auction_output
   
   
# tweet_data['v'] = tweet_data['action_per_month']# / advertiser_scaler
# tweet_data['e'] = tweet_data['ext_per_month']# / externality_scaler
# externality_cost_per_impression = 0.01
# num_advertisers = 20
# num_auctions = 500
# random_seed = None
# k=1
# polynomial_degree=1
# gene_space = [
#         { 'low': 0, 'high': 15 },
#         { 'low': 0, 'high': 5 },
#         { 'low': 0, 'high': 0.001 },
#         { 'low': 0, 'high': 0.001 },
#     ]
# gene_space = gene_space[:polynomial_degree+1]
#
# ga_instance, auction_output = run_ga(tweet_data, externality_cost_per_impression, num_advertisers, num_auctions, random_seed, gene_space, k, polynomial_degree)


#%%
# ga_instance.plot_fitness()
#%%
externality_cost_variations = [0.01]#[0.001, 0.005, 0.0075, 0.01, 0.025, 0.05, 0.1]
k_variations = [1]#, 2, 5]
polynomial_variations = [1, 2, 3]
results = []


num_advertisers = 20
num_auctions = 500

for k in k_variations:
    for externality_cost_per_impression in externality_cost_variations:
        tweet_data['v'] = tweet_data['action_per_month']
        tweet_data['e'] = tweet_data['ext_per_month'] * externality_cost_per_impression

        random_seed = 39207276#random.randint(1, 123456789)

        rng = np.random.default_rng(random_seed)

        advertisers = [ad_distribution(tweet_data, num_advertisers, rng) for _ in range(num_auctions)]
        # advertisers = [[(row['v'], row['e']) for index, row in tweet_data.iterrows()]]

        initial_genes = None
        for polynomial_degree in polynomial_variations:
            gene_space = [
                {'low': -15, 'high': 15},
                {'low': -5, 'high': 5},
                {'low': -0.001, 'high': 0.001},
                {'low': -0.001, 'high': 0.001},
            ]
            gene_space = gene_space[:polynomial_degree + 1]


            ga_instance, auction_output = run_ga(
                data=tweet_data,
                advertisers=advertisers,
                externality_cost_per_impression=externality_cost_per_impression,
                num_advertisers=num_advertisers,
                num_auctions=num_auctions,
                random_seed=random_seed,
                gene_space=gene_space,
                k=k,
                polynomial_degree=polynomial_degree,
                initial_genes = initial_genes,
            )
            best_solution, best_fitness, _ = ga_instance.best_solution()
            ga_instance.plot_fitness()
            initial_genes = [best_solution[i] if i<len(best_solution) else 0.0 for i in range(polynomial_degree+2)]

            result = compile_results(
                externality_cost_per_impression=externality_cost_per_impression,
                num_advertisers=num_advertisers,
                num_auctions=num_auctions,
                random_seed=random_seed,
                k=k,
                polynomial_degree=polynomial_degree,
                auction_output=auction_output)
            results.append(result)
            with open('ga_results.pkl', 'wb') as f:
                pkl.dump(results, f)
#%%
# print(results)
#%%
# with open('ga_results2.pkl', 'rb') as f:
#     results = pkl.load(f)
#
# print(len(results))
# print(results[0])
#
# print([r['polynomial_degree'] for r in results])
# print(tweet_data.shape)