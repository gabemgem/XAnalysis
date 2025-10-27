#%%
import random
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
import pygad
import time
import pickle as pkl
import sys


def main(args):
    #%%
    data = pd.read_csv(args['data'])

    k = args['k']
    externality_cost = args['externality_cost']
    action_cost = args['action_cost']
    polynomial_degree = args['polynomial_degree']
    random_seed = args['seed']
    num_generations = args['num_generations']
    num_items = args['num_items']
    num_auctions = args['num_auctions']
    random_number_of_items = args['random_number_of_items']
    random_k = args['random_k']
    id = args['id']


    data['v'] = data['v_score'] * action_cost
    data['e'] = data['e_score'] * externality_cost

    rng = np.random.default_rng(random_seed)
    if random_number_of_items:
        num_items = rng.integers(1, int(len(data) * 0.1))
    if random_k:
        k = rng.integers(1, int(num_items*0.1))

    advertisers = [ad_distribution(data, num_items, rng) for _ in range(num_auctions)]

    initial_genes = None

    gene_space = [
        {'low': -15, 'high': 15},
        {'low': -5, 'high': 5},
        {'low': -0.001, 'high': 0.001},
        {'low': -0.001, 'high': 0.001},
    ]
    gene_space = gene_space[:polynomial_degree + 1]

    ga_instance, auction_output = run_ga(
        data=data,
        advertisers=advertisers,
        externality_cost_per_impression=externality_cost,
        num_advertisers=num_items,
        num_auctions=num_auctions,
        random_seed=random_seed,
        gene_space=gene_space,
        k=k,
        polynomial_degree=polynomial_degree,
        initial_genes=initial_genes,
        num_generations=num_generations,
    )
    best_solution, best_fitness, _ = ga_instance.best_solution()
    ga_instance.plot_fitness()

    result = compile_results(
        externality_cost_per_impression=externality_cost,
        num_advertisers=num_items,
        num_auctions=num_auctions,
        random_seed=random_seed,
        k=k,
        polynomial_degree=polynomial_degree,
        auction_output=auction_output)
    results = [result]
    with open(f'./output/ga_results_{id}.pkl', 'wb') as f:
        pkl.dump(results, f)

def print_usage():
    print("Usage: python script.py [OPTIONS]")
    print("Options:")
    print("  --k INT                    Required. Integer value.")
    print("  --externality-cost FLOAT   Required. Floating point value.")
    print("  --polynomial-degree INT    Required. Integer value.")
    print("  --data FILENAME            Required. Input file name.")
    print("  --action-cost FLOAT        Optional. Floating point value. Defaults to 1.0.")
    print("  --seed INT                 Optional. Integer value.")
    print("  --num-generations INT      Optional. Defaults to 600.")
    print("  --num-items INT            Optional. Integer value. Defaults to 20.")
    print("  --num-auctions INT         Optional. Integer value. Defaults to 500.")
    print("  --random-number-of-items   Optional flag.")
    print("  --random-k                 Optional flag.")
    print("  --id STRING                Optional run ID.")
    print("  --scc-it INT               Optional. Integer value for cluster iterations. Will overwrite k and externality cost.")
    print("  --help                     Show this message and exit.")

def parse_args(argv):
    args = {
        'random_number_of_items': False,
        'random_k': False,
        'num_generations': 600,
        'action_cost': 1.0,
        'num_items': 20,
        'num_auctions': 500,
        'seed': None,
    }

    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == '--help':
            print_usage()
            sys.exit(0)
        elif arg == '--k':
            i += 1
            args['k'] = int(argv[i])
        elif arg == '--externality-cost':
            i += 1
            args['externality_cost'] = float(argv[i])
        elif arg == '--action-cost':
            i += 1
            args['action_cost'] = float(argv[i])
        elif arg == '--polynomial-degree':
            i += 1
            args['polynomial_degree'] = int(argv[i])
        elif arg == '--seed':
            i += 1
            args['seed'] = int(argv[i])
        elif arg == '--data':
            i += 1
            args['data'] = argv[i]
        elif arg == '--num-generations':
            i += 1
            args['num_generations'] = int(argv[i])
        elif arg == '--num-items':
            i += 1
            args['num_items'] = int(argv[i])
        elif arg == '--num-auctions':
            i += 1
            args['num_auctions'] = int(argv[i])
        elif arg == '--random-number-of-items':
            args['random_number_of_items'] = True
        elif arg == '--random-k':
            args['random_k'] = True
        elif arg == '--id':
            i += 1
            args['id'] = str(argv[i])
        elif arg == '--scc-it':
            i += 1
            scc_it = int(argv[i])
            k_variations = [1, 2, 5]
            externality_cost_variations = [0.001, 0.005, 0.0075, 0.01, 0.025, 0.05, 0.1]
            args['k'] = k_variations[scc_it % len(k_variations)]
            args['externality_cost'] = externality_cost_variations[(scc_it // len(k_variations)) % len(externality_cost_variations)]
            print(f"Setting k to {args['k']} and externality cost to {args['externality_cost']} based on scc-it {scc_it}")

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

# def plot_auctions(auction_output, ad_scaler=1, ext_scaler=1):
#     counterfactual_welfare = auction_output['vcg_welfare']
#     collateralized_welfare = auction_output['coll_welfare']
#
#     individual_welfares = auction_output['individual_welfares']
#     w_vcg_ad = [v*ad_scaler for v in individual_welfares['vcg_ad']]
#     w_coll_ad = [v*ad_scaler for v in individual_welfares['coll_ad']]
#     w_vcg_ex = [e*ext_scaler for e in individual_welfares['vcg_ex']]
#     w_coll_ex = [e*ext_scaler for e in individual_welfares['coll_ex']]
#
#     # plot 3 welfare histograms in one plot
#     fig = plt.figure(figsize=(8,7))
#     gs = gridspec.GridSpec(2,2,figure=fig)
#
#     # plot individual advertiser and externality welfares
#     ax1 = fig.add_subplot(gs[0,0])
#     ax1.hist([w_vcg_ad, w_coll_ad], bins=30, alpha=0.5, color=['red', 'blue'],
#                 label=['Counterfactual', 'Collateralized'])
#     ax1.legend(loc='upper right')
#     ax1.set_xlabel('Advertiser Welfare')
#     ax1.set_ylabel('Frequency')
#     ax1.set_title('Advertiser Welfare Distribution (ζ=0.01)')
#
#     ax2 = fig.add_subplot(gs[0,1])
#     ax2.hist([w_vcg_ex, w_coll_ex], bins=30, alpha=0.5, color=['red', 'blue'],
#                 label=['Counterfactual', 'Collateralized'])
#     ax2.legend(loc='upper right')
#     ax2.set_xlabel('Externality Welfare')
#     ax2.set_ylabel('Frequency')
#     ax2.set_title('Externality Welfare Distribution (ζ=0.01)')
#     ax2.tick_params(axis='x', rotation=45)
#
#     ax3 = fig.add_subplot(gs[1,:])
#     ax3.hist([counterfactual_welfare, collateralized_welfare], bins=30, alpha=0.5, color=['red', 'blue'], label=['Counterfactual', 'Collateralized'])
#     ax3.legend(loc='upper right')
#     ax3.set_xlabel('Total Welfare')
#     ax3.set_ylabel('Frequency')
#     ax3.set_title('Total Welfare Distribution (ζ=0.01)')
#     plt.tight_layout()
#     plt.show()
#
#
# def plot_advertisers(auction_output, ad_scaler=1, ext_scaler=1):
#     # plot all advertisers and best line
#     advertisers = auction_output['advertisers']
#     tau_coeffs = auction_output['tau']
#     v = [ad[0]*ad_scaler for advertisers_set in advertisers for ad in advertisers_set]
#     e = [ad[1]*ext_scaler for advertisers_set in advertisers for ad in advertisers_set]
#     min_e, max_e = min(e)/3, max(e)
#     division_size = int((max_e - min_e) / 100)
#     tau_x = [min_e + i*division_size for i in range(100)]
#     tau_y = [tau(t, tau_coeffs) for t in tau_x]
#     # tau_min = sum([tau_coeffs[i]*(min_e**i) for i in range(len(tau_coeffs))])
#     # tau_max = sum([tau_coeffs[i]*(max_e**i) for i in range(len(tau_coeffs))])
#
#     # %matplotlib notebook
#     fig, axs = plt.subplots(1, 1, figsize=(10, 5))
#
#     axs.scatter(e, v, alpha=0.5)
#     axs.plot(tau_x, tau_y, color='red')
#     axs.set_ylabel('Advertiser Value')
#     axs.set_xlabel('Externality Value')
#     axs.set_title('Social Welfare Values vs Advertiser Values with Best Tau Line')
#
#     # set x and y limits
#     axs.set_xlim(min_e, max_e)
#     axs.set_ylim(0, max(v)*1.1)
#     fig.show()
    
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
        'tested_functions': auction_output['tested_functions'],
    }
    return result
    

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
           initial_genes=None,
           num_generations=600
           ):
    externality_cost = externality_cost_per_impression
    tested_functions = []


    def run_auction_set_fitness(ga_instance, tau_coeffs, solution_idx):
        w_coll = []
        # convert tau_coeffs from ln to exp
        coeffs = ln_coeffs_to_coeffs(tau_coeffs)

        tested_functions.append(coeffs)
            
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
        num_generations=num_generations,
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

    
    auction_output = {
            'advertisers': advertisers,
            'tau': best_tau_coeffs_2,
            'avg_coll_welfare': best_avg_coll_welfare,
            'avg_vcg_welfare': best_avg_vcg_welfare,
            'coll_welfare': best_coll_welfare,
            'vcg_welfare': best_vcg_welfare,
            'individual_welfares': best_individual_welfares,
            'tested_functions': tested_functions,
        }
    print_stats(auction_output)
    # plot_auctions(auction_output)
    # plot_advertisers(auction_output)
    return ga_instance, auction_output

if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    main(args)



