import numpy as np
from bayes_opt import BayesianOptimization
import math
import csv
from src import decision_making
from src import graphing
import h5py
from pathlib import Path

"""
Intialize internal priors & beliefs to uniform or empty

Parameters
----------
belief_over_time_: NDArray
    shape -> (trials, 4)
decision_history_ : NDArray
    shape -> (trials, 7)

Returns
-------
deviant_given_sequence : NDArray
    shape -> (1, 4)
belief_over_time_: NDArray
    shape -> (trials, 4)
decision_history_ : NDArray
    shape -> (trials, 7)
"""
def initialize_priors(belief_over_time_, decision_history_):
        if (belief_over_time_ is None or decision_history_ is None):
            belief_over_time = []  
            decision_history = []         
        else:
            belief_over_time = belief_over_time_  
            decision_history = decision_history_

        deviant_given_sequence = np.array([1/4, 1/4, 1/4, 1/4])      


        return deviant_given_sequence, belief_over_time, decision_history

"""
Exponentially weighs the most recent observations and decays unobserved states.

Parameters
----------
prior_belief : NDArray
    shape -> (1, 4)
choice : int
alpha : float
    controls the amount of exponential weight over new observations. Larger -> more weight on
    recent observations.

Returns
-------
updated_prior : NDArray
    shape -> (1, 4)
"""
def unbiased_internal_update(prior_belief, choice, alpha):    
    updated_prior = prior_belief.copy()

    if(choice == None or choice <= 2):
        return prior_belief

    for pos in range(3, 7): 
        if (choice == pos):
            Ct = 1
            updated_prior[pos - 3] = alpha * Ct + (1 - alpha) * updated_prior[pos - 3]
        if (choice != pos):
            Ct = 0
            updated_prior[pos - 3] = alpha * Ct + (1 - alpha) * updated_prior[pos - 3]


    return updated_prior

"""
Computes the entire sequence of unbiased internal belief updates for a single run

Parameters
----------
alpha : float
alpha1_3 : float
alpha4 : float
alpha5 : float
alpha6 : float
detect4 : float
detect5 : float
detect6 : float
df : DataFrame
    Must be updated and filtered to a single rat following the use of dataframe_filter.load_dataframe()

Returns
-------
decision_history : list
belief_over_time : NDArray
    shape -> (1, 4)
"""
def exponential_model(alpha, alpha1_3, alpha4, alpha5, alpha6, detect4, detect5, detect6, df):

    M = decision_making.build_M(alpha1_3, alpha4, alpha5, alpha6, detect4, detect5, detect6)
    
    deviant_given_sequence, belief_over_time, decision_history = initialize_priors(None, None)

    dates = df['Date'].unique()

    for date in dates:
        df_copy = df.copy()
        df_copy = df_copy.loc[df_copy['Date'] == date]  

        for trial in range(len(df_copy['Trial_number'])):
            choice = (df_copy.iloc[trial]['Decision'])

            decision = decision_making.decision_prior(deviant_given_sequence, M)
            
            decision_history.append(decision)

            deviant_given_sequence = unbiased_internal_update(deviant_given_sequence, choice, alpha)
            belief_over_time.append(deviant_given_sequence.copy())
    
    return decision_history, np.asarray(belief_over_time)

"""
The loss function that is to be used by the bayesian optimizer in unbiased_optimized. Computes the log 
likelihood of a decision made for every internal belief over time. Note these are OBSERVED desired states.

Parameters
----------
alpha : float
df : DataFrame
    Must be updated and filtered to a single rat following the use of dataframe_filter.load_dataframe()

Returns
-------
-nll : float
"""
def neg_log_likelihood(alpha, df):
    data = df['Decision'].to_numpy()
    deviant_data = df['Position'].to_numpy()

    decision_history, belief_over_time = exponential_model(alpha, *decision_making.behavior_metrics(df), df)
    decision_history = np.asarray(decision_history, dtype=float)

    if len(decision_history) != len(data):
        raise ValueError(f"Length mismatch: decision_history={len(decision_history)} vs data={len(data)}")

    nll = 0.0
    for i in range(len(decision_history)):
        k = int(data[i])           
        if k == -2:
            t = int(deviant_data[i])   
            p = decision_history[i, 0:t].sum()   
        elif (k == 7):
            continue
        else:
            # normal case: probability of the chosen bin
            p = decision_history[i, k]     
            if (p <= 0):
                print(decision_history[i, :])             
        nll -= math.log(p)                
    return -nll

"""
Optimizes the hyperparameters (alpha) for the loss function (neg_log_likelihood) using 10 random
points and 20 iterations of bayesian optimization. You can change the seed, iterations, and bounds to obtain
different results.

Parameters
----------
df : DataFrame
    Must be updated and filtered to a single rat following the use of dataframe_filter.load_dataframe()

Returns
-------
None
    Produces entries to a clean CSV and H5 file of the best run's parameters, loss, and belief_over_time.
    Also graphs the belief_over_time into an SVG located in the figures folder.
"""
def unbiased_optimized(df):
    pbounds = {
        'alpha': (0.001, 0.01),
    }

    def work_around(alpha):
        return neg_log_likelihood(alpha, df)

    # Initialize the optimizer
    optimizer = BayesianOptimization(
        f=work_around,
        pbounds=pbounds,
        random_state=6,
        verbose=2
    )

    # Run optimization
    optimizer.maximize(
        init_points=10,  
        n_iter=20    
    )    


    result = optimizer.max

    best_run = result['params']
    decision_history, belief_over_time = exponential_model(best_run['alpha'], *decision_making.behavior_metrics(df), df)

    graphing.plot_belief_evolution(belief_over_time, *graphing.phases(df), 
                                   savepath="outputs/figures", name="unbiased "+df['Rat_name'].unique()[0])

    # Flatten the dict into a single row
    row = {
        "rat_name": df['Rat_name'].unique()[0],
        "genotype": df['Genotype'].unique()[0],
        "target": result["target"]/(df['Trial_number'].count()),
        **result["params"]
    }

    csv_file = Path("outputs/figures/Unbiased_runs.csv")
    csv_file.parent.mkdir(parents=True, exist_ok=True)

    file_exists = csv_file.exists()

    with csv_file.open(mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)

    out_dir = Path("outputs/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    h5_path = out_dir / "unbiased.h5"  

    rat_id = df['Rat_name'].unique()[0]  # whatever your rat name/id is

    with h5py.File(h5_path, "a") as f:
        grp = f.require_group(f"rats/{rat_id}") 

        # overwrite safely if rerun
        if "belief_over_time" in grp:
            del grp["belief_over_time"]

        grp.create_dataset(
            "belief_over_time",
            data=belief_over_time,
            compression="gzip",
            compression_opts=4
        )