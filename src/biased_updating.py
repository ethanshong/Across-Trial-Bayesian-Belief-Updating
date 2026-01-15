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
dirichlet_prior_belief_ : NDArray
    shape -> (1, 4)
weighted_counts_ : NDArray
    shape -> (1, 4). Not a probability vector.
belief_over_time_ : NDArray
    shape -> (trials, 4)
decision_history_ : NDArray
    shape -> (trials, 7)

Returns
-------
dirichlet_prior_belief_ : NDArray
    shape -> (1, 4)
weighted_counts_ : NDArray
    shape -> (1, 4). Not a probability vector.
belief_over_time_ : NDArray
    shape -> (trials, 4)
decision_history_ : NDArray
    shape -> (trials, 7)
"""
def biased_initialize_priors(dirichlet_prior_belief_, weighted_counts_, belief_over_time_, decision_history_):

    if (dirichlet_prior_belief_ is None or weighted_counts_ is None or belief_over_time_ is None or decision_history_ is None):
        dirichlet_prior_belief = np.array([1/4, 1/4, 1/4, 1/4])  
        weighted_counts = np.array([0.0, 0.0, 0.0, 0.0])
        belief_over_time = []  
        decision_history = []         
    else:
        dirichlet_prior_belief = dirichlet_prior_belief_
        weighted_counts = weighted_counts_
        belief_over_time = belief_over_time_  
        decision_history = decision_history_ 

    return dirichlet_prior_belief, weighted_counts, belief_over_time, decision_history

"""
Internal belief updating for the biased model using exponentially weighted pseudo-counts
and the previous prior

Parameters
----------
alpha : float
    exponential weight where larger favors more recent observations
decay : float
    affects how uniform the prior stays after updating and larger decays prevent exploding values
weighted_counts : NDArray
    shape -> (1, 4). Pseudo-count of each observed desired state over trials
dirichlet_prior_belief : NDArray
    previous prior belief leading to bayesian updating
choice : int
    decision the rat made

Returns
-------
updated_dirichlet_prior : NDArray
    returns a posterior belief based on the evidence (current decision) provided
"""
def biased_internal_update(alpha, decay, weighted_counts, dirichlet_prior_belief, choice):    
    if (choice == None or choice <= 2):
        return dirichlet_prior_belief
    for pos in range(3, 7): 
        if (choice == pos):
            Ct = 1
        else :
            Ct = 0
        
        weighted_counts[pos - 3] = alpha * Ct + (1 - alpha) * weighted_counts[pos - 3]
    
    numerators = (weighted_counts + dirichlet_prior_belief) - decay
    numerators = np.maximum(numerators, 1e-12)  # avoid negatives/zeros
    updated_dirichlet_prior = numerators / numerators.sum()

    return updated_dirichlet_prior

"""
Computes the entire sequence of biased internal belief updates for a single run

Parameters
----------
alpha : float
decay : float
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
def exponential_model(alpha, decay, alpha1_3, alpha4, alpha5, alpha6, detect4, detect5, detect6, df):

    M = decision_making.build_M(alpha1_3, alpha4, alpha5, alpha6, detect4, detect5, detect6)

    #Initial priors whether finished training or starting catch
    dirichlet_prior_belief, weighted_counts, belief_over_time, decision_history = biased_initialize_priors(
            None, None, None, None)

    dates = df['Date'].unique()

    for date in dates:
        df_copy = df.copy()
        df_copy = df_copy.loc[df_copy['Date'] == date]  

        for trial in range(len(df_copy['Trial_number'])):
            choice = (df_copy.iloc[trial]['Decision'])

            decision = decision_making.decision_prior(dirichlet_prior_belief, M)
            
            decision_history.append(decision)

            dirichlet_prior_belief = biased_internal_update(alpha, decay, weighted_counts, dirichlet_prior_belief, choice)
            dirichlet_prior_belief = decision_making.normalize(dirichlet_prior_belief)
            belief_over_time.append(dirichlet_prior_belief.copy())

    return decision_history, np.asarray(belief_over_time)

"""
The loss function that is to be used by the bayesian optimizer in biased_optimized. Computes the log 
likelihood of a decision made for every internal belief over time. Note these are OBSERVED desired states.

Parameters
----------
alpha : float
decay : float
df : DataFrame
    Must be updated and filtered to a single rat following the use of dataframe_filter.load_dataframe()

Returns
-------
-nll : float
"""
def neg_log_likelihood(alpha, decay, df):
    data = df['Decision'].to_numpy()
    deviant_data = df['Position'].to_numpy()

    alpha1_3, alpha4, alpha5, alpha6, detect4, detect5, detect6 = decision_making.behavior_metrics(df)

    #All post observation priors (decision prior) & cast into array
    priors, belief_over_time = exponential_model(alpha, decay, alpha1_3, alpha4, alpha5, alpha6, detect4, detect5, detect6, df)
    priors = np.asarray(priors, dtype=float)

    # Check for dataframe and model length mismatch
    if len(priors) != len(data):
        raise ValueError(f"Length mismatch: priors={len(priors)} vs data={len(data)}")

    nll = 0.0
    for i in range(len(priors)):
        k = int(data[i])           
        if k == -2:
            t = int(deviant_data[i])   
            p = priors[i, 0:t].sum()   
        elif (k == 7):
            continue
        else:
            # normal case: probability of the chosen bin
            p = priors[i, k]     
            if (p <= 0):
                print(priors[i, :])             
        nll -= math.log(p)               
    return -nll

"""
Optimizes the hyperparameters (alpha, decay) for the loss function (neg_log_likelihood) using 10 random
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
def biased_optimized(df):
    #bounds on free parameters
    pbounds = {
        'alpha': (0.00001, 0.1),
        'decay': (0.000001, 0.1)
    }

    def work_around(alpha, decay):
        return neg_log_likelihood(alpha, decay, df)

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
    decision_history, belief_over_time = exponential_model(best_run['alpha'], 
        best_run['decay'], *decision_making.behavior_metrics(df), df)

    graphing.plot_belief_evolution(belief_over_time, *graphing.phases(df), 
                                   savepath="outputs/figures", name="biased "+df['Rat_name'].unique()[0])

    # Flatten the dict into a single row
    row = {
        "rat_name": df['Rat_name'].unique()[0],
        "genotype": df['Genotype'].unique()[0],
        "target": result["target"]/(df['Trial_number'].count()),
        **result["params"]
    }

    csv_file = Path("outputs/figures/Biased_runs.csv")
    csv_file.parent.mkdir(parents=True, exist_ok=True)

    file_exists = csv_file.exists()

    with csv_file.open(mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)
    
    out_dir = Path("outputs/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    h5_path = out_dir / "biased.h5"  

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