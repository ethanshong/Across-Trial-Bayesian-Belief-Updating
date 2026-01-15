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
global_prior : NDArray
    shape -> (1, 3). Prior for the global phase. Remains static throughout all trials.
belief_over_time_ : NDArray
    shape -> (trials, 4)
decision_history_ : list

Returns
-------
deviant_given_sequence : NDArray
    shape -> (1, 4). This is the same as the dirichlet_prior_belief in unbiased or other true state priors.
global_prior : NDArray
    shape -> (1, 3)
belief_over_time_ : NDArray
    shape -> (trials, 4)
decision_history_ : list
    all decisions for the unknown current phase
"""
def initialize_priors_and_history(global_prior = None, belief_over_time = None, decision_history = None):
        if (belief_over_time is None or decision_history is None):          
            global_prior = np.array([1/3, 1/3, 1/3])
            belief_over_time = [] 
            decision_history = []

        else:
            global_prior = global_prior
            belief_over_time = belief_over_time  
            decision_history = decision_history

        deviant_given_sequence = np.array([1/7, 1/7, 1/7, 1/7])      

        return deviant_given_sequence, global_prior, belief_over_time, decision_history

"""
The probability of the sequence of observed trials given the global phase expressed 
as the cumulative product of each individual trial’s observed state likelihood given the global phase.

Parameters
----------
sequence : list
    All decisions for a single block of trials. Resets per block.
trial_by_global : NDArray
    shape -> (3, 2)
trial_logic : NDArray
    shape -> (2, 4)

Returns
-------
deviant_given_state : NDArray
    shape -> (1, 3). Probability vector describing likelihood of each global state.
"""
def sequence_given_state(sequence, trial_by_global, trial_logic):
    deviant_given_state = np.ones(3)
    for i in range(len(sequence)):
        if (sequence[i] <= 2 or sequence[i] == 7):
            continue
        deviant_given_trial = np.array([trial_logic[sequence[i] - 3][0], trial_logic[sequence[i] - 3][1]])
        deviant_given_state *= deviant_given_trial @ trial_by_global
        deviant_given_state = deviant_given_state / deviant_given_state.sum()

    return deviant_given_state

"""
Calculates the likely global phase given the whole sequence of decisions. Bayes update using the global prior.
Note global prior is uniform.

Parameters
----------
sequence : list
state_prob : NDArray
    shape -> (1, 3)
trial_by_global : NDArray
    shape -> (3, 2)
trial_logic : NDArray
    shape -> (2, 4)

Returns
-------
norm_state_given_sequence : NDArray
    shape -> (1, 3)
"""
def state_given_sequence(sequence, state_prob, trial_by_global, trial_logic):
    SEQUENCE_GIVEN_STATE = sequence_given_state(sequence, trial_by_global, trial_logic)
    state_given_sequence = SEQUENCE_GIVEN_STATE * state_prob
    norm_state_given_sequence = state_given_sequence / state_given_sequence.sum()
    return norm_state_given_sequence

"""
Calculates the probability of the next desired observed state given all previous runs. Refer to the method's
section of the related paper for more information.

Parameters
----------
sequence : list
prior : NDArray
    shape -> (1, 3). Refers to the global prior.
trial_by_global : NDArray
    shape -> (3, 2)
trial_logic : NDArray
    shape -> (2, 4)
FUTURE_GIVEN_STATE : NDArray
    shape -> (1, 3). Static global variable that only needs to be calculated once.

Returns
-------
FUTURE_GIVEN_STATE @ STATE_GIVEN_SEQUENCE.T
    shape -> (1, 4)
"""
def future_given_sequence(sequence, prior, trial_by_global, trial_logic, FUTURE_GIVEN_STATE):
    STATE_GIVEN_SEQUENCE = state_given_sequence(sequence, prior, trial_by_global, trial_logic)    
    
    return FUTURE_GIVEN_STATE @ STATE_GIVEN_SEQUENCE.T

"""
Updates the internal prior given a new decision

Parameters
----------
decision_history : list
    sequence and decision_history are interchangeable here.
global_prior : NDArray
    shape -> (1, 3). prior and global_prior are also interchangeable here.
trial_by_global : NDArray
    shape -> (3, 2)
trial_logic : NDArray
    shape -> (2, 4)
FUTURE_GIVEN_STATE : NDArray
    shape -> (1, 3).

Returns
-------
deviant_given_sequence
    shape -> (1, 4)
"""
def CP_internal_update(decision_history, global_prior, trial_by_global, trial_logic, FUTURE_GIVEN_STATE):    

    deviant_given_sequence = future_given_sequence(decision_history, global_prior, trial_by_global, trial_logic, FUTURE_GIVEN_STATE)

    return deviant_given_sequence


"""
Computes the entire sequence of change point internal belief updates for a single run

Parameters
----------
trial_by_global : NDArray
    shape -> (3, 2)
trial_logic : NDArray
    shape -> (2, 4)
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
def change_point_model(trial_by_global, trial_by_logic, alpha1_3, alpha4, alpha5, alpha6, detect4, detect5, detect6, df):

    M = decision_making.build_M(alpha1_3, alpha4, alpha5, alpha6, detect4, detect5, detect6)

    FUTURE_GIVEN_STATE = trial_by_logic @ trial_by_global

    deviant_given_sequence, global_prior, belief_over_time, decision_history = initialize_priors_and_history(None, None, None)
    
    dates = df['Date'].unique()

    for date in dates:
        df_copy = df.copy()
        df_copy = df_copy.loc[df_copy['Date'] == date]  
        deviant_given_sequence = np.array([1/4, 1/4, 1/4, 1/4])

        choices = []

        for trial in range(len(df_copy['Trial_number'])):
            choice = (df_copy.iloc[trial]['Decision'])

            decision = decision_making.decision_prior(deviant_given_sequence, M)

            decision_history.append(decision)
            choices.append(int(choice))

            deviant_given_sequence = CP_internal_update(choices, global_prior, trial_by_global, trial_by_logic, FUTURE_GIVEN_STATE)
            belief_over_time.append(deviant_given_sequence)
    
    return decision_history, np.asarray(belief_over_time)

"""
The loss function that is to be used by the bayesian optimizer in change_point_optimized. Computes the log 
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
def neg_log_likelihood(hm_standard, hm_probe, hm_catch, dev4, df):
    data = df['Decision'].to_numpy()
    deviant_data = df['Position'].to_numpy()

    trial_by_global = np.array([
        [hm_standard, 1 - hm_standard],
        [hm_probe, 1 - hm_probe],
        [hm_catch, 1 - hm_catch]
    ]).T
    trial_by_logic = np.array([
        [dev4, dev4, 1 - (2 * dev4), 0],
        [0, 0, 0, 1]
    ]).T

    decision_history, belief_over_time = change_point_model(trial_by_global, trial_by_logic, *decision_making.behavior_metrics(df), df)
    decision_history = np.asarray(decision_history, dtype=float)

    if len(decision_history) != len(data):
        raise ValueError(f"Length mismatch: decision_history={len(decision_history)} vs data={len(data)}")

    # floor + nan guard
    eps = 1e-12
    decision_history = np.where(np.isfinite(decision_history), decision_history, 0.0)
    decision_history = np.maximum(decision_history, eps)

    nll = 0.0
    for i in range(len(decision_history)):
        k = int(data[i])           
        if k == -2:
            t = int(deviant_data[i])  
            p = decision_history[i, 0:t].sum()   
        elif (k == 7):
            continue
        else:
            p = decision_history[i, k]                  
        nll -= math.log(p)                
    return -nll

"""
Optimizes the hyperparameters (hm_standard, hm_probe, hm_catch, dev4) for the loss function 
(neg_log_likelihood) using 10 random points and 50 iterations of bayesian optimization. You can change the 
seed, iterations, and bounds to obtain different results.

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
def change_point_optimized(df):
    pbounds = {
        'hm_standard': (0.8, 0.99),
        'hm_probe': (0.8, 0.99),
        'hm_catch': (0.45, 0.75),
        'dev4': (0.25, 0.4),
    }

    def work_around(hm_standard, hm_probe, hm_catch, dev4):
        return neg_log_likelihood(hm_standard, hm_probe, hm_catch, dev4, df)

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
        n_iter=50     
    )    

    result = optimizer.max

    best_run = result['params']
    trial_by_global = np.array([
        [best_run['hm_standard'], 1 - best_run['hm_standard']],
        [best_run['hm_probe'], 1 - best_run['hm_probe']],
        [best_run['hm_catch'], 1 - best_run['hm_catch']]
    ]).T
    trial_by_logic = np.array([
        [best_run['dev4'], best_run['dev4'], 1 - (2 * best_run['dev4']), 0],
        [0, 0, 0, 1]
    ]).T
    decision_history, belief_over_time = change_point_model(trial_by_global, trial_by_logic, *decision_making.behavior_metrics(df), df)

    graphing.plot_belief_evolution(belief_over_time, *graphing.phases(df), 
                                   savepath="outputs/figures", name="changepoint "+df['Rat_name'].unique()[0])

    # Flatten the dict into a single row
    row = {
        "rat_name": df['Rat_name'].unique()[0],
        "genotype": df['Genotype'].unique()[0],
        "target": result["target"]/(df['Trial_number'].count()),
        **result["params"]  
    }

    csv_file = Path("outputs/figures/Changepoint_runs.csv")
    csv_file.parent.mkdir(parents=True, exist_ok=True)

    file_exists = csv_file.exists()

    with csv_file.open(mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


    out_dir = Path("outputs/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    h5_path = out_dir / "changepoint.h5"  

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