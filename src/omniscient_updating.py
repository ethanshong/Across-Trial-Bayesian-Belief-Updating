import numpy as np
import math
import csv
from src import decision_making
from src import graphing
import h5py
from pathlib import Path
from src import change_point_updating


"""
Computes the entire sequence of biased internal belief updates for a single run

Parameters
----------
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
def omniscient_model(alpha1_3, alpha4, alpha5, alpha6, detect4, detect5, detect6, df):
    
    M = decision_making.build_M(alpha1_3, alpha4, alpha5, alpha6, detect4, detect5, detect6)
    
    trial_by_global = np.array([
    # HM FC
    #standard        
    [1, 0],
    #probe
    [0.6, 0.4],
    #catch
    [0.96, 0.04]
    ]).T

    trial_logic = np.array([
    #HM
    [0.33, 0.33, 0.33, 0],
    #FC
    [0, 0, 0, 1]
    ]).T

    FUTURE_GIVEN_STATE = trial_logic @ trial_by_global

    deviant_given_sequence, global_prior, belief_over_time, decision_history = change_point_updating.initialize_priors_and_history(None, None, None)
    
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

            deviant_given_sequence = change_point_updating.CP_internal_update(choices, global_prior, trial_by_global, trial_logic, FUTURE_GIVEN_STATE)
            belief_over_time.append(deviant_given_sequence)
    
    return decision_history, np.asarray(belief_over_time)

"""
Unlike the other models, the omniscient mode has no hyperparameters to optimize. Computes the log 
likelihood of a decision made for every internal belief over time. Note these are OBSERVED desired states.

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
def omniscient(df):
    data = df['Decision'].to_numpy()
    deviant_data = df['Position'].to_numpy()

    decision_history, belief_over_time = omniscient_model(*decision_making.behavior_metrics(df), df)
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
    nll *= -1

    graphing.plot_belief_evolution(belief_over_time, *graphing.phases(df), 
                                   savepath="outputs/figures", name="omniscient "+df['Rat_name'].unique()[0])

    # Flatten the dict into a single row
    row = {
        "rat_name": df['Rat_name'].unique()[0],
        "genotype": df['Genotype'].unique()[0],
        "target": nll/(df['Trial_number'].count())
    }

    csv_file = Path("outputs/figures/Omniscient_runs.csv")
    csv_file.parent.mkdir(parents=True, exist_ok=True)

    file_exists = csv_file.exists()

    with csv_file.open(mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)

    out_dir = Path("outputs/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    h5_path = out_dir / "omniscient.h5"  

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