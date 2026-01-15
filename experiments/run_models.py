from pathlib import Path
import sys
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import dataframe_filter
from src import helper

"""
Given proper trial and day dataframes, generates the internal beliefs over time svg, optimized 
hyperparameters, and H5 files containing all internal beliefs for the specified models and rats in dataframes.

Parameters
----------
model_names_ : list
    CHANGE THIS to include any of the following ["Biased", "Unbiased", "Changepoint" or "Omniscient"]
days : DataFrame
    Requirements specified in data/README.md
trials : DataFrame
    Requirements specified in data/README.md

Returns
-------
None
    Outputs to "outputs/figures"
"""
def run_models(model_names_, trials_, days_):
    trials = trials_
    days = days_
    model_names = model_names_

    # resets all figures/data in "outputs/figures"
    helper.clear_dir(Path("outputs/figures"))

    for rat in days['rat_name'].unique():
        days_trials = dataframe_filter.load_dataframe(trials, days, rat)
        functions = helper.model_functions(model_names)
        for function in functions:
            function(days_trials)


### Change parameters here
trials = pd.read_csv("data/example/Oddball_data_exported_trials.csv")
days = pd.read_csv("data/example/Oddball_data_exported.csv")
# If you add the changepoint keep in mind it takes a long time to run even given the small example dataset
model_names = ['Unbiased', 'Omniscient', 'Biased']

run_models(model_names, trials, days)