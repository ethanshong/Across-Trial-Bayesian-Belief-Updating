import pandas as pd
from src import dataframe_filter
from src import graphing

days = pd.read_csv("experiments/Oddball_data_exported_20251013.csv")
trials = pd.read_csv("experiments/Oddball_data_exported_trials_20251013.csv")

days = days.loc[days['rat_name'].isin(['Green1', 'LP1', 'GP4', 'Purple1'])]

days["date"] = pd.to_datetime(days["date"].astype(str), format="%Y%m%d", errors="raise")

RANGES = {
    "Green1":  [("2024-06-13", "2024-08-06"), ("2025-07-01", "2025-07-15")],
    "LP1":     [("2023-04-04", "2023-05-23"), ("2023-09-08", "2023-09-25")],
    "GP4":     [("2023-04-08", "2023-05-23"), ("2023-09-23", "2023-10-05")],
    "Purple1": [("2024-04-17", "2025-06-12"), ("2025-08-19", "2025-08-29")],
}

keep = False

for rat, ranges in RANGES.items():
    rat_mask = (days["rat_name"] == rat)
    in_any_range = False
    for start, end in ranges:
        in_any_range = in_any_range | days["date"].between(start, end, inclusive="both")
    keep = keep | (rat_mask & in_any_range)

days = days.loc[keep].copy()

trials = trials.loc[trials["UUID"].isin(days["UUID"].unique())].copy()

days.to_csv("data/example/Oddball_data_exported.csv", index=False)
trials.to_csv("data/example/Oddball_data_exported_trials.csv", index=False)
