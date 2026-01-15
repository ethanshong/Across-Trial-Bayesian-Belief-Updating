import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

"""
Find trials of continuous phase and record the start and end trial numbers

Parameters
----------
df : Dataframe
    must contain 'Global' column
phase : String
    must be of 'Baseline', 'Catch trials', or 'Probe trials'

Returns
-------
out : DataFrame
    Returns dataframe that specifies phase start and end rows/trial_number as well as length
"""
def find_intervals_by_index(df, phase):
    # Obtain dataframe during given phase
    mask = df['Global'].astype(str).str.strip().str.casefold().eq(phase.casefold())

    # New run whenever mask flips (contiguous by row index)
    run_id = mask.ne(mask.shift()).cumsum()

    out = (
        df.loc[mask]
        .assign(
            row=lambda d: d.index,      
            run_id=run_id[mask].values 
        )
        .groupby("run_id", as_index=False)
        .agg(
            start_row=("row", "min"),
            end_row=("row", "max"),
            n_rows=("row", "size"),
            start_trial=("Trial_number", "first"),
            end_trial=("Trial_number", "last"),
        )
    )

    out.insert(0, "label", phase)
    return out



"""
Find intervals for probe and catch trials to be later graphed

Parameters
----------
df_outer : Dataframe

Returns
-------
probe_intervals : DataFrame
catch_intervals: DataFrame
    Returns dataframe that specifies phase start and end rows/trial_number as well as length for both phases
"""
def phases(df_outer):
    df_outer = df_outer.sort_values(['Date','Trial_number']).reset_index(drop=True)

    probe_intervals = find_intervals_by_index(df_outer, 'Probe trials')
    catch_intervals = find_intervals_by_index(df_outer, 'Catch trials')

    return probe_intervals, catch_intervals

"""
Plots the internal belief over trial length

Parameters
----------
belief_over_time : list
    shape -> (Trials, 4)
probe_intervals : DataFrame
catch_intervals : DataFrame
figsize : Any
savepath : String

Returns
-------
None
    Produces a matplotlib figure with the highlighted catch phases and demarcated probe starts
"""
def plot_belief_evolution(belief_over_time,
                        probe_intervals=None,
                        catch_intervals=None,
                        figsize=(10, 6),
                        savepath="outputs/figures",
                        name="default"):
    
    out_dir = Path(savepath)
    out_dir.mkdir(parents=True, exist_ok=True)

    B = np.asarray(belief_over_time) 
    T = B.shape[0]

    fig, ax = plt.subplots(figsize=figsize)

    # Positions 4,5,6 (zero-indexed)
    for i in range(3): 
        ax.plot(B[:, i], label=f"Position {i+4}", linewidth=1.5)

    # "No deviant" (7) as dashed line
    ax.plot(B[:, 3], label="No deviant", linestyle='--', color='black', linewidth=1.5)

    # Probe trials
    if probe_intervals is not None and len(probe_intervals) > 0:
        added_probe_label = False
        for _, r in probe_intervals.iterrows():
            x = int(r['start_row'])
            ax.axvline(x=x, color='tab:red', linestyle='--', linewidth=1.2, alpha=0.9,
                    label=None if added_probe_label else "Probe start")
            added_probe_label = True

    # Catch trials
    if catch_intervals is not None and len(catch_intervals) > 0:
        added_catch_span_label = False
        added_catch_edge_label = False
        for _, r in catch_intervals.iterrows():
            xs, xe = int(r['start_row']), int(r['end_row'])
            ax.axvspan(xs, xe, color='tab:blue', alpha=0.08,
                    label=None if added_catch_span_label else "Catch interval")
            added_catch_span_label = True
            ax.axvline(x=xs, color='tab:blue', linestyle='-', linewidth=0.8, alpha=0.6,
                    label=None if added_catch_edge_label else "Catch boundary")
            ax.axvline(x=xe, color='tab:blue', linestyle='-', linewidth=0.8, alpha=0.6,
                    label=None)
            added_catch_edge_label = True

    ax.set_xlim(0, T-1)
    ax.set_xlabel("Trial")
    ax.set_ylabel("Belief")
    ax.set_title("Evolution of Belief Over Time")

    # legend outside on the right
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    leg = ax.legend(
        by_label.values(), by_label.keys(),
        loc='center left', bbox_to_anchor=(1.02, 0.5),
        title='Legend',
        frameon=True, fancybox=True,
        framealpha=1.0,
        edgecolor='0.3',
        ncol=1, fontsize=9, title_fontsize=10,
        borderpad=0.3, labelspacing=0.3, handlelength=1.2, handletextpad=0.5
    )
    leg.get_frame().set_linewidth(1.0)

    out_file = out_dir / f"{name}.svg"
    plt.tight_layout()
    plt.savefig(out_file, format="svg", bbox_inches="tight")
    plt.close()