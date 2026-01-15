import pandas as pd
import numpy as np


"""
Given the dataframe's Position, Decision and trial logic columns determine what decision the rat made

Parameters
----------
df : DataFrame
    includes the specified columns for a given rat: 
    ['Date', 'Trial_number', 'Global', 'Trial_logic', 'Frequency', 'Position', 'Decision', 'Response']
    Position and Decision correspond to Stim_ID and Reaction_(s) in the original Dataframe, where Stim_ID is 
    the actual deviant location and the Reaction_(s) is the rat's RT relative to the
    beginning of a trial used for calculating False-alarm_correct-rejection trials. The spacing between 
    each beep is 0.75 ms.

Returns
-------
df : DataFrame
    Returns the Position of the deviant and the Decision the rat made for each trial. Decision and
    position are 0-indexed.
"""
def decision_bins(df):
    # Masks
    m_HM   = df['Trial_logic'].eq('HM')
    m_FC   = df['Trial_logic'].eq('FC')
    m_Hit  = df['Response'].eq('Hit')
    m_FA   = df['Response'].eq('FA')
    m_Miss = df['Response'].eq('Miss')
    m_CR   = df['Response'].eq('CR')

    # Parse position string to position
    pos_num = pd.to_numeric(
        df['Position'].astype(str).str.extract(r'(\d+)', expand=False),
        errors='coerce'
    )
    
    # Acquire Reaction_(s) column and find the decision bin.
    dec_num = pd.to_numeric(df['Decision'], errors='coerce')
    fa_calc = (np.floor(dec_num / 0.750) + 1).astype('Int64')


    #Handle each trial_logic (HM, FC) and possible decision type (Hit, FA, Miss, CR)
    # HM & Hit  -> Decision = Position
    df.loc[m_HM & m_Hit, 'Decision'] = pos_num[m_HM & m_Hit]

    # HM & FA   -> Decision = (Decision // 0.750) + 1
    df.loc[m_HM & m_FA, 'Decision'] = fa_calc[m_HM & m_FA]

    # HM & Miss -> Decision = -1
    df.loc[m_HM & m_Miss, 'Decision'] = -1

    # FC & CR   -> Decision = 7
    df.loc[m_FC & m_CR, 'Decision'] = 7

    # FC & FA   -> Decision = (Decision // 0.750) + 1
    df.loc[m_FC & m_FA, 'Decision'] = fa_calc[m_FC & m_FA]

    df['Decision'] = df['Decision'] - 1
    df['Position'] = df['Position'] - 1

    return df


"""
Given two dataframes that contain trial specific and day specific information, merge to include repetitive
day information for every trial by a UUID (hash map for single day's worth of trials and unique rat). Must at
least include specified columns:
days[['genotype', 'rat_name', 'date', 'file_name', 'task', 'UUID']]
trials[['Stim_ID', 'Trial_type', 'UUID', 'Trial_number', 'Response', 'Reaction_(s)']]

Additionally, trials and days csvs must be at correct folder level.

Parameters
----------
rat_name : String
    filters dataframe for specified rat

Returns
-------
df : DataFrame
    Returns single filtered dataframe with decision bins the specified rat made across all trials.
"""
def load_dataframe(trials_, days_, rat_name):
    # load in single trial & day specific data
    trials = trials_
    days   = days_ 

    # filter for given rat
    days = days.loc[days['rat_name'] == rat_name]

    # merge single trial data to include all day specific data by UUID (each day/rat has unique UUID)
    days = days[['genotype', 'rat_name', 'date', 'file_name', 'task', 'UUID']]
    trials = trials[['Stim_ID', 'Trial_type', 'UUID', 'Trial_number', 'Response', 'Reaction_(s)']]
    days_trials = trials.merge(days, on='UUID', how='inner')
    days_trials = days_trials[['date', 'Trial_number', 'task', 'Trial_type', 'file_name', 'Stim_ID', 'Reaction_(s)', 'Response', 'genotype', 'rat_name']]
    days_trials.columns = ['Date', 'Trial_number', 'Global', 'Trial_logic', 'Frequency', 'Position', 'Decision', 'Response', 'Genotype', 'Rat_name']

    # update trial_logic names to hit/miss or false-alarm/correct-rejection trial
    # & parse for frequency
    days_trials['Trial_logic'] = days_trials['Trial_logic'].replace({5: 'HM', 0: 'FC'})
    days_trials['Trial_logic'] = pd.Categorical(days_trials['Trial_logic'], categories=['HM','FC'])
    days_trials['Frequency'] = (
        days_trials['Frequency']
        .astype(str)
        .str.extract(r'(?i)(\d+)\s*kHz', expand=False)  # case-insensitive 'kHz'
        .astype(int)
    )

    # filter to include only standard, probe, and catch trials
    days_trials = days_trials.loc[days_trials['Global'].isin(['Base case', 'Probe trials', 'Catch trials'])]

    days_trials = decision_bins(days_trials)

    days_trials = days_trials.sort_values(['Date','Trial_number']).reset_index(drop=True)

    return days_trials