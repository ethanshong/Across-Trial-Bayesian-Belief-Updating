## Requirements

Behavioral data was sorted into a single-trial and day averaged data csv. During analysis, the two dataframes are merged together using a common UUID that is hashed on the date and rat. The properties are displayed below:

```bash
print(days.columns)
print(trials.columns)
```

```bash
Index(['date', 'rat_name', 'rat_ID', 'DOB', 'Sex', 'weight', 'Genotype',
       'HL_date', 'invalid', 'file_name', 'experiment', 'phase', 'task',
       'detail', 'stim_type', 'analysis_type', 'complete_block_count',
       'trial_count', 'hit_percent', 'FA_percent', 'UUID', 'block_size',
       'line', 'genotype'],
      dtype='object')
Index(['Time_since_file_start_(s)', 'Stim_ID', 'Trial_type',
       'Attempts_to_complete', 'Response', 'Reaction_(s)', 'Type',
       'Stim Source', 'Freq (kHz)', 'Inten (dB)', 'Dur (ms)',
       'Nose Out TL (s)', 'Time Out (s)', 'Train Setting',
       'Extra Check Point (s)', 'Delay (s)', 'Trial_number', 'Block_number',
       'complete_block_number', 'UUID'],
      dtype='object')
```

The minimal columns needed for each dataframe though are shown below:
```bash
days = days[['genotype', 'rat_name', 'date', 'file_name', 'task', 'UUID']]
trials = trials[['Stim_ID', 'Trial_type', 'UUID', 'Trial_number', 'Response', 'Reaction_(s)']]
```

Notable properties include:

'file_name' : includes information such as frequency and varying task design

'task' : global phase of ['Base case', 'Catch trials', 'Probe trials']

'Stim_ID' : location of true deviant for the trial

'Trial_type' : type of trial of ['HM', 'FC'], where HM is hit/miss and FC is false-alarm/correct-rejection

'Response' : Rats response of ['Hit', 'Miss', 'FA', 'CR']

'Reaction_(s)' : Reaction time relative to the start of the trial

'Trial_number' : 1-indexed, restarts per day


This folder contains partial behavioral data generated to demonstrate the analysis pipeline.

The data does not correspond to complete or continuous behavioral data and is **purely** for modeling purposes