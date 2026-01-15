import numpy as np
import pandas as pd

"""
Normalize an array

Parameters
----------
x : NDArray

Returns
-------
x / s : NDArray
    if array is full of zeros or sums to zero returns an array full of zeros
"""
def normalize(x):
    x = np.asarray(x, float)
    s = x.sum()
    return x / s if s > 0 else np.zeros_like(x)

"""
Generates P(choice d = 1..7 | s = deviant_position) for a specific s

Parameters
----------
alpha1_3 : float
alpha4 : float
alpha5: float
alpha6: float
detect4: float
detect5: float
detect6: float
deviant_position : int

Returns
-------
out : NDArray
    The hazard-based array that describes the likelihood of the rat making the decision at this given position
    for the actual s.
"""
def likelihood_prior(alpha1_3, alpha4, alpha5, alpha6, detect4, detect5, detect6, deviant_position):
    no = (1.0 - alpha1_3)

    if deviant_position == 4:
        s = alpha1_3 + no*alpha1_3 + no**2*alpha1_3 + no**3*detect4
        out = np.array([
            alpha1_3,
            no * alpha1_3,
            no**2 * alpha1_3,
            no**3 * detect4,
            (1 - s) / 3,
            (1 - s) / 3,
            (1 - s) / 3
        ], dtype=float)
        return normalize(out)

    if deviant_position == 5:
        s = (alpha1_3
             + no*alpha1_3
             + no**2*alpha1_3
             + no**3*alpha4
             + no**3*(1 - alpha4)*detect5)
        out = np.array([
            alpha1_3,
            no * alpha1_3,
            no**2 * alpha1_3,
            no**3 * alpha4,
            no**3 * (1 - alpha4) * detect5,
            (1 - s) / 2,
            (1 - s) / 2
        ], dtype=float)
        return normalize(out)

    if deviant_position == 6:
        s = (alpha1_3
             + no*alpha1_3
             + no**2*alpha1_3
             + no**3*alpha4
             + no**3*(1 - alpha4)*alpha5
             + no**3*(1 - alpha4)*(1 - alpha5)*detect6)
        out = np.array([
            alpha1_3,
            no * alpha1_3,
            no**2 * alpha1_3,
            no**3 * alpha4,
            no**3 * (1 - alpha4) * alpha5,
            no**3 * (1 - alpha4) * (1 - alpha5) * detect6,
            (1 - s)
        ], dtype=float)
        return normalize(out)

    # deviant_position == 7 (no deviant)
    s = (alpha1_3
         + no*alpha1_3
         + no**2*alpha1_3
         + no**3*alpha4
         + no**3*(1 - alpha4)*alpha5
         + no**3*(1 - alpha4)*(1 - alpha5)*alpha6)
    out = np.array([
        alpha1_3,
        no * alpha1_3,
        no**2 * alpha1_3,
        no**3 * alpha4,
        no**3 * (1 - alpha4) * alpha5,
        no**3 * (1 - alpha4) * (1 - alpha5) * alpha6,
        (1 - s)
    ], dtype=float)
    return normalize(out)

"""
Iterates over all true states, s, to generate the complete decision matrix
    Returns M (7x4) with columns:
    col 0 = P(choice | K=4)
    col 1 = P(choice | K=5)
    col 2 = P(choice | K=6)
    col 3 = P(choice | K=7)

Parameters
----------
alpha1_3 : float
alpha4 : float
alpha5: float
alpha6: float
detect4: float
detect5: float
detect6: float

Returns
-------
M : NDArray 
    shape -> (7, 4)
"""
def build_M(alpha1_3, alpha4, alpha5, alpha6, detect4, detect5, detect6):
    cols = []
    for K in (4, 5, 6, 7):
        col = likelihood_prior(alpha1_3, alpha4, alpha5, alpha6,
                               detect4, detect5, detect6, K)
        cols.append(col.reshape(-1, 1))
    M = np.concatenate(cols, axis=1)
    return M

"""
Multiplies the decision matrix by the current internal prior to generate P(d | o_(1:t))

Parameters
----------
prior : NDArray 
    shape -> (1, 4)
M : NDArray
    shape -> (7, 4)

Returns
-------
M @ prior : NDArray
    shape -> (1, 7)
"""
def decision_prior(prior, M):
    return M @ prior

"""
Acquires the hit, miss, false alarm, and correct rejection rates averaged across all phases, 
trial logics, and true states, s.

Parameters
----------
df : DataFrame

Returns
-------
alpha1_3 : float
alpha4 : float
alpha5 : float
alpha6 : float
detect4 : float
detect5 : float
detect6 : float
"""
def behavior_metrics(df):
    m_matrix = df[['Position', 'Trial_logic', 'Response', 'Decision']].copy()

    m_matrix['Decision'] = m_matrix['Decision'].astype(int)

    m_matrix['FA_rate'] = (m_matrix['Response'] == 'FA').astype(int)
    m_matrix['Hit_rate'] = (m_matrix['Response'] == 'Hit').astype(int)


    series = (m_matrix.groupby(['Decision', 'Position'])[['FA_rate', 'Hit_rate']].sum().reset_index()).astype(int)
    FA_df = series.pivot_table(index = 'Decision', columns='Position', values='FA_rate')
    Hit_df = series.pivot_table(index = 'Decision', columns='Position', values='Hit_rate')

    response_df = m_matrix.groupby(['Position'])['Response'].count()

    FA_df = FA_df / response_df
    Hit_df = Hit_df / response_df

    FA_df = FA_df.replace(float(0), pd.NA)
    Hit_df = Hit_df.replace(float(0), pd.NA)
    
    #Some rats don't contain position 7 data
    FA_df = FA_df.drop([-2, 6, 7], errors = 'ignore').apply(pd.to_numeric, errors='coerce')
    Hit_df = Hit_df.drop([-2, 6, 7], errors = 'ignore').apply(pd.to_numeric, errors='coerce')

    FA_rate = FA_df.mean(axis=1, numeric_only=True)

    alpha1_3 = FA_rate[0:2].mean()
    alpha4 = FA_rate[3]
    alpha5 = FA_rate[4]
    alpha6 = FA_rate[5]

    detect4 = Hit_df.at[3, 3]
    detect5 = Hit_df.at[4, 4]
    detect6 = Hit_df.at[5, 5]

    return alpha1_3, alpha4, alpha5, alpha6, detect4, detect5, detect6