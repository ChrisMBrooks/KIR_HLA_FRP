import numpy as np
import pandas as pd
import time, uuid, sys

import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme()

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from Controllers.DataScienceManager import DataScienceManager as dsm


def get_partitioned_data(
        phenos_subset:str, 
        partition_training_dataset:bool, 
        n_splits:int, n_repeats:int, 
        ith_repeat:int, random_state:int
    ):

    phenos_t = data_sci_mgr.data_mgr.outcomes(fill_na=False, fill_na_value=None, partition='training')
    scores_t = data_sci_mgr.data_mgr.features(fill_na=False, fill_na_value=None, partition='training')
    phenos_t = phenos_t[phenos_subset].copy()

    phenos_v = data_sci_mgr.data_mgr.outcomes(fill_na=False, fill_na_value=None, partition='validation')
    scores_v = data_sci_mgr.data_mgr.features(fill_na=False, fill_na_value=None, partition='validation')
    phenos_v = phenos_v[phenos_subset].copy()

    if partition_training_dataset:
        phenos_t, scores_t, phenos_v, scores_v = data_sci_mgr.data_mgr.partition_training_data(
            phenos_t, scores_t, n_splits, n_repeats, ith_repeat, random_state
        )

    return phenos_t, scores_t, phenos_v, scores_v

def preprocess_for_validation(
        phenos_t:pd.DataFrame, scores_t:pd.DataFrame, 
        phenos_v:pd.DataFrame, scores_v:pd.DataFrame,
        impute, strategy, standardise, normalise,
        dependent_var
    ):
    phenos_t, scores_t = data_sci_mgr.data_mgr.reshape(phenos_t, scores_t, dependent_var)
    phenos_v, scores_v = data_sci_mgr.data_mgr.reshape(phenos_v, scores_v, dependent_var)

    phenos_t, scores_t, phenos_v, scores_v = data_sci_mgr.data_mgr.preprocess_data_v(
        X_t=phenos_t, Y_t=scores_t, X_v=phenos_v, Y_v=scores_v,
        impute = impute, strategy=strategy, standardise = standardise, 
        normalise = normalise
    )

    scores_t = scores_t.ravel()
    scores_v = scores_v.ravel()
    return phenos_t, scores_t, phenos_v, scores_v

def get_final_score(phenos_subset, partition_training_dataset:bool, h_params:dict, 
        validation_approach:str, n_splits:int, n_repeats:int, random_state:int,
        impute:bool, strategy:str, standardise:bool, normalise:bool,
        dependent_var:str
    ):
    
    results = []
    for i in range(1, n_repeats+1):
        ith_repeat = i

        phenos_t, scores_t, phenos_v, scores_v = get_partitioned_data(
            phenos_subset, partition_training_dataset, 
            n_splits, n_repeats, ith_repeat, random_state
        )

        phenos_t, scores_t, phenos_v, scores_v = preprocess_for_validation(
            phenos_t, scores_t, phenos_v, scores_v, 
            impute, strategy, standardise, normalise,
            dependent_var
        )

        if validation_approach == 'tt':
            phenos_t = phenos_t
            phenos_v = phenos_t
            scores_t = scores_t
            scores_v = scores_t
        elif validation_approach == 'tv':
            phenos_t = phenos_t
            phenos_v = phenos_v
            scores_t = scores_t
            scores_v = scores_v
        elif validation_approach == 'vv':
            phenos_t = phenos_v
            phenos_v = phenos_v
            scores_t = scores_v
            scores_v = scores_v

        model = RandomForestRegressor(
            max_depth=h_params["max_depth"], 
            n_estimators=h_params["n_estimators"],
            bootstrap=h_params["bootstrap"],
            max_features=h_params["max_features"],
            max_samples=h_params["max_samples"],
            min_samples_split = h_params['min_samples_split'],
            random_state=42, 
            verbose=0,
            n_jobs=-1
        )

        model.fit(phenos_t, scores_t)
        
        # Computer Predictions and Summary Stats
        y_hat = model.predict(phenos_v)
        neg_mae = -1*mean_absolute_error(scores_v, y_hat)
        results.append(neg_mae)

    results = np.array(results)
    return results.mean()

def get_baseline(phenos_subset, partition_training_dataset:bool, 
        h_params, random_state, n_jobs, 
        impute, strategy, standardise, normalise,
        dependent_var
    ):

    phenos_t, scores_t, phenos_v, scores_v = get_partitioned_data(
        phenos_subset, partition_training_dataset, n_splits, 1, 1, random_state
    )

    phenos_t, scores_t, phenos_v, scores_v = preprocess_for_validation(
        phenos_t, scores_t, phenos_v, scores_v, 
        impute, strategy, standardise, normalise,
        dependent_var
    )

    model = RandomForestRegressor(
        max_depth=h_params["max_depth"], 
        n_estimators=h_params["n_estimators"],
        bootstrap=h_params["bootstrap"],
        max_features=h_params["max_features"],
        max_samples=h_params["max_samples"],
        random_state=random_state, 
        verbose=0,
        n_jobs=n_jobs
    )
    
    predictions = []
    n_repeats = 10

    for i in range(n_repeats):
        model.fit(phenos_t, scores_t)
        shuffled = np.copy(phenos_v)
        np.random.shuffle(shuffled)
        y_hat = model.predict(shuffled)
        neg_mae = -1*mean_absolute_error(scores_v, y_hat)
        predictions.append(neg_mae)

    neg_mae = np.array(predictions).mean()
    return neg_mae

#Instantiate Controllers
use_full_dataset = True
use_database = False

data_sci_mgr = dsm.DataScienceManager(
    use_full_dataset=use_full_dataset,
    use_database=use_database)

# Declare Config Params 
test_id = 11
dependent_var = 'f_kir_score'

n_jobs = 16 - 1
n_splits = 4
n_repeats = 5
random_state = 336*2

partition_training_dataset = True
sigma_cut_off = 2

#date_str = data_sci_mgr.data_mgr.get_date_str()
date_str = '13062023'

test_plan_filename = 'Analysis/RandomForest/{0}/rf_test_plan_attributes_{0}.csv'.format(date_str)
test_plan_atts = pd.read_csv(test_plan_filename, index_col=None)
test_plan_atts = test_plan_atts[test_plan_atts['test_id'] == test_id].copy()

impute = bool(test_plan_atts['impute'].values[0])
strategy = test_plan_atts['strategy'].values[0]
normalise = bool(test_plan_atts['normalise'].values[0])
standardise = bool(test_plan_atts['standardise'].values[0])

h_params_filename = 'Analysis/RandomForest/{0}/rf_selected_h_params_r2_{0}.csv'.format(date_str)
h_params_df = pd.read_csv(h_params_filename, index_col=None)
h_params_df = h_params_df[h_params_df['test_id']==test_id].copy()

h_params = dict()
h_params['max_depth'] = int(h_params_df['max_depth'].values[0])
h_params['n_estimators'] = int(h_params_df['n_estimators'].values[0])
h_params['max_features'] = float(h_params_df['max_features'].values[0])
h_params['max_samples'] = float(h_params_df['max_samples'].values[0])
h_params['bootstrap'] = True
h_params['min_samples_split'] = int(h_params_df['min_samples_split'].values[0])

source_filename = test_plan_atts['pit_results'].values[0]
output_filename = 'Analysis/RandomForest/{1}/Test{0}/rf_final_score_r3_{0}_{1}.csv'.format(test_id, date_str)

# Pull Data from DB
importances_df = pd.read_csv(source_filename, index_col=0)

mean = importances_df.values.mean(axis=0)
std  = importances_df.values.std(axis=0)
filter_condition = mean -1*sigma_cut_off*std
indeces = np.where(filter_condition > 0)[0]
columns = importances_df.columns.values
phenos_subset = [columns[i] for i in indeces]

validation_approaches = ['tt', 'vv', 'tv']
output = {}

output['baseline'] = get_baseline(
    phenos_subset, partition_training_dataset, h_params, random_state, n_jobs, 
    impute, strategy, standardise, normalise, dependent_var
)

for idx, approach in enumerate(validation_approaches):
    neg_mae = get_final_score(
        phenos_subset=phenos_subset,
        partition_training_dataset = partition_training_dataset,
        h_params=h_params, 
        validation_approach=approach,
        n_splits=n_splits,
        n_repeats=n_repeats, 
        random_state = random_state,
        impute = impute, 
        strategy = strategy, 
        standardise = standardise, 
        normalise = normalise,
        dependent_var = dependent_var
    )
    key = 'avg_neg_mae' + '_' + validation_approaches[idx]
    output[key] = neg_mae

#Export Results
output['data_source'] = source_filename
output['test_plan'] = test_plan_filename
output['h_parmas'] = h_params_filename
output['test_id'] = test_id
output['dependent_var'] = dependent_var
output['partition_training_dataset'] = partition_training_dataset
output['sigma_cut_off'] = sigma_cut_off

output['n_splits'] = n_splits
output['n_repeats'] = n_repeats
output['random_state'] = random_state

output['impute'] = impute
output['standardise'] = standardise
output['normalise'] = normalise
output['strategy'] = strategy

for key in h_params:
    output[key] = h_params[key]

output['features'] = phenos_subset
output = pd.Series(output)
print(output)
output.to_csv(output_filename)

print('Complete.')