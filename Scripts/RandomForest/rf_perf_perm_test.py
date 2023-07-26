# Using Forward Selection & Backward Selection to stress test the "significant" immunophenotypes

import sys, os, argparse, copy
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import mean_absolute_error

from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from Controllers.DataScienceManager import DataScienceManager as dsm

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Random Forest Performance Permutation Test",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser._action_groups.pop()
    required = parser.add_argument_group("required arguments")

    required.add_argument(
        "-id",
        "--TestID",
        help="Test Identifier as type Int",
        required=True,
        type=int,
    )

    required.add_argument(
        "-tp",
        "--TestPlan",
        help="Test Plan as .csv",
        required=True,
        type=str,
    )

    required.add_argument(
        "-hp",
        "--HyperParams",
        help="Selected hyper parameters filename.",
        required=True,
        type=str,
    )

    required.add_argument(
        "-i",
        "--Input",
        help="FS-BS Results as .csv",
        required=True,
        type=str,
    )

    required.add_argument(
        "-it",
        "--IterationID",
        help="Iteration ID as int",
        required=True,
        type=int,
    )

    required.add_argument(
        "-s",
        "--IterationStep",
        help="Iteration step as int",
        required=True,
        type=int,
    )

    required.add_argument(
        "-d",
        "--DateStr",
        help="Date ID, DDMMYYYY",
        required=True,
        type=str,
    )

    required.add_argument(
        "-o",
        "--OutputDir",
        help="Output Directory",
        required=True,
        type=str,
    )

    required.add_argument(
        "-n",
        "--NumJobs",
        help="Number of Jobs as int.",
        required=True,
        type=int,
    )
    
    return vars(parser.parse_args())

def get_permuted_scores(
        phenos:np.array, scores:np.array,
        h_params:dict, 
        n_splits:int, n_repeats:int, 
        random_state:int,
        n_jobs:int
    ):

    cv = RepeatedKFold(
        n_splits=n_splits, 
        n_repeats=n_repeats, 
        random_state=random_state
    )
    splits_gen = cv.split(phenos)

    performance_results = []
    for i in range(0, n_repeats+1):
        split = next(splits_gen)
        train_indeces = split[0]
        test_indeces = split[1]

            # Fit the Model
        model = RandomForestRegressor(
            max_depth=h_params["max_depth"], 
            n_estimators=h_params["n_estimators"],
            bootstrap=h_params["bootstrap"],
            max_features=h_params["max_features"],
            max_samples=h_params["max_samples"],
            min_samples_split=h_params["min_samples_split"],
            random_state=random_state, 
            verbose=0,
            n_jobs=n_jobs
        )

        shuffled_scores = copy.deepcopy(scores[train_indeces])
        np.random.shuffle(shuffled_scores)

        model.fit(phenos[train_indeces, :], shuffled_scores)
        y_hat = model.predict(phenos[test_indeces, :])
        neg_mae = -1*mean_absolute_error(scores[test_indeces], y_hat)
        performance_results.append(neg_mae)
    
    return performance_results

def load_rf_h_params(h_params_filename:str, test_id:int):
    h_params_df = pd.read_csv(h_params_filename, index_col=None)
    h_params_df = h_params_df[h_params_df['test_id']==test_id].copy()

    h_params = dict()
    h_params['max_depth'] = int(h_params_df['max_depth'].values[0])
    h_params['n_estimators'] = int(h_params_df['n_estimators'].values[0])
    h_params['max_features'] = float(h_params_df['max_features'].values[0])
    h_params['max_samples'] = float(h_params_df['max_samples'].values[0])
    h_params['bootstrap'] = True
    h_params['min_samples_split'] = int(h_params_df['min_samples_split'].values[0])
    return h_params

def main():
    #Load Inputs
    args = parse_arguments()
    test_id = args["TestID"]
    h_params_filename = args["HyperParams"]
    h_params = load_rf_h_params(
        h_params_filename=h_params_filename, 
        test_id=test_id
    )
    output_dir = args["OutputDir"]
    source_filename = args["Input"]
    iteration_id = args["IterationID"]
    iteration_step = args["IterationStep"]
    test_plan_filename = args["TestPlan"]
    date_str = args["DateStr"]
    n_jobs = args["NumJobs"]

    #Set Configuration Params
    metric = 'f_kir_score'
    fs_bs_filter = 2

    n_splits = 4
    n_repeats = 5

    test_plan_atts = pd.read_csv(test_plan_filename, index_col=None)
    test_plan_atts = test_plan_atts[test_plan_atts['test_id'] == test_id].copy()

    impute = bool(test_plan_atts['impute'].values[0])
    strategy = test_plan_atts['strategy'].values[0]
    normalise = bool(test_plan_atts['normalise'].values[0])
    standardise = bool(test_plan_atts['standardise'].values[0])

    results_filename = os.path.join(
        output_dir, 
        "rf_model_performance_perm_values.{it_id}.{test_id}.{date_str}.csv".format(
            it_id=iteration_id, date_str=date_str, test_id=test_id
        )
    )

    #Retrieve Data
    phenos_subset = pd.read_csv(source_filename, index_col=0)
    indeces = phenos_subset.values[:,1:3].sum(axis=1)
    indeces = np.where(indeces >= fs_bs_filter)
    phenos_subset = list(phenos_subset.iloc[indeces]['label'].values)

    scores = data_sci_mgr.data_mgr.features(
        fill_na=False, fill_na_value=None, partition='training'
    )
    phenos = data_sci_mgr.data_mgr.outcomes(
        fill_na=False, fill_na_value=None, partition='training'
    )
    phenos = phenos[phenos_subset]

    # Standardise Data
    scores = scores[metric].values.reshape(-1,1)
    if len(phenos.values.shape) == 1:
        phenos = phenos.values.reshape(-1,1)
    else:
        phenos = phenos.values[:, 0:]

    phenos, scores = data_sci_mgr.data_mgr.preprocess_data(
        X=phenos, Y=scores, impute = impute, standardise = standardise, 
        normalise = normalise, strategy=strategy
    )

    scores = scores.ravel()

    #Evaluate Permutations
    performance_results = [] 
    for i in range(0, iteration_step, 1):
        random_state = int(np.random.random_sample(size=(1,1))[0,0]*iteration_step)
        perf_results_i = get_permuted_scores(
            phenos=phenos, scores=scores, 
            h_params=h_params,
            n_repeats=n_repeats, n_splits=n_splits, 
            random_state=random_state,
            n_jobs = n_jobs
        )
        performance_results.extend(perf_results_i)
    importances_df = pd.DataFrame(performance_results, columns=['psuedo_neg_mae'])

    # Export Results
    importances_df.to_csv(results_filename)

# Initiate Script
print('Starting...')

try:
    #Instantiate Controllers
    use_full_dataset=True
    use_database = False
    data_sci_mgr = dsm.DataScienceManager(
        use_full_dataset=use_full_dataset, 
        use_database=use_database
    )
    main()
except Exception as e:
    print('Unhandled exception:')
    print(e)

print('Complete.')
