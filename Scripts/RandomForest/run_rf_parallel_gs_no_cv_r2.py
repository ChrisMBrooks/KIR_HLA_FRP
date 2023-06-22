import os, random, math, time, uuid, sys, argparse
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme()

from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from Controllers.DataScienceManager import DataScienceManager as dsm

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Partial Random Forest Grid Search Search w/ CV",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser._action_groups.pop()
    required = parser.add_argument_group("required arguments")

    required.add_argument(
        "-tp",
        "--TestPlan",
        help="Test Plan Fullpath Filename",
        required=True,
        type=str,
    )

    required.add_argument(
        "-hp",
        "--HyperParams",
        help="Hyper parameter permutations filename.",
        required=True,
        type=str,
    )

    required.add_argument(
        "-id",
        "--TestID",
        help="Test Identifier as type Integer.",
        required=True,
        type=int,
    )

    required.add_argument(
        "-ds",
        "--DateStr",
        help="Date string in format DDMMYYYY.",
        required=True,
        type=str,
    )

    required.add_argument(
        "-ssr",
        "--SeqSelectionResults",
        help="Sequential Selection Results as fullpath filename.",
        required=True,
        type=str,
    )

    required.add_argument(
        "-si",
        "--StartIndex",
        help="Start index as Integer.",
        required=True,
        type=int,
    )

    required.add_argument(
        "-s",
        "--Step",
        help="Increment step as integer.",
        required=True,
        type=int,
    )

    required.add_argument(
        "-o",
        "--OutputDir",
        help="Output directory",
        required=True,
        type=str,
    )

    required.add_argument(
        "-j",
        "--NumJobs",
        help="Number of jobs to run in paralle.",
        required=True,
        type=int,
    )

    return vars(parser.parse_args())

def main():
    args = parse_arguments()
    test_plan_filename = args["TestPlan"]
    h_params_filename = args["HyperParams"]
    test_id = args["TestID"]
    date_str = args["DateStr"]
    source_filename = args["SeqSelectionResults"]
    start_index = args["StartIndex"]
    step = args["Step"]
    output_dir = args["OutputDir"]
    n_jobs = args["NumJobs"]

    #Instantiate Controllers
    use_full_dataset=True
    use_database = False
    data_sci_mgr = dsm.DataScienceManager(
        use_full_dataset=use_full_dataset, 
        use_database=use_database
    )

    # Declare Config Params
    n_splits = None
    n_repeats = None
    random_state = 210

    fs_bs_filter = 2

    test_plan_atts = pd.read_csv(test_plan_filename, index_col=None)
    test_plan_atts = test_plan_atts[test_plan_atts['test_id'] == test_id].copy()

    impute = bool(test_plan_atts['impute'].values[0])
    strategy = test_plan_atts['strategy'].values[0]
    normalise = bool(test_plan_atts['normalise'].values[0])
    standardise = bool(test_plan_atts['standardise'].values[0])
    scoring = 'neg_mean_absolute_error'

    output_filename = os.path.join(
        output_dir, 
        'rf_parallel_gs_results_r2_wo_cv.{start_index}.{step}.{test_id}.{date_str}.csv'.format(
            start_index=start_index, 
            step=step, 
            test_id=test_id, 
            date_str=date_str
        )
    )

    #Read in Subset of Immunophenotypes
    phenos_subset = pd.read_csv(source_filename, index_col=0)
    indeces = phenos_subset.values[:,1:3].sum(axis=1)
    indeces = np.where(indeces >= fs_bs_filter)
    phenos_subset = list(phenos_subset.iloc[indeces]['label'].values)

    scores = data_sci_mgr.data_mgr.features(fill_na=False, fill_na_value=None, partition='training')
    phenos = data_sci_mgr.data_mgr.outcomes(fill_na=False, fill_na_value=None, partition='training')

    phenos = phenos[phenos_subset]

    # Standardise Data
    scores = scores['f_kir_score'].values.reshape(-1,1)
    if len(phenos.values.shape) == 1:
        phenos = phenos.values.reshape(-1,1)
    else:
        phenos = phenos.values[:, 0:]

    phenos, scores = data_sci_mgr.data_mgr.preprocess_data(
        X=phenos, Y=scores, impute = impute, standardise = standardise, 
        normalise = normalise, strategy=strategy
    )

    scores = scores.ravel()

    #Import Matrix of H Params
    h_params_df = pd.read_csv(h_params_filename, index_col=0)

    # Run Models
    records = []
    for idx in range(start_index, start_index + step, 1):
        if idx < h_params_df.shape[0]:
            record = dict()

            if h_params_df.iloc[idx]["bootstrap"] == False:
                max_samples = None
            else: 
                max_samples = h_params_df.iloc[idx]["max_samples"]

            # Instantiate Model    
            model = RandomForestRegressor(
                max_depth=h_params_df.iloc[idx]["max_depth"], 
                n_estimators=h_params_df.iloc[idx]["n_estimators"],
                max_features=h_params_df.iloc[idx]["max_features"],
                max_samples=max_samples,
                bootstrap=h_params_df.iloc[idx]["bootstrap"],
                min_samples_split=h_params_df.iloc[idx]['min_samples_split'],
                random_state=random_state, 
                verbose=0,
                n_jobs=n_jobs
            )

            model.fit(phenos, scores)

            # Computer Predictions and Summary Stats
            y_hat = model.predict(phenos)
            neg_mae = -1*mean_absolute_error(scores, y_hat)

            record['index'] = idx
            record['max_depth'] = h_params_df.iloc[idx]['max_depth']
            record['n_estimators'] = h_params_df.iloc[idx]['n_estimators']
            record['max_features'] = h_params_df.iloc[idx]['max_features']
            record['max_samples'] = h_params_df.iloc[idx]['max_samples']
            record['bootstrap'] = h_params_df.iloc[idx]['bootstrap']
            record['min_samples_split'] = h_params_df.iloc[idx]['min_samples_split']
            record['mean_neg_mae'] = neg_mae
            records.append(record)

    output = pd.DataFrame(records)
    output.to_csv(output_filename)

try:
    main()
except Exception as e:
    print("Execution failed due to the following error:")
    print(e)
    raise e
