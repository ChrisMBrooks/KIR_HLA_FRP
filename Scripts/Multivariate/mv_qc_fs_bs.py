# Using Forward Selection & Backward Selection to stress test the "significant" immunophenotypes

import time, uuid, sys, os
import numpy as np
import pandas as pd
import argparse

from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SequentialFeatureSelector

from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from Controllers.DataScienceManager import DataScienceManager as dsm

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Elastic Net Grid Search Without CV",
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
        "-i",
        "--Input",
        help="GS Feature Coefficients as .csv",
        required=True,
        type=str,
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
        "-j",
        "--NumJobs",
        help="Number of jobs to run in paralle.",
        required=True,
        type=int,
    )
    
    return vars(parser.parse_args())

def main():
    args = parse_arguments()
    test_id = args["TestID"]
    output_dir = args["OutputDir"]
    source_filename = args["Input"]
    test_plan_filename = args["TestPlan"]
    date_str = args["DateStr"]
    n_jobs = args["NumJobs"]

    start_time = time.time()
    run_id = str(uuid.uuid4().hex)

    #Instantiate Controllers
    use_full_dataset=True
    use_database = False
    data_sci_mgr = dsm.DataScienceManager(
        use_full_dataset=use_full_dataset, 
        use_database=use_database
    )

    # Declare Config Params
    forward_selection = True
    backward_selection = True

    dependent_var = 'f_kir_score' #'kir_count'
    scoring = 'neg_mean_absolute_error'

    n_splits = 4
    n_repeats = 10
    random_state = 42
    tolerance = None

    test_plan_atts = pd.read_csv(test_plan_filename, index_col=None)
    test_plan_atts = test_plan_atts[test_plan_atts['test_id'] == test_id].copy()

    impute = bool(test_plan_atts['impute'].values[0])
    strategy = test_plan_atts['strategy'].values[0]
    normalise = bool(test_plan_atts['normalise'].values[0])
    standardise = bool(test_plan_atts['standardise'].values[0])

    results_filename = os.path.join(output_dir, "mv_qc_fs_bs_candidate_features.{1}.{0}.csv".format(date_str, test_id))
    output_filename = os.path.join(output_dir, "mv_qc_fs_bs_summary.{1}.{0}.csv".format(date_str, test_id))

    #Load Subset of Immunophenotypes
    phenos_subset = pd.read_csv(source_filename, index_col=0)
    phenos_subset = list(phenos_subset.loc[np.count_nonzero(phenos_subset.values[:, 0:5], axis=1) >= 3].index)

    if len(phenos_subset) < 3:
        phenos_subset = list(pd.read_csv(source_filename, index_col=0).index)

    scores = data_sci_mgr.data_mgr.features(fill_na=False, fill_na_value=None, partition='training')
    phenos = data_sci_mgr.data_mgr.outcomes(fill_na=False, fill_na_value=None, partition='training')
    phenos = phenos[phenos_subset]

    # Standardise Data
    scores = scores[dependent_var].values.reshape(-1,1)
    if len(phenos.values.shape) == 1:
        phenos = phenos.values.reshape(-1,1)
    else:
        phenos = phenos.values[:, 0:]

    phenos, scores = data_sci_mgr.data_mgr.preprocess_data(
        X=phenos, Y=scores, impute = impute, standardise = standardise, 
        normalise = normalise, strategy=strategy
    )

    scores = scores.ravel()

    # Fit the Model 
    cv = RepeatedKFold(
        n_splits=n_splits, 
        n_repeats=n_repeats, 
        random_state=random_state
    )

    model = LinearRegression()

    if forward_selection:
        print('Starting Forward Selection...')
        sfs_for = SequentialFeatureSelector(
            model, direction='forward', 
            n_features_to_select='auto', 
            scoring=scoring, 
            tol=tolerance, cv=cv, n_jobs=n_jobs
        )

        sfs_for.fit(phenos, scores)
        for_selected_features = sfs_for.get_support()
        print('Forward Selection Complete.')

    if backward_selection:
        print('Starting Backward Selection...')
        sfs_bac = SequentialFeatureSelector(
            model, direction='backward', 
            n_features_to_select='auto', 
            scoring=scoring, 
            tol=tolerance, cv=cv, n_jobs=n_jobs
        )
        sfs_bac.fit(phenos, scores)
        bac_selected_features = sfs_bac.get_support()
        print('Backward Selection Complete.')

    print('Exporting Results...')
    flag = ''
    if forward_selection and not backward_selection:
        flag = 'fs'
        summary = [[phenos_subset[i], for_selected_features[i]] for i in range(0, len(phenos_subset), 1)]
        summary_df = pd.DataFrame(summary, columns=['label', 'forward_selected'])
    elif backward_selection and not forward_selection:
        flag = 'bs'
        summary = [[phenos_subset[i], bac_selected_features[i]] for i in range(0, len(phenos_subset), 1)]
        summary_df = pd.DataFrame(summary, columns=['label', 'backward_selected'])
    else:
        flag='fs_bs'
        summary = [[phenos_subset[i], for_selected_features[i], bac_selected_features[i]] for i in range(0, len(phenos_subset), 1)]
        summary_df = pd.DataFrame(summary, columns=['label', 'forward_selected', 'backward_selected'])

    summary_df.to_csv(results_filename)

    output = {}
    output['test_plan'] = test_plan_filename
    output['data_source'] = source_filename
    output['dependent_var'] = dependent_var
    output['flag'] = flag
    output['tolerance'] = tolerance
    output['scoring'] = scoring
    output['impute'] = impute
    output['strategy'] = strategy
    output['standardise'] = standardise
    output['normalise'] = normalise
    output['n_splits'] = n_splits
    output['n_repeats'] = n_repeats
    output['random_state'] = random_state
    output = pd.Series(output)
    output.to_csv(output_filename)

print('Starting...')
main()
print('Complete.')

