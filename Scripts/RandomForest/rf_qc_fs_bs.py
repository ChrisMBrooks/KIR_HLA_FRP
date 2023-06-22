# Using Forward Selection & Backward Selection to stress test the "significant" immunophenotypes

import time, uuid, sys, argparse, os
import numpy as np
import pandas as pd

from sklearn.model_selection import RepeatedKFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SequentialFeatureSelector

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
        help="Selected hyper parameters filename.",
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
        "-fi",
        "--FeatureImportances",
        help="MDI Importances as fullpath filename.",
        required=True,
        type=str,
    )

    required.add_argument(
        "-thr",
        "--SelectionThreshold",
        help="Candidate selection threshold as int.",
        required=True,
        type=int,
    )

    required.add_argument(
        "-fs",
        "--ForwardSelection",
        help="Perform forward selection, bool.",
        required=True,
        type=bool,
    )

    required.add_argument(
        "-bs",
        "--BackwardSelection",
        help="Perform backward selection, bool.",
        required=True,
        type=bool,
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
        help="Number of jobs to run in parallel as int.",
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
    source_filename = args["FeatureImportances"]
    selection_threshold = args["SelectionThreshold"]
    forward_selection = args["ForwardSelection"]
    backward_selection = args["BackwardSelection"]
    output_dir = args["OutputDir"]
    n_jobs = args["NumJobs"]

    use_full_dataset=True
    use_database = False

    #Instantiate Controllers
    data_sci_mgr = dsm.DataScienceManager(
        use_full_dataset=use_full_dataset, 
        use_database=use_database
    )

    # Declare Config Params
    n_splits = 4
    n_repeats = 10
    random_state_1 = 84
    random_state_2 = 168
    tolerance = None

    test_plan_atts = pd.read_csv(test_plan_filename, index_col=None)
    test_plan_atts = test_plan_atts[test_plan_atts['test_id'] == test_id].copy()

    impute = bool(test_plan_atts['impute'].values[0])
    strategy = test_plan_atts['strategy'].values[0]
    normalise = bool(test_plan_atts['normalise'].values[0])
    standardise = bool(test_plan_atts['standardise'].values[0])
    scoring = 'neg_mean_absolute_error'

    h_params_df = pd.read_csv(h_params_filename, index_col=None)
    h_params_df = h_params_df[h_params_df['test_id']==test_id].copy()

    h_params = dict()
    h_params['max_depth'] = int(h_params_df['max_depth'].values[0])
    h_params['n_estimators'] = int(h_params_df['n_estimators'].values[0])
    h_params['max_features'] = float(h_params_df['max_features'].values[0])
    h_params['max_samples'] = float(h_params_df['max_samples'].values[0])
    h_params['bootstrap'] = True
    h_params['min_samples_split'] = int(h_params_df['min_samples_split'].values[0])

    #Read in Subset of Immunophenotypes
    phenos_subset = pd.read_csv(source_filename, index_col=0)
    phenos_subset = phenos_subset.sort_values(by='mdi_mean', axis=0, ascending=False)
    phenos_subset = phenos_subset['phenotype_id'].values[0:selection_threshold]

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

    # Fit the Model 
    cv = RepeatedKFold(
        n_splits=n_splits, 
        n_repeats=n_repeats, 
        random_state=random_state_1
    )

    # Instantiate Model    
    model = RandomForestRegressor(
        max_depth=h_params["max_depth"], 
        n_estimators=h_params["n_estimators"],
        bootstrap=h_params["bootstrap"],
        max_features=h_params["max_features"],
        max_samples=h_params["max_samples"],
        random_state=random_state_2, 
        verbose=0,
        n_jobs=n_jobs
    )

    if forward_selection:
        print('Starting Forward Selection...')
        sfs_for = SequentialFeatureSelector(
            model, direction='forward', 
            n_features_to_select='auto', scoring=scoring, 
            tol=tolerance, cv=cv, n_jobs=n_jobs
        )

        sfs_for.fit(phenos, scores)
        for_selected_features = sfs_for.get_support()
        print('Forward Selection Complete.')

    if backward_selection:
        print('Starting Backward Selection...')
        sfs_bac = SequentialFeatureSelector(
            model, direction='backward', 
            n_features_to_select='auto', scoring=scoring, 
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

    # Format Output Files
    results_filename = os.path.join(
        output_dir, 
        "rf_seq_selection_candidates.{flag}.{thresh}.{test_id}.{date_str}.csv".format(
            flag=flag, 
            thresh=selection_threshold, 
            date_str=date_str, 
            test_id=test_id
        )
    )
    run_details_filename = os.path.join(
        output_dir, 
        "rf_seq_selection_summary.{flag}.{thresh}.{test_id}.{date_str}.csv".format(
            flag=flag, 
            thresh=selection_threshold, 
            date_str=date_str, 
            test_id=test_id
        )
    )
    # Export Results
    summary_df.to_csv(results_filename)

    output = dict()
    output['data_source'] = source_filename
    output['test_plan'] = test_plan_filename
    output['h_params'] = h_params_filename
    output['results'] = results_filename
    output['test_id'] = test_id
    output['selection_type'] = flag
    output['selection_threshold'] = selection_threshold

    output['n_splits'] = n_splits
    output['n_repeats'] = n_repeats
    output['random_state_1'] = random_state_1
    output['random_state_2'] = random_state_2
    output['tolerance'] = tolerance

    output['impute'] = impute
    output['standardise'] = standardise
    output['normalise'] = normalise
    output['strategy='] = strategy

    for key in h_params:
        output[key] = h_params[key]

    output = pd.Series(output)
    output.to_csv(run_details_filename)

print('Starting...')

try:
    main()
except Exception as e:
    print("Execution failed due to the following error:")
    print(e)
    raise e

print('Complete.')

