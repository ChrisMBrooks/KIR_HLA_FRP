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
    output_dir = args["OutputDir"]
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
    n_splits = 5
    n_repeats = 4
    random_state = 42+21

    test_plan_atts = pd.read_csv(test_plan_filename, index_col=None)
    test_plan_atts = test_plan_atts[test_plan_atts['test_id'] == test_id].copy()

    impute = bool(test_plan_atts['impute'].values[0])
    strategy = test_plan_atts['strategy'].values[0]
    normalise = bool(test_plan_atts['normalise'].values[0])
    standardise = bool(test_plan_atts['standardise'].values[0])

    h_params_df = pd.read_csv(h_params_filename, index_col=None)
    h_params_df = h_params_df[h_params_df['test_id']==test_id].copy()

    h_params = dict()
    h_params['max_depth'] = int(h_params_df['max_depth'].values[0])
    h_params['n_estimators'] = int(h_params_df['n_estimators'].values[0])
    h_params['max_features'] = float(h_params_df['max_features'].values[0])
    h_params['max_samples'] = float(h_params_df['max_samples'].values[0])
    h_params['bootstrap'] = True
    h_params['min_samples_split'] = int(h_params_df['min_samples_split'].values[0])

    source_filename = test_plan_atts['source'].values[0]

    output_filename = os.path.join(
        output_dir, 
        'rf_results_r1_w_cv.{test_id}.{date_str}.csv'.format(
            test_id=test_id, 
            date_str=date_str
        )
    )
    weights_filename = os.path.join(
        output_dir, 
        'rf_r1_feature_importance_impurity_rankings.{test_id}.{date_str}.csv'.format(
            test_id=test_id, 
            date_str=date_str
        )
    )

    #Read in Subset of Immunophenotypes
    phenos_subset = list(pd.read_csv(source_filename, index_col=0).values[:, 0])

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

    # Instantiate Model    
    model = RandomForestRegressor(
        max_depth=h_params["max_depth"], 
        n_estimators=h_params["n_estimators"],
        bootstrap=h_params["bootstrap"],
        max_features=h_params["max_features"],
        max_samples=h_params["max_samples"],
        min_samples_split=h_params['min_samples_split'],
        random_state=random_state, 
        verbose=0,
        n_jobs=n_jobs
    )

    cv = RepeatedKFold(
        n_splits=n_splits, 
        n_repeats=n_repeats, 
        random_state=random_state
    )
    splits_gen = cv.split(phenos)

    results = []
    feature_weights_cv = []
    for i in range(0, n_repeats+1):
        split = next(splits_gen)
        train_indeces = split[0]
        test_indeces = split[1]

        model.fit(phenos[train_indeces, :], scores[train_indeces])

        feature_weights_cv.append(model.feature_importances_)

        # Computer Predictions and Summary Stats
        y_hat = model.predict(phenos[test_indeces, :])
        neg_mae = -1*mean_absolute_error(scores[test_indeces], y_hat)
        results.append(neg_mae)

    results = np.array(results)
    neg_mae = results.mean()

    columns = ['mdi_{}'.format(x) for x in range(0, n_repeats+1)]
    feature_weights_cv = np.vstack(feature_weights_cv).T
    feature_weights_cv_df = pd.DataFrame(feature_weights_cv, columns=columns)
    feature_weights_cv_df['mdi_mean'] = feature_weights_cv_df.values.sum(axis=1)/feature_weights_cv_df.values.shape[1]
    feature_weights_cv_df['mdi_std'] = feature_weights_cv_df.values.std(axis=1)
    feature_weights_cv_df['phenotype_id'] = phenos_subset
    feature_weights_cv_df = feature_weights_cv_df.sort_values(by='mdi_mean', ascending=False)
    feature_weights_cv_df.to_csv(weights_filename)

    output = {}
    output['data_source'] = source_filename
    output['test_plan'] = test_plan_filename
    output['h_params'] = h_params_filename
    output['weights'] = weights_filename
    output['avg_neg_mae'] = neg_mae
    output['max_depth'] = h_params['max_depth']
    output['n_estimators'] = h_params['n_estimators']
    output['max_features'] = h_params['max_features']
    output['max_samples'] = h_params['max_samples']
    output['bootstrap'] = h_params['bootstrap']
    output['min_samples_split'] = h_params['min_samples_split']

    output['test_id'] = test_id
    output['n_splits'] = n_splits
    output['n_repeats'] = n_repeats
    output['random_state'] = random_state

    output['impute'] = impute
    output['standardise'] = standardise
    output['normalise'] = normalise
    output['strategy'] = strategy

    output['run_id'] = run_id

    output = pd.Series(output)
    output.to_csv(output_filename)

    print(output)

    run_time = time.time() - start_time 
    print('run time:', run_time)

print('Starting...')

try:
    main()
except Exception as e:
    print("Execution failed due to the following error:")
    print(e)
    raise e

print('Complete.')