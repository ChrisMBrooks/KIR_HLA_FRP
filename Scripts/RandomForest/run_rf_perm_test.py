# Using Forward Selection & Backward Selection to stress test the "significant" immunophenotypes

import time, uuid, random, math, sys, os, argparse
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme()

from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance

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
        help="Test Plan as fullpath filename (.csv)",
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
        "-ssr",
        "--SeqSelectionResults",
        help="Sequential candidate selection results as fullpath filename, (.csv).",
        required=True,
        type=str,
    )


    required.add_argument(
        "-ds",
        "--DateStr",
        help="Date string in format DDMMYYYY.",
        required=True,
        type=str,
    )

    required.add_argument(
        "-sco",
        "--SigmaCutOff",
        help="Sigma Cut Off as int.",
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
        help="Number of jobs to run in parallel as int.",
        required=True,
        type=int,
    )

    return vars(parser.parse_args()) 

def preprocess_for_validation(
        phenos_t:pd.DataFrame, scores_t:pd.DataFrame, 
        phenos_v:pd.DataFrame, scores_v:pd.DataFrame,
        impute, strategy, standardise, normalise 
    ):
    phenos_t, scores_t = data_sci_mgr.data_mgr.reshape(phenos_t, scores_t)
    phenos_v, scores_v = data_sci_mgr.data_mgr.reshape(phenos_v, scores_v)

    phenos_t, scores_t, phenos_v, scores_v = data_sci_mgr.data_mgr.preprocess_data_v(
        X_t=phenos_t, Y_t=scores_t, X_v=phenos_v, Y_v=scores_v,
        impute = impute, strategy=strategy, standardise = standardise, 
        normalise = normalise
    )

    scores_t = scores_t.ravel()
    scores_v = scores_v.ravel()
    return phenos_t, scores_t, phenos_v, scores_v

def main():

    args = parse_arguments()
    test_plan_filename = args["TestPlan"]
    h_params_filename = args["HyperParams"]
    test_id = args["TestID"]
    source_filename = args["SeqSelectionResults"]
    date_str = args["DateStr"]
    sigma_cut_off = args["SigmaCutOff"]
    output_dir = args["OutputDir"]
    n_jobs = args["NumJobs"]


    #Declare Config Params
    scoring = 'neg_mean_absolute_error'
    n_splits = 5
    num_repeats = 4
    random_state = 672

    partition_training_dataset = True
    fs_bs_filter = 2

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
    
    results_filename = ""
    plot_filename = ""
    if sigma_cut_off == 0.0:
        results_filename = os.path.join(
            output_dir, 
            'rf_feature_importance_perm_values.{test_id}.{date_str}.csv'.format(
                test_id=test_id, 
                date_str=date_str
            )
        )
        plot_filename = os.path.join(
            output_dir,
            "rf_feature_importance_box_plot.{test_id}.{date_str}.png".format(
                test_id=test_id, 
                date_str=date_str
            )
        )
    elif sigma_cut_off == 2:
        results_filename = os.path.join(
            output_dir,
            'rf_feature_importance_perm_values_2sig.{test_id}.{date_str}.csv'.format(
                test_id=test_id, 
                date_str=date_str
            )
        )
        plot_filename = os.path.join(
            output_dir,
            "rf_feature_importance_box_plot_2sig.{test_id}.{date_str}.png".format(
                test_id=test_id, 
                date_str=date_str
            )
        )

    #Retrieve Data
    phenos_subset = pd.read_csv(source_filename, index_col=0)
    indeces = phenos_subset.values[:,1:3].sum(axis=1)
    indeces = np.where(indeces >= fs_bs_filter)
    phenos_subset = list(phenos_subset.iloc[indeces]['label'].values)

    scores_t = data_sci_mgr.data_mgr.features(fill_na=False, fill_na_value=None, partition='training')
    phenos_t = data_sci_mgr.data_mgr.outcomes(fill_na=False, fill_na_value=None, partition='training')
    phenos_t = phenos_t[phenos_subset]

    scores_v = data_sci_mgr.data_mgr.features(fill_na=False, fill_na_value=None, partition='validation')
    phenos_v = data_sci_mgr.data_mgr.outcomes(fill_na=False, fill_na_value=None, partition='validation')
    phenos_v = phenos_v[phenos_subset]

    # Partition Data
    if partition_training_dataset:
        phenos_t, scores_t, phenos_v, scores_v = data_sci_mgr.data_mgr.partition_training_data(
            phenos_t, scores_t, n_splits, random_state
        )

    # Massage Data
    phenos_t, scores_t, phenos_v, scores_v = preprocess_for_validation(
        phenos_t, scores_t, phenos_v, scores_v, impute, strategy, 
        standardise, normalise
    )

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

    fitted_model = model.fit(phenos_t, scores_t)

    # Run Permutation Tests
    results = permutation_importance(fitted_model, phenos_v, scores_v, n_repeats=num_repeats,
        random_state=random_state, scoring=scoring, n_jobs=n_jobs
    )

    # Format Results
    results_values = []
    for i in range(0, results.importances.shape[1]):
        importance_values = results.importances[:, i]
        results_values.append(importance_values)

    importances_df = pd.DataFrame(results_values, columns=phenos_subset)

    sorted_indeces = np.argsort(importances_df.values.mean(axis=0))
    columns = [phenos_subset[x] for x in sorted_indeces]
    importances_df = importances_df[columns].copy()

    # Export Results
    importances_df.to_csv(results_filename)

    # Plot Results
    if partition_training_dataset:
        perm_type = 'train-test'
    else:
        perm_type = 'train-validate'

    mean = importances_df.values.mean(axis=0)
    std  = importances_df.values.std(axis=0)
    filter_condition = mean -1*sigma_cut_off*std
    indeces = np.where(filter_condition > 0)[0]
    columns = importances_df.columns.values
    columns = [columns[i] for i in indeces]
    importances_df = importances_df[columns]

    ax = importances_df.plot.box(vert=False, whis=1.5)
    ax.set_title("Permutation Importances ({})".format(perm_type))
    ax.axvline(x=0, color="k", linestyle="--")
    ax.set_xlabel("Decrease in accuracy score")
    ax.figure.tight_layout()

    plt.savefig(plot_filename)

print("Starting...")

try:
    #Instantiate Controllers
    use_full_dataset = True
    use_database = False

    data_sci_mgr = dsm.DataScienceManager(
    use_full_dataset=use_full_dataset,
    use_database=use_database)
    main()
except Exception as e:
    print("Execution failed due to the following error:")
    print(e)
    raise e

print('Complete.')
