# Using Forward Selection & Backward Selection to stress test the "significant" immunophenotypes

import time, uuid, random, math, sys, os, argparse
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme()

from sklearn.linear_model import LinearRegression
from sklearn.inspection import permutation_importance

from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from Controllers.DataScienceManager import DataScienceManager as dsm

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Multivariate Feature Importance Permutation Test",
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
        "-ssr",
        "--SeqSelectionResults",
        help="FS-BS Results as .csv",
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
        "-sco",
        "--SigmaCutOff",
        help="Sigma Cut Off as int.",
        required=True,
        type=int,
    )

    required.add_argument(
        "-o",
        "--OutputDir",
        help="Output Directory",
        required=True,
        type=str,
    )
    
    return vars(parser.parse_args())

def format_output_filename(source_filename, sigma_cut_off, test_id, date_str, output_dir):
    optimised = ""
    if 'optimised' in source_filename:
        optimised = "optimised."
        
    results_filename = ""
    plot_filename = ""
    if sigma_cut_off == 0.0:
        results_filename = os.path.join(
            output_dir, 
            'mv_feature_importance_perm_values.{optimised}{test_id}.{date_str}.csv'.format(
                optimised=optimised,
                test_id=test_id, 
                date_str=date_str
            )
        )
        plot_filename = os.path.join(
            output_dir,
            "mv_feature_importance_box_plot.{optimised}{test_id}.{date_str}.png".format(
                optimised=optimised,
                test_id=test_id, 
                date_str=date_str
            )
        )
    elif sigma_cut_off == 2:
        results_filename = os.path.join(
            output_dir,
            'mv_feature_importance_perm_values_2sig.{optimised}{test_id}.{date_str}.csv'.format(
                optimised=optimised,
                test_id=test_id, 
                date_str=date_str
            )
        )
        plot_filename = os.path.join(
            output_dir,
            "mv_feature_importance_box_plot_2sig.{optimised}{test_id}.{date_str}.png".format(
                optimised=optimised,
                test_id=test_id, 
                date_str=date_str
            )
        )
    print(results_filename, plot_filename)
    return results_filename, plot_filename

def load_phenos_subset(source_filename:str, fs_bs_filter):

    if 'fs_bs_candidate_features' in source_filename:
        phenos_subset = pd.read_csv(source_filename, index_col=0)
        indeces = phenos_subset.values[:,1:3].sum(axis=1)
        indeces = np.where(indeces >= fs_bs_filter)
        phenos_subset = list(phenos_subset.iloc[indeces]['label'].values)
    else:
        phenos_subset = pd.read_csv(source_filename, index_col=0)
        phenos_subset = list(phenos_subset['optimised_rep'].values)
    
    return phenos_subset

def preprocess_for_validation(
        phenos_t:pd.DataFrame, scores_t:pd.DataFrame, 
        phenos_v:pd.DataFrame, scores_v:pd.DataFrame,
        dependent_var:str,
        impute, strategy, standardise, normalise 
    ):
    phenos_t, scores_t = data_sci_mgr.data_mgr.reshape(phenos_t, scores_t, dependent_var= dependent_var)
    phenos_v, scores_v = data_sci_mgr.data_mgr.reshape(phenos_v, scores_v, dependent_var= dependent_var)

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
    test_id = args["TestID"]
    output_dir = args["OutputDir"]
    source_filename = args["SeqSelectionResults"]
    test_plan_filename = args["TestPlan"]
    date_str = args["DateStr"]
    sigma_cut_off = args["SigmaCutOff"]

    #Set Configuration Params
    dependent_var = 'f_kir_score' #'kir_count'
    fs_bs_filter = 2
    partition_training_dataset = True
    scoring = 'neg_mean_absolute_error'

    num_repeats = 5
    n_jobs = 16 - 1
    random_state = 42*32
    n_splits = 4

    test_plan_atts = pd.read_csv(test_plan_filename, index_col=None)
    test_plan_atts = test_plan_atts[test_plan_atts['test_id'] == test_id].copy()

    impute = bool(test_plan_atts['impute'].values[0])
    strategy = test_plan_atts['strategy'].values[0]
    normalise = bool(test_plan_atts['normalise'].values[0])
    standardise = bool(test_plan_atts['standardise'].values[0])
        
    results_filename, plot_filename = format_output_filename(source_filename, sigma_cut_off, test_id, date_str, output_dir)

    #Retrieve Data
    phenos_subset = load_phenos_subset(source_filename=source_filename, fs_bs_filter=fs_bs_filter)

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
        phenos_t, scores_t, phenos_v, scores_v, 
        dependent_var = dependent_var,
        impute = impute, 
        strategy = strategy, 
        standardise = standardise, 
        normalise = normalise
    )

    # Fit the Model
    model = LinearRegression()
    fitted_model = model.fit(phenos_t, scores_t)

    # Run Permutation Tests
    results = permutation_importance(fitted_model, phenos_v, scores_v, n_repeats=num_repeats,
        random_state=random_state, scoring = scoring, n_jobs=n_jobs
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

    ax = importances_df.plot.box(vert=False, whis=1.5)
    ax.set_title("Permutation Importances ({})".format(perm_type))
    ax.axvline(x=0, color="k", linestyle="--")
    ax.set_xlabel("Decrease in accuracy score")
    ax.figure.tight_layout()

    plt.savefig(plot_filename)

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
