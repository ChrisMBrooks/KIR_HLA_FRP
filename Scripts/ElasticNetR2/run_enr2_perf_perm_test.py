# Using Forward Selection & Backward Selection to stress test the "significant" immunophenotypes

import sys, os, argparse, copy
import numpy as np
import pandas as pd

from sklearn.linear_model import ElasticNet
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import mean_absolute_error

from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from Controllers.DataScienceManager import DataScienceManager as dsm

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="ElasticNetR2 Performance Permutation Test",
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
    
    return vars(parser.parse_args())

def load_hyperparameters(filename:str):
    hp_frame = pd.read_csv(filename, index_col=0)
    alpha = float(hp_frame.loc["alpha"][0])
    l1_ratio = float(hp_frame.loc["l1_ratio"][0])

    return alpha, l1_ratio

def get_permuted_scores(
        phenos:np.array, scores:np.array, 
        n_splits:int, n_repeats:int, 
        random_state:int,
        alpha:float, 
        l1_ratio:float
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

        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
        shuffled_scores = copy.deepcopy(scores[train_indeces])
        np.random.shuffle(shuffled_scores)

        model.fit(phenos[train_indeces, :], shuffled_scores)
        y_hat = model.predict(phenos[test_indeces, :])
        neg_mae = -1*mean_absolute_error(scores[test_indeces], y_hat)
        performance_results.append(neg_mae)
    
    return performance_results

def get_output_filename(source_filename:str,test_id:int, 
    iteration_id:int, date_str:str, output_dir:str):
    
    optimised = ""
    if "optimised" in source_filename or "Optimised" in output_dir:
        optimised = "optimised."

    results_filename = os.path.join(
        output_dir, 
        "enr2_model_performance_perm_values.{optimised}{it_id}.{test_id}.{date_str}.csv".format(
            optimised=optimised,it_id=iteration_id, date_str=date_str, test_id=test_id
        )
    )

    return results_filename

def load_phenos_subset(source_filename:str, fs_bs_filter):

    if 'seq_selection' in source_filename:
        phenos_subset = pd.read_csv(source_filename, index_col=0)
        indeces = phenos_subset.values[:,1:3].sum(axis=1)
        indeces = np.where(indeces >= fs_bs_filter)
        phenos_subset = list(phenos_subset.iloc[indeces]['label'].values)
    else:
        phenos_subset = pd.read_csv(source_filename, index_col=0)
        phenos_subset = list(phenos_subset['optimised_rep'].values)
        
    return phenos_subset

def main():

    args = parse_arguments()
    test_id = args["TestID"]
    output_dir = args["OutputDir"]
    source_filename = args["Input"]
    iteration_id = args["IterationID"]
    test_plan_filename = args["TestPlan"]
    h_params_filename = args["HyperParams"]
    date_str = args["DateStr"]

    #Set Configuration Params
    metric = 'f_kir_score'
    fs_bs_filter = 2

    n_splits = 4
    n_repeats = 5

    test_plan_atts = pd.read_csv(test_plan_filename, index_col=None)
    test_plan_atts = test_plan_atts[test_plan_atts['test_id'] == test_id].copy()

    alpha, l1_ratio = load_hyperparameters(filename=h_params_filename)

    impute = bool(test_plan_atts['impute'].values[0])
    strategy = test_plan_atts['strategy'].values[0]
    normalise = bool(test_plan_atts['normalise'].values[0])
    standardise = bool(test_plan_atts['standardise'].values[0])

    # Compute Output Filenames
    results_filename = get_output_filename(
        source_filename=source_filename,test_id=test_id, 
        iteration_id=iteration_id, date_str=date_str, 
        output_dir=output_dir
    )

    #Retrieve Data
    phenos_subset = load_phenos_subset(
        source_filename=source_filename, 
        fs_bs_filter=fs_bs_filter
    )

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
    iteration_count = 1000
    performance_results = [] 
    for i in range(0, iteration_count, 1):
        random_state = int(np.random.random_sample(size=(1,1))[0,0]*iteration_count)
        perf_results_i = get_permuted_scores(
            phenos=phenos, scores=scores, 
            n_repeats=n_repeats, n_splits=n_splits, 
            random_state=random_state,
            l1_ratio=l1_ratio,
            alpha=alpha
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
