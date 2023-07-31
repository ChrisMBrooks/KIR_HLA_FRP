import numpy as np
import pandas as pd
import time, uuid, os, sys, argparse, glob

from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from Controllers.DataScienceManager import DataScienceManager as dsm

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Optimise model candidates for correlation w/ iKIR score.",
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
        "-s",
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
        "-o",
        "--OutputDir",
        help="Output Directory",
        required=True,
        type=str,
    )
    
    return vars(parser.parse_args())

def get_list_of_all_phenos(data_sci_mgr:object):
    phenos = data_sci_mgr.data_mgr.outcomes(
        fill_na=False, fill_na_value=None, partition='training').columns[1:-2]
    phenos = list(phenos)
    return phenos

def load_training_data(data_sci_mgr:object, phenos_subset:list, 
        impute:bool, standardise:bool, 
        normalise:bool, strategy:str,
        dependent_var:str
    ):

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

    return phenos, scores

def load_correlates_df(clustering_threshold:float):
    correlates = pd.DataFrame()
    if clustering_threshold != 1.0:
        pattern = "Data/phenos_corr_dict_{threshold}_*.parquet".format(threshold=clustering_threshold)
        matches = glob.glob(pathname=pattern)
        filename = matches[0]
        correlates = pd.read_parquet(filename)

    return correlates

def get_cohort_from_rep(correlates_frame:pd.DataFrame, pheno_rep:str):
    if correlates_frame.empty:
        return [pheno_rep]
    else:
        correlates = list(correlates_frame[correlates_frame['label'] == pheno_rep]['correlates'].values[0])

        if len(correlates) < 1:
            cohort = [pheno_rep]
        else:
            cohort = correlates + [pheno_rep]
        return cohort

def get_best_cohort_rep(cohort:list, phenos_labels:list, phenos, scores, dependent_var:str):

    phenos = pd.DataFrame(phenos, columns=phenos_labels)
    phenos[dependent_var] = scores

    phenos_subset = phenos[cohort+[dependent_var]].copy()
    corr = phenos_subset.corr()
    corr = corr.sort_values(by=dependent_var, ascending=False)
    corr = corr[dependent_var]
    indeces = corr.index.values  

    for item in indeces:
        if item != dependent_var:
            return item

def load_phenos_subset(source_filename:str, fs_bs_filter:int):
    phenos_subset = pd.read_csv(source_filename, index_col=0)
    indeces = phenos_subset.values[:,1:3].sum(axis=1)
    indeces = np.where(indeces >= fs_bs_filter)
    phenos_subset = list(phenos_subset.iloc[indeces]['label'].values)
    
    return phenos_subset

def main():
    # Instantiate Controller
    use_full_dataset = True
    use_database = False
    data_sci_mgr = dsm.DataScienceManager(
        use_full_dataset=use_full_dataset, 
        use_database=use_database
    )
    
    #Load & Parse Required Inputs
    args = parse_arguments()
    test_id = args["TestID"]
    output_dir = args["OutputDir"]
    source_filename = args["SeqSelectionResults"]
    test_plan_filename = args["TestPlan"]
    date_str = args["DateStr"]

    dependent_var = 'f_kir_score'
    fs_bs_filter = 2

    test_plan_atts = pd.read_csv(test_plan_filename, index_col=None)
    test_plan_atts = test_plan_atts[test_plan_atts['test_id'] == test_id].copy()

    original_phenos_subset = load_phenos_subset(
        source_filename=source_filename, fs_bs_filter=fs_bs_filter)

    impute = bool(test_plan_atts['impute'].values[0])
    strategy = test_plan_atts['strategy'].values[0]
    normalise = bool(test_plan_atts['normalise'].values[0])
    standardise = bool(test_plan_atts['standardise'].values[0])
    clustering_threshold = float(test_plan_atts['clustering_threshold'].values[0])

    correlates_frame = load_correlates_df(clustering_threshold)
    all_phenos = get_list_of_all_phenos(data_sci_mgr=data_sci_mgr)
    phenos, scores = load_training_data(
        data_sci_mgr=data_sci_mgr, phenos_subset=all_phenos, 
        impute=impute, standardise=standardise, 
        normalise=normalise, strategy=strategy, 
        dependent_var=dependent_var
    )

    # Format Output Filename
    results_filename = os.path.join(output_dir, "optimised_model_candidates.{1}.{0}.csv".format(date_str, test_id))

    # Optimise Candidates
    optimised_phenos_subset = []
    for cohort_rep in original_phenos_subset:

        cohort = get_cohort_from_rep(
            correlates_frame=correlates_frame, 
            pheno_rep=cohort_rep
        )

        best_cohort_rep = get_best_cohort_rep(
            cohort=cohort, 
            phenos_labels=all_phenos,
            phenos=phenos, 
            scores=scores,
            dependent_var=dependent_var
        )

        optimised_phenos_subset.append(best_cohort_rep)

    # Perform Quality Check
    A = set(original_phenos_subset)
    B = set(original_phenos_subset)
    C = A.intersection(B)
    no_change = len(A) == len(C)
    if no_change:
        print('Model was not changed. All reps were already optimal.')

    #Export Results
    output = pd.DataFrame()
    output['original_rep'] = original_phenos_subset
    output['optimised_rep'] = optimised_phenos_subset
    output.to_csv(results_filename)

print('Starting...')
main()
print('Compelte.')