import numpy as np
import pandas as pd
import time, uuid, os, sys, argparse

import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme()

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_validate
from sklearn.metrics import mean_absolute_error

from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from Controllers.DataScienceManager import DataScienceManager as dsm

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Univariate Validation",
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
        "-i",
        "--Input",
        help="Optimised model candidate perm test results as .csv",
        required=True,
        type=str,
    )

    required.add_argument(
        "-m",
        "--MeffFile",
        help="Meff results filename as .csv",
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

def get_mean_meff(filename:str):
    contents = pd.read_csv(filename, index_col=0)

    meff = (float(contents.loc["m_eff_nyholt"][0]) + float(contents.loc["m_eff_liji"][0]) + \
            float(contents.loc["m_eff_gao"][0]) + float(contents.loc["m_eff_galwey"][0]))/4.0
    
    return meff

def get_validation_data(phenos_subset:str):

    phenos = data_sci_mgr.data_mgr.outcomes(fill_na=False, fill_na_value=None, partition='validation')
    scores = data_sci_mgr.data_mgr.features(fill_na=False, fill_na_value=None, partition='validation')
    phenos = phenos[phenos_subset].copy()

    return phenos, scores

def get_final_score(candidate:str, dependent_var:str, phenos_df:pd.DataFrame, scores_df:pd.DataFrame):

    phenos_df1 = phenos_df[['public_id', candidate]].copy()
    Z = scores_df.merge(phenos_df1, on='public_id', how='inner')
        
    #Filter NAs
    Z0 = Z[~Z.isna().any(axis=1)]

    Z1 = Z0[[candidate, dependent_var]].values
    phenos = Z1[:, 0].astype(float)
    scores = Z1[:, 1].astype(float) 

    p_vals = data_sci_mgr.lrn_mgr.regression_p_score2(X=phenos, Y=scores)
    p_val = p_vals[-1]

    return p_val

def load_candidate_phenos(source_filename:str, sigma_cut_off:int):
    importances_df = pd.read_csv(source_filename, index_col=0)
    candidates = list(importances_df.columns.values)

    return candidates

def main():
    #Load & Parse Data
    args = parse_arguments()
    test_id = args["TestID"]
    output_dir = args["OutputDir"]
    source_filename = args["Input"]
    date_str = args["DateStr"]
    m_eff_filename = args["MeffFile"]

    m_eff = get_mean_meff(filename=m_eff_filename)

    optimised = ""
    if "optimised" in source_filename:
        optimised = "optimised."

    output_filename = os.path.join(output_dir, "univar_final_scores.{optimised}{test_id}.{date_str}.csv".format(
        optimised=optimised, date_str=date_str, test_id=test_id))

    #Declare Config Params
    dependent_var = 'f_kir_score' #'kir_count'

    # Pull Data from DB
    #Read in Subset of Immunophenotypes
    sigma_cut_off = 2
    candidate_phenos = load_candidate_phenos(source_filename=source_filename, sigma_cut_off=sigma_cut_off)

    partition = 'partition'
    scores_df = data_sci_mgr.data_mgr.features(fill_na = False, partition = partition)
    scores_df = scores_df[['public_id', dependent_var]]

    phenos_df = data_sci_mgr.data_mgr.outcomes(fill_na = False, partition = 'training')

    records = []
    for candidate in candidate_phenos:
        p_val = get_final_score(candidate, dependent_var, phenos_df, scores_df)
        alpha = 0.05
        alpha_eff = 0.05/m_eff
        significance_test = p_val < alpha_eff 

        record = [candidate, p_val, m_eff, alpha, alpha_eff, significance_test]
        records.append(record)
    
    columns = ['candidate', 'univar_p_val', 'm_eff', 'alpha', 'alpha_eff', 'significance_test']
    results = pd.DataFrame(records, columns=columns)

    results.to_csv(output_filename)
    print(results)

print('Starting...')

try:
    #Instantiate Controllers
    use_full_dataset=True
    use_database = False
    data_sci_mgr = dsm.DataScienceManager(
        use_full_dataset=use_full_dataset,
        use_database=use_database)
    main()
except Exception as e:
    print('Exception thrown due to the following error:')
    print(e)

print('Complete.')