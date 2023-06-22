# Compute Results w/ Standardisation. 
# More robust to outliers. 

import numpy as np
import pandas as pd
import time, uuid, sys, os, argparse

import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme()

from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import ElasticNet

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
        "-hp",
        "--HyperParams",
        help="Hyper Params File as .csv",
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

def main():
    args = parse_arguments()
    test_id = args["TestID"]
    output_dir = args["OutputDir"]
    test_plan_filename = args["TestPlan"]
    h_params_filename = args["HyperParams"]
    date_str = args["DateStr"]

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
    dependent_var = 'f_kir_score'
    n_splits = None
    n_repeats = None

    test_plan_atts = pd.read_csv(test_plan_filename, index_col=None)
    test_plan_atts = test_plan_atts[test_plan_atts['test_id'] == test_id].copy()

    impute = bool(test_plan_atts['impute'].values[0])
    strategy = test_plan_atts['strategy'].values[0]
    normalise = bool(test_plan_atts['normalise'].values[0])
    standardise = bool(test_plan_atts['standardise'].values[0])

    #Create Export Directories
    source_filename = test_plan_atts['source'].values[0]

    # Declare export filename
    output_filename = os.path.join(output_dir,'en_gs_run_summary_results_no_cv.{1}.{0}.csv'.format(date_str, test_id))
    gs_details_filename = os.path.join(output_dir,'en_gs_details_no_cv.{1}.{0}.csv'.format(date_str, test_id))

    # Pull Data from DB
    #Read in Subset of Immunophenotypes
    phenos_subset = list(pd.read_csv(source_filename, index_col=0).values[:, 0])
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
        X=phenos, Y=scores, 
        impute = impute, 
        strategy = strategy,
        standardise = standardise, 
        normalise = normalise
    )

    #Import Matrix of H Params
    h_params_df = pd.read_csv(h_params_filename, index_col=0)

    grid_results = []
    for index, row in h_params_df.iterrows():

        # Instantiate evaluation method model
        model = ElasticNet(alpha=row['alpha'], 
            l1_ratio=row['l1_ratio']
        )

        model.fit(phenos, scores)

        y_hat = model.predict(phenos)
        neg_mae = -1*mean_absolute_error(scores, y_hat)
        record = [row['alpha'], row['l1_ratio'], neg_mae]
        grid_results.append(record)

    grid_results = pd.DataFrame(grid_results, columns=['alpha', 'l1_ratio', 'mae'])
    grid_results.to_csv(gs_details_filename)

    run_time = time.time() - start_time 

    #Export Results
    #Ouput the Core Results
    output = {}
    output['test_plan'] = test_plan_filename
    output['data_source'] = source_filename
    output['h_params_source'] = h_params_filename
    output['dependent_var'] = dependent_var
    output['impute'] = impute
    output['strategy'] = strategy
    output['standardise'] = standardise
    output['normalise'] = normalise
    output['n_splits'] = n_splits
    output['n_repeats'] = n_repeats
    output['run_time'] = run_time
    output['run_id'] = run_id
    output = pd.Series(output)
    output.to_csv(output_filename)
    print(output)
    

print('Starting...')
main()
print('Complete.')