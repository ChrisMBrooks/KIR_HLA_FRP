# Import Python Packages
import numpy as np
import pandas as pd
import sys, os, argparse

# Import Custom Python Packages
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from Controllers.DataScienceManager import DataScienceManager as dsm

HOME = os.path.expanduser('~')
os.environ['R_HOME'] = os.path.join(HOME, "anaconda3/envs/rpy2_env/lib/R")
os.environ['LD_LIBRARY_PATH'] = os.path.join(HOME, "anaconda3/envs/rpy2_env/lib/R/lib/:${LD_LIBRARY_PATH}")

# Deal with the R Stuff
import rpy2
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()

from rpy2.robjects.vectors import StrVector
import rpy2.robjects.packages as rpackages
from rpy2.robjects.packages import importr

# import R's utility package
utils = rpackages.importr('utils')

# select a mirror for R packages
utils.chooseCRANmirror(ind=1) # select the first mirror in the list

# Declare Dependencies
packnames = ('ggplot2', 'poolr')

# Selectively install what needs to be install.
names_to_install = [x for x in packnames if not rpackages.isinstalled(x)]
print(names_to_install)
if len(names_to_install) > 0:
    utils.install_packages(StrVector(names_to_install))

BASE  = importr('base')
POOLR = importr('poolr')

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="This is a python wrapper to run the m_effective R function in the poolr package.",
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
        help="Fullpath filename as string.",
        required=True,
        type=str,
    )

    required.add_argument(
        "-o",
        "--OutputDir",
        help="Output path as string.",
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
    
    return vars(parser.parse_args())

def format_output_filename(source_filename, test_id, date_str, output_dir):
    prefix = ''
    if 'rf_' in source_filename:
        prefix = 'rf'
    elif 'enr2' in source_filename:
        prefix = 'enr2'
    else:
        prefix = 'mv'

    output_filename = '{prefix}_m_effective_results.{test_id}.{date_str}.csv'.format(prefix=prefix, test_id=test_id, date_str=date_str)
    output_filename = os.path.join(output_dir, output_filename)

    return output_filename

def run_pool_r_in_r(correlation_matrix:np.array):
    results = {}
    results["m_eff_nyholt"] = POOLR.meff(correlation_matrix, method = "nyholt")[0]
    results["m_eff_liji"] = POOLR.meff(correlation_matrix, method = "liji")[0]
    results["m_eff_gao"] = POOLR.meff(correlation_matrix, method = "gao")[0]
    results["m_eff_galwey"] = POOLR.meff(correlation_matrix, method = "galwey")[0]
    return results

def main():
    # Parse Inputs
    args = parse_arguments()
    source_filename = args["Input"]
    output_dir = args["OutputDir"]
    test_id = args["TestID"]
    date_str = args["DateStr"]

    # Instantiate Controller
    use_full_dataset = True
    use_database = False
    data_sci_mgr = dsm.DataScienceManager(
        use_full_dataset=use_full_dataset, 
        use_database=use_database
    )

    # Load Phenotype Candidates
    phenos_subset = pd.read_csv(source_filename, index_col=0)
    phenos_subset = list(phenos_subset.columns)

    # Load Phenotype Measurements
    phenos = data_sci_mgr.data_mgr.outcomes(fill_na=False, fill_na_value=None, partition='training')
    phenos = phenos[phenos_subset]
    phenos = pd.DataFrame(phenos.astype(float), columns=phenos.columns)

    # Compute Pearson Correlation Matrix
    trait_corr = phenos.corr()
    trait_corr = trait_corr.values
    print(trait_corr)

    # Compute Meff
    results = run_pool_r_in_r(correlation_matrix=trait_corr)
    results["source_filename"] = source_filename

    # Export Results
    output_filename = format_output_filename(source_filename, test_id, date_str, output_dir)
    results = pd.Series(results)
    results.to_csv(output_filename)

    print(results)
    print(BASE.warnings())

print('Starting...')
main()
print('Complete.')







