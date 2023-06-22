import itertools, argparse,sys
import numpy as np
import pandas as pd

from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from Controllers.DataScienceManager import DataScienceManager as dsm


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="This is a test...",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser._action_groups.pop()
    required = parser.add_argument_group("required arguments")

    required.add_argument(
        "-o",
        "--Output",
        help="Fullpath Filename Output",
        required=True,
        type=str,
    )
    
    return vars(parser.parse_args())

def get_h_params_df():
    # Declare Grid Search Ranges 
    max_depth_step = 2
    max_depth_min = 2
    max_depth_max = 20 + max_depth_step

    # max_depth: the maximum depth of the tree 
    max_depth = range(
        max_depth_min, max_depth_max, max_depth_step
    )

    # n_estimators: the number of trees in the forest
    num_trees_step = 10
    num_trees_min = 10
    num_trees_max = 130 + num_trees_step

    n_estimators = np.arange(
        num_trees_min, num_trees_max, num_trees_step
    )

    max_features_step = .1
    max_features_min = .1
    max_features_max = .9 + max_features_step

    max_features = np.arange(
        max_features_min, max_features_max, max_features_step
    )

    max_samples_step = .1
    max_samples_min = .1
    max_samples_max = .9 + max_samples_step

    max_samples = np.arange(
        max_samples_min, max_samples_max, max_samples_step
    )

    min_samples_split_step = 1
    min_samples_split_min = 5
    min_samples_split_max = 40 + min_samples_split_step

    min_samples_split = np.arange(
        min_samples_split_min, min_samples_split_max, min_samples_split_step
    )

    bootstrap = [True]

    columns = ['max_depth', 'n_estimators', 'max_features', 'max_samples', 'bootstrap', 'min_samples_split']

    x = [max_depth,n_estimators,max_features,max_samples,bootstrap,min_samples_split]

    params = [p for p in itertools.product(*x)]

    params_df = pd.DataFrame(params, columns=columns)
    return params_df

def main():
    args = parse_arguments()
    output_filepath = args["Output"]
    
    params_df = get_h_params_df()
    filename = output_filepath
    params_df.to_csv(filename)

print('Starting..')

try:
    main()
except Exception as e:
    print("Execution failed due to the following error:")
    print(e)
    
print('Complete.')