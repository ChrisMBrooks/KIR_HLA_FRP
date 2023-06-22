import itertools
import argparse
import numpy as np
import pandas as pd

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
    l1_ratio_step = 0.005
    l1_ratio_min = 0.001
    l1_ratio_max = 0.9 + l1_ratio_step

    l1_ratio_range = np.arange(l1_ratio_min, l1_ratio_max, l1_ratio_step)

    alpha_step = 0.001
    alpha_min = 0.001
    alpha_max = 0.04 + alpha_step

    alpha_range = np.arange(alpha_min, alpha_max, alpha_step)


    columns = ['l1_ratio', 'alpha']

    x = [l1_ratio_range, alpha_range]

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