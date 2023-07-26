import os, argparse
import pandas as pd 

import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme()

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Multivariate Performance Permutation Test",
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
        "--InputDir",
        help="Input Directory for raw perf perm test results, as .csv",
        required=True,
        type=str,
    )

    required.add_argument(
        "-c",
        "--IterationCount",
        help="Iteration Count, i.e. number of files as int",
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

def main():

    args = parse_arguments()
    test_id = args["TestID"]
    output_directory = args["OutputDir"]
    input_directory = args["InputDir"]
    date_str = args["DateStr"]
    iteration_count = args["IterationCount"]

    filename_template = "mv_model_performance_perm_values.{it_id}.{test_id}.{date_str}.csv"

    consolidated_results_filename = os.path.join(
        output_directory, 
        "mv_model_performance_perm_values.{test_id}.{date_str}.csv".format(
            test_id=test_id, 
            date_str=date_str
        )
    )
    plot_filename = os.path.join(
        output_directory, 
        "mv_model_performance_perm_hist.{test_id}.{date_str}.png".format(
            test_id=test_id, 
            date_str=date_str
        )
    )

    neg_mae_data = []
    for it_id in range(0, iteration_count):
        filename = filename_template.format(it_id=it_id, test_id=test_id, date_str=date_str)
        filename = os.path.join(input_directory, filename)
        neg_mae_data_i = pd.read_csv(filename, index_col=0)
        neg_mae_data.append(neg_mae_data_i)

    neg_mae_data = pd.concat(neg_mae_data, ignore_index=True)

    neg_mae_data.to_csv(consolidated_results_filename)

    sns.histplot(data=neg_mae_data, bins=100)
    plt.savefig(plot_filename)

print('Starting...')
main()
print('Complete.')