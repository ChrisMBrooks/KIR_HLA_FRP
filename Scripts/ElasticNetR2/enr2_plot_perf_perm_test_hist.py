import os, argparse
import pandas as pd 
import numpy as np

from scipy.stats import norm

import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme()

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="ElaticNetR2 Performance Permutation Test",
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
        "-mr",
        "--ModelResults",
        help="Train Test Results filename as csv.",
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

def get_p_val(data:pd.DataFrame, result_value:float):
    result_value = float(result_value)
    x = data.values[:, 0].astype(float)
    std = x.std()
    mean = x.mean()
    p_val = norm.sf(x=result_value, loc=mean, scale=std)
    return p_val

def format_filename_template(results_filename:str):
    optimised = ""
    if "optimised" in results_filename:
        optimised = "optimised."
    
    filename_template = "enr2_model_performance_perm_values.{optimised}".format(
        optimised=optimised
    )

    filename_template = filename_template + "{it_id}.{test_id}.{date_str}.csv"

    return filename_template

def format_output_filenames(
        results_filename:str, output_directory:str, 
        test_id:int, date_str:str
):

    optimised = ""
    if "optimised" in results_filename:
        optimised = "optimised."

    # Format Output Filenames
    consolidated_results_filename = os.path.join(
        output_directory, 
        "enr2_model_performance_perm_values.{optimised}{test_id}.{date_str}.csv".format(
            optimised=optimised,
            test_id=test_id, 
            date_str=date_str
        )
    )
    plot_filename = os.path.join(
        output_directory, 
        "enr2_model_performance_perm_hist.{optimised}{test_id}.{date_str}.png".format(
            optimised=optimised,
            test_id=test_id, 
            date_str=date_str
        )
    )
    cdf_results_filename = os.path.join(
        output_directory, 
        "enr2_model_performance_cdf_results.{optimised}{test_id}.{date_str}.csv".format(
            optimised=optimised,
            test_id=test_id, 
            date_str=date_str
        )
    )

    return consolidated_results_filename, plot_filename, cdf_results_filename

def get_closest_bin_edge(neg_mae_data, model_score):
    bin_edges = np.histogram_bin_edges(neg_mae_data, bins=100)
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2

    # Find the bin closest to the model_score
    closest_bin_index = np.argmin(np.abs(bin_centers - model_score))
    closest_bin_value = bin_centers[closest_bin_index]
    return closest_bin_value

def main():
    # Load Arguments
    args = parse_arguments()
    test_id = args["TestID"]
    output_directory = args["OutputDir"]
    input_directory = args["InputDir"]
    date_str = args["DateStr"]
    iteration_count = args["IterationCount"]
    model_results_filename = args["ModelResults"]

    filename_template = format_filename_template(
        results_filename=model_results_filename
    )
    
    # Format Output Filenames
    consolidated_results_filename, plot_filename, cdf_results_filename = \
        format_output_filenames(
            results_filename=model_results_filename, 
            output_directory=output_directory, 
            test_id=test_id, date_str=date_str
        )

    # Load Model Results
    model_result = pd.read_csv(model_results_filename, index_col=0)
    model_score = float(model_result.loc["avg_neg_mae_tv"]["0"])

    #Consolidate Data
    neg_mae_data = []
    for it_id in range(0, iteration_count):
        filename = filename_template.format(it_id=it_id, test_id=test_id, date_str=date_str)
        filename = os.path.join(input_directory, filename)
        neg_mae_data_i = pd.read_csv(filename, index_col=0)
        neg_mae_data.append(neg_mae_data_i)
    neg_mae_data = pd.concat(neg_mae_data, ignore_index=True)

    #Export Raw Data
    neg_mae_data.to_csv(consolidated_results_filename)
    
    #Get P Value
    p_val = get_p_val(data=neg_mae_data, result_value=model_score)

    #Generate & Export Histogram
    bin_value = get_closest_bin_edge(
        neg_mae_data=neg_mae_data.values.astype(float), 
        model_score=model_score
    )

    sns.histplot(data=neg_mae_data, bins=100)
    plt.axvline(bin_value, ls='--')
    plt.text(model_score, 20, f' mae: {model_score:.4f}, \n p-val: {p_val:.3f}', color='darkblue', ha='left')
    plt.savefig(plot_filename)

    #Export Summary Results 
    output = {}
    output["test_id"] = test_id
    output["source_1"] = consolidated_results_filename
    output["source_2"] = model_results_filename
    output["neg_mae"] = model_score
    output["p_val"] = p_val
    output = pd.Series(output)
    output.to_csv(cdf_results_filename) 

print('Starting...')
main()
print('Complete.')