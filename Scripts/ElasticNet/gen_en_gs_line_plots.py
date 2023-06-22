import os, itertools
import pandas as pd
import numpy as np
import seaborn as sns
import argparse
import matplotlib.pyplot as plt
sns.set_theme()

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
        "-d",
        "--DateStr",
        help="Date ID, DDMMYYYY",
        required=True,
        type=str,
    )
    
    required.add_argument(
        "-fw",
        "--FileWCV",
        help="Output Directory",
        required=True,
        type=str,
    )
    required.add_argument(
        "-fwo",
        "--FileWOCV",
        help="Output Directory",
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

def generate_en_gs_line_plots(test_id:list, date_str:str, 
        filename_w_cv:str, filename_wo_cv:str, output_dir:str
):
    
    gs_results_w_cv = pd.read_csv(filename_w_cv, index_col=0)
    gs_results_wo_cv = pd.read_csv(filename_wo_cv, index_col=0)

    palette = itertools.cycle(sns.color_palette())
    c1 = next(palette)
    c2 = next(palette)

    for key in ['alpha', 'l1_ratio']:
            
        fig = plt.figure()
        ax1 = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 1, 2, sharex=ax1)

        sns.lineplot(data=gs_results_w_cv, x=key, y='mae', color=c1, estimator='mean', errorbar='sd', label='w/ cv', ax=ax1)
        sns.lineplot(data=gs_results_wo_cv, x=key, y='mae', color=c2, estimator='mean', errorbar='sd', label='w/o cv', ax=ax2)
        plt.legend(loc='upper right')
            
        fig.suptitle('EN GS Results - Test: {0} - {1}'.format(test_id, key))

        output_filename = 'rf_gs_results_line_plot.{0}.{1}.{2}.png'.format(key, test_id, date_str)
        output_filename = os.path.join(output_dir, output_filename)

        plt.savefig(output_filename)
        plt.clf()

print('Starting...')

try:
    args = parse_arguments()
    test_id = args["TestID"]
    output_dir = args["OutputDir"]
    filename_w_cv = args["FileWCV"]
    filename_wo_cv = args["FileWOCV"]
    date_str = args["DateStr"]

    generate_en_gs_line_plots(
        test_id=test_id, 
        date_str=date_str,
        filename_w_cv=filename_w_cv,
        filename_wo_cv=filename_wo_cv,
        output_dir=output_dir
    )

except Exception as e:
    print("Job failed due to the following error message:")
    print(e)

print('Complete.')