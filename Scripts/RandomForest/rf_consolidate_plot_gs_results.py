import os, itertools, sys, argparse
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme()

from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from Controllers.DataScienceManager import DataScienceManager as dsm

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Partial Random Forest Grid Search Search w/ CV",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser._action_groups.pop()
    required = parser.add_argument_group("required arguments")

    required.add_argument(
        "-id",
        "--TestID",
        help="Test Identifier as type Integer.",
        required=True,
        type=int,
    )

    required.add_argument(
        "-ds",
        "--DateStr",
        help="Date string in format DDMMYYYY.",
        required=True,
        type=str,
    )

    required.add_argument(
        "-si",
        "--StartIndex",
        help="Start index as Integer.",
        required=True,
        type=int,
    )

    required.add_argument(
        "-spi",
        "--StopIndex",
        help="Final index as integer.",
        required=True,
        type=int,
    )

    required.add_argument(
        "-s",
        "--Step",
        help="Increment step as integer.",
        required=True,
        type=int,
    )

    required.add_argument(
        "-it",
        "--Iteration",
        help="RF iteration as int (i.e. r1, r2).",
        required=True,
        type=int,
    )

    required.add_argument(
        "-i",
        "--InputDir",
        help="Input directory",
        required=True,
        type=str,
    )

    required.add_argument(
        "-o",
        "--OutputDir",
        help="Output directory",
        required=True,
        type=str,
    )

    return vars(parser.parse_args())

def consolidate_files(test_id:int, date_str:int, start_index:int, final_index:int, step:int, input_dir:str, output_dir:str, rn:str):
    filename_template_w_cv = os.path.join(input_dir,'rf_parallel_gs_results_{rn}_w_cv.{start_index}.{step}.{test_id}.{date_str}.csv') 
    filename_template_wo_cv = os.path.join(input_dir,'rf_parallel_gs_results_{rn}_wo_cv.{start_index}.{step}.{test_id}.{date_str}.csv') 

    frames_w_cv = []
    frames_wo_cv = []
    for index in range(start_index, final_index + step, step):
        filename_w_cv = filename_template_w_cv.format(
            start_index=index, 
            step=step, 
            test_id=test_id, 
            date_str=date_str,
            rn=rn
        )
        frame_w_cv = pd.read_csv(filename_w_cv, index_col=0)
        frames_w_cv.append(frame_w_cv)

        filename_wo_cv = filename_template_wo_cv.format(
            start_index=index, 
            step=step, 
            test_id=test_id, 
            date_str=date_str,
            rn=rn
        )
        frame_wo_cv = pd.read_csv(filename_wo_cv, index_col=0)
        frames_wo_cv.append(frame_wo_cv)

    grid_search_results_w_cv = pd.concat(frames_w_cv)
    grid_search_results_wo_cv = pd.concat(frames_wo_cv)

    output_filename_w_cv = os.path.join(
         output_dir, 
         'rf_gs_results_{rn}_w_cv.{test_id}.{date_str}.csv'.format(
            rn=rn, test_id=test_id, date_str=date_str
        )
    )
    output_filename_wo_cv = os.path.join(
         output_dir,
         'rf_gs_results_{rn}_wo_cv.{test_id}.{date_str}.csv'.format(
            rn=rn, 
            test_id=test_id,
            date_str= date_str
        )
    )

    grid_search_results_w_cv.to_csv(output_filename_w_cv)
    grid_search_results_wo_cv.to_csv(output_filename_wo_cv)
    
    return (grid_search_results_w_cv, grid_search_results_wo_cv)

def generate_plots(
        grid_search_results_w_cv:pd.DataFrame, 
        grid_search_results_wo_cv:pd.DataFrame, 
        test_id:int, date_str:int,
        output_dir:str,rn:str
    ):

        palette = itertools.cycle(sns.color_palette())
        c1 = next(palette)
        c2 = next(palette)

        for key in ['max_depth', 'n_estimators', 'max_features', 'max_samples', 'min_samples_split']:
            
            fig = plt.figure()
            ax1 = fig.add_subplot(2, 1, 1)
            ax2 = fig.add_subplot(2, 1, 2, sharex=ax1)

            sns.lineplot(data=grid_search_results_w_cv, x=key, y='mean_neg_mae', color=c1, estimator='mean', errorbar='sd', label='w/ cv', ax=ax1)
            sns.lineplot(data=grid_search_results_wo_cv, x=key, y='mean_neg_mae', color=c2, estimator='mean', errorbar='sd', label='w/o cv', ax=ax2)
            plt.legend(loc='upper right')
            
            fig.suptitle('RF GS Results {rn} - Test {test_id} - {key}'.format(rn=rn, test_id=test_id, key= key))

            output_filename = os.path.join(
                 output_dir, 
                 'rf_gs_results_{rn}_line_plot.{key}.{test_id}.{date_str}.png'.format(
                    key=key, 
                    test_id=test_id,
                    date_str=date_str,
                    rn=rn
                )
            )

            plt.savefig(output_filename)
            plt.clf()

def main():
    args = parse_arguments()
    test_id = args["TestID"]
    date_str = args["DateStr"]
    start_index = args["StartIndex"]
    stop_index = args["StopIndex"]
    step= args["Step"]
    iteration = args["Iteration"]
    input_dir = args["InputDir"]
    output_dir = args["OutputDir"]

    if iteration == 1:
         rn = "r1"
    elif iteration ==2:
         rn = "r2"
    else:
         rn = None
    
    grid_search_results_w_cv, grid_search_results_wo_cv = consolidate_files(
         test_id=test_id, date_str=date_str, start_index=start_index, 
         final_index=stop_index, step=step, input_dir=input_dir, 
         output_dir=output_dir, rn=rn
    )

    generate_plots(
        grid_search_results_w_cv = grid_search_results_w_cv, 
        grid_search_results_wo_cv = grid_search_results_wo_cv, 
        test_id=test_id,
        date_str=date_str,
        output_dir=output_dir,
        rn=rn
    )

print('Starting...')

try:
    main()
except Exception as e:
    print("Execution failed due to the following error:")
    print(e)
    raise e

print('Complete.')

