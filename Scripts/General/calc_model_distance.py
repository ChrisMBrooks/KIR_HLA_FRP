import itertools, re, argparse, os, sys
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
        description="Parallelised model distance calculation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser._action_groups.pop()
    required = parser.add_argument_group("required arguments")

    required.add_argument(
        "-id",
        "--TestCombinationID",
        help="Test Case Combination ID as integer.",
        required=True,
        type=int,
    )

    required.add_argument(
        "-o",
        "--OutputDir",
        help="Output directory path name as string.",
        required=True,
        type=str,
    )
    
    required.add_argument(
        "-p",
        "--ProjectName",
        help="Project name as string.",
        required=True,
        type=str,
    )

    return vars(parser.parse_args())

def get_candidates(project, model, test_id):

    if project == 'KIR_HLA_STUDY_MI': 
        date_str = "13062023"
        path = "~/{project}/KIR_HLA_FRP/Output/{project}/{model}/{date_str}/Test{test_id}".format(
            project=project,
            model=model,
            date_str=date_str,
            test_id=test_id
        )
        
    elif project == 'KIR_HLA_TWINS_MM05':
        date_str = "22062023"
        path = "~/KIR_HLA_STUDY_TWINS_MM_05/KIR_HLA_FRP/Output/{project}/{model}/{date_str}/Test{test_id}".format(
            project = project,
            model=model,
            date_str=date_str,
            test_id=test_id
        )
        
    if model == 'ElasticNet':
        prefix = 'enr2'
        path = os.path.join(path, "ElasticNetR2")
    else:
        prefix = 'rf'

    filename = "{prefix}_feature_importance_perm_values.{test_id}.{date_str}.csv".format(
        prefix=prefix,
        test_id=test_id, 
        date_str=date_str
    )

    filename = os.path.join(path, filename)
    candidates = pd.read_csv(filename, index_col=0).T
    candidates['mean'] = candidates.values.mean(axis=1)
    candidates['std'] = candidates.values.std(axis=1)
    std_thresh = 0 # We want the whole model. 
    candidates['threshold'] = (candidates['mean'] -1*std_thresh*candidates['std']) > 0
    candidates = list(candidates[candidates['threshold'] == True].index)
    return candidates

def get_candidates_summary(project):
    models = ["ElasticNet", "RandomForest"]
    test_ids = range(2, 27, 1)

    records = []
    for model in models:
        for test_id in test_ids:
            candidates = get_candidates(
                project=project, 
                model=model, 
                test_id=test_id
            )
        
            record = [project, model, test_id, len(candidates), candidates]
            records.append(record)
    columns = ['project', 'model', 'test_id', 'num_candidates', 'candidates']
    results = pd.DataFrame(records, columns=columns)
    return results

def export_candidates_summary(candidate_summary:pd.DataFrame):
    filename = "Data/test_plan_candidate_summary_results.parquet"
    candidate_summary.to_parquet(filename)

def get_panel_id(phenotype_id): 
    pattern = '([\w\s]+):' 
    matches = re.findall(pattern=pattern, string=phenotype_id)
    panel_id = matches[0]
    return panel_id

def get_summary_record(pheno_ref_data:pd.DataFrame, candidate:str):
    record = pheno_ref_data[pheno_ref_data['phenotype_id'] == candidate]
    if not record.empty:
        record = record.iloc[0].to_dict()
        record['panel'] = get_panel_id(candidate)
        record['parent'] = record['full_subset_name'].replace(record['marker_posession_def'] + "/", "")
    else:
        record = None
    return record

def get_marker_overlap_frequency(markers1:list, markers2:list):
    if len(markers1) == 0 or len(markers2) == 0:
        overlap_frequency = 0.0
    else:
        all_markers = list(np.unique(markers1 + markers2))
        count = 0.0
        for marker in all_markers:
            if marker in markers1 and marker in markers2:
                count += 1.0
        overlap_frequency = count/len(all_markers)
    return overlap_frequency

def get_correlation(phenotype1:str, phenotype2:str):
    if phenotype1 == phenotype2:
        pearson_corr = 1.0
    else:
        phenotypes = data_sci_mgr.data_mgr.outcomes(fill_na = False, partition = partition)
        phenotype_data1 = phenotypes[['subject_id', phenotype1]]
        phenotype_data2 = phenotypes[['subject_id', phenotype2]]
        phenotype_data = phenotype_data1.merge(phenotype_data2, how='inner', on='subject_id')
        phenotype_data = phenotype_data[[phenotype1, phenotype2]]
        pearson_corr = phenotype_data.corr().values[0,1]
    return pearson_corr

def get_phenotype_distance(pheno_ref_data:pd.DataFrame, candidate:str, alternate:str):
    candidate_record = get_summary_record(pheno_ref_data=pheno_ref_data, candidate=candidate)
    alternate_record = get_summary_record(pheno_ref_data=pheno_ref_data, candidate=alternate)

    if not candidate_record or not alternate_record:
        id_condition = candidate == alternate
        parent_condition = False
        overlap_frequency = 0.0
        pearson_correlation = get_correlation(candidate, alternate)
    else:
        id_condition = candidate_record['phenotype_id'] == alternate_record['phenotype_id']
        parent_condition = candidate_record['parent'] == alternate_record['parent']
        overlap_frequency = get_marker_overlap_frequency(list(candidate_record['relevant_markers']), list(alternate_record['relevant_markers']))
        pearson_correlation = get_correlation(candidate_record['phenotype_id'], alternate_record['phenotype_id'])
    
    score = 0.0
    if id_condition:
        distance = 0.0
    else:
        if parent_condition:
            score += 1.0
        score += overlap_frequency
        score += pearson_correlation
        distance = (3.0 - score)/3.0
    
    distance_record = {
        'candidate':candidate, 'alternate':alternate,
        'distance':distance, 'same_id':id_condition, 
        'same_parent':parent_condition, 'marker_overlap':overlap_frequency, 
        'correlation':pearson_correlation
    }

    return distance_record

def get_phenotype_distances(pheno_ref_data:pd.DataFrame, candidate:str, selection_pool:list):
    distance_records = []
    for alternate in selection_pool:
        record = get_phenotype_distance(
            pheno_ref_data=pheno_ref_data, 
            candidate=candidate, 
            alternate=alternate
        )
        distance_records.append(record)
    
    distance_records = pd.DataFrame(distance_records)
    return distance_records

def find_closest_phenotype(
    pheno_ref_data:pd.DataFrame,
    candidate:str, 
    selection_pool:list
):
    distance_records = get_phenotype_distances(
        pheno_ref_data=pheno_ref_data, 
        candidate=candidate, 
        selection_pool=selection_pool
    )

    distance_records = distance_records.sort_values(by='distance', ascending=True)

    closest_record = distance_records.iloc[0].to_dict()
    return closest_record

def find_distance_between_two_model_candidate_sets(
    pheno_ref_data:pd.DataFrame,
    candidate_pool1:list, 
    candidate_pool2:list
):

    distance = 0.0
    selection_pool = list(candidate_pool2)
    for candidate in candidate_pool1:
        if len(selection_pool) == 0:
            distance +=1
        else:
            closest_record = find_closest_phenotype(
                pheno_ref_data=pheno_ref_data, 
                candidate=candidate, 
                selection_pool=selection_pool
            )
            distance += closest_record['distance']
            selection_pool.remove(closest_record['alternate'])
    
    distance += len(selection_pool)

    return distance

def get_case_permutations(candidate_summary:pd.DataFrame):
    test_cases = range(0, candidate_summary.shape[0], 1)
    cases = set()
    for case in itertools.permutations(test_cases, 2):
        cases.add(tuple(sorted(list(case))))
    return list(cases)

def export_permutations(candidate_summary:pd.DataFrame):
    cases = get_case_permutations(candidate_summary=candidate_summary)
    cases = pd.Series(data=cases, name='test_case')
    filename = 'Data/model_difference_test_cases.csv'
    cases.to_csv(filename)

def get_distance_record_from_two_models(
    pheno_ref_data:pd.DataFrame, 
    candidate_summary:pd.DataFrame, 
    test_cases:pd.DataFrame, 
    index:int
):
    case = test_cases.iloc[index].values[0].split(",")
    case = (int(case[0][1:]), int(case[1][:-1]))
    candidate_pool_1 = candidate_summary.iloc[case[0]]['candidates'] 
    candidate_pool_2 = candidate_summary.iloc[case[1]]['candidates']
    distance = find_distance_between_two_model_candidate_sets(
        pheno_ref_data=pheno_ref_data,
        candidate_pool1=candidate_pool_1, 
        candidate_pool2=candidate_pool_2
    )
    record = {
        "index":index, 
        "test_id_1": candidate_summary.iloc[case[0]]['test_id'], 
        "model_type_1": candidate_summary.iloc[case[0]]['model'], 
        "test_id_2": candidate_summary.iloc[case[1]]['test_id'], 
        "model_type_2": candidate_summary.iloc[case[1]]['model'], 
        "model_distance": distance
    }
    record = pd.Series(record)
    return record

def plot_distances_between_models(pheno_ref_data:pd.DataFrame, candidate_summary:pd.DataFrame):
    cases = get_case_permutations(candidate_summary)
    distance_records = []
    index = 0
    for case in cases:
        candidate_pool_1 = candidate_summary.iloc[case[0]]['candidates'] 
        candidate_pool_2 = candidate_summary.iloc[case[1]]['candidates']
        distance = find_distance_between_two_model_candidate_sets(
            pheno_ref_data=pheno_ref_data,
            candidate_pool1=candidate_pool_1, 
            candidate_pool2=candidate_pool_2
        )
        record = [index, 
                  candidate_summary.iloc[case[0]]['test_id'] , candidate_summary.iloc[case[0]]['model'], 
                  candidate_summary.iloc[case[1]]['test_id'] , candidate_summary.iloc[case[1]]['model'], 
                  distance
        ]
        print(record)
        distance_records.append(record)
        index += 1
    
    columns = ['comparison_id', 'test_id_1', 'model_1', 'test_id_2', 'model_2', 'distance']
    distance_records = pd.DataFrame(distance_records, columns=columns)
    filename = 'Analysis/model_distances_summary_results.csv'
    distance_records.to_csv(filename)
    
    distance_records_en = distance_records[distance_records['model_1'] == 'ElasticNet']
    distance_records_rf = distance_records[distance_records['model_1'] == 'RandomForest']

    palette = itertools.cycle(sns.color_palette())
    c1 = next(palette)
    c2 = next(palette)

    sns.scatterplot(data=distance_records_en, x='test_id_1', y='distance', color=c1, label='RandomForest')
    sns.scatterplot(data=distance_records_rf, x='test_id_1', y='distance', color=c2, label='ElasticNet')

    plt.show()

def main():
    args = parse_arguments()
    test_combo_id = args["TestCombinationID"]
    output_dir = args["OutputDir"]
    project = args["ProjectName"]

    #test_combo_id = 0 #Not a test case, but a tuple test (test1, test2)
    #project = "KIR_HLA_TWINS_MM05"

    # Get Candidate Results Summary
    filename = "Data/test_plan_candidate_summary_results.parquet"
    if not os.path.exists(filename):
        candidates_summary = get_candidates_summary(project=project)
        export_candidates_summary(candidate_summary=candidates_summary)
    
    candidate_summary = pd.read_parquet(filename)

    # Get Phenotype Reference Data
    filename = 'Data/phenotype_marker_reference_data.parquet'
    pheno_ref_data = pd.read_parquet(filename)

    # Get Test ID Combinations
    filename = 'Data/model_difference_test_cases.csv'
    if not os.path.exists(filename):
        test_cases = get_case_permutations(candidate_summary=candidate_summary)
        test_cases = pd.Series(data=test_cases, name='test_case')
        test_cases.to_csv(filename)
    test_cases = pd.read_csv(filename, index_col=0)

    # Evaluate Model Distance
    record = get_distance_record_from_two_models(
        pheno_ref_data=pheno_ref_data, 
        candidate_summary=candidate_summary, 
        test_cases=test_cases, 
        index=test_combo_id
    )

    # Export Model Distance Results
    # Output/SummaryAnalysis/ModelDistance/ParallelResults/
    output_filename = 'test_case_results.{test_combo_id}.csv'.format(test_combo_id=test_combo_id)
    output_filename = os.path.join(output_dir, output_filename)
    record.to_csv(output_filename)

print('Starting..')

#Instantiate Controllers
use_full_dataset=True
use_database=False
project = 'KIR_HLA_TWINS_MM05'
partition = 'training'
data_sci_mgr = dsm.DataScienceManager(
    use_full_dataset=use_full_dataset, 
    use_database=use_database
)

try:
    main()
except Exception as e:
    print("Execution failed due to the following error:")
    print(e)
    
print('Complete.')
