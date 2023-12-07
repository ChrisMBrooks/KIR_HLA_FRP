import numpy as np
import pandas as pd
import time, uuid, sys, os, argparse

from sklearn.linear_model import ElasticNet
from sklearn.model_selection import RepeatedKFold
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.metrics import mean_absolute_error

from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from Controllers.DataScienceManager import DataScienceManager as dsm

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Globale ElasticNet FRP Permutation Test",
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
        "-hp1",
        "--HyperParamsR1",
        help="Selected hyper parameters r1 filename.",
        required=True,
        type=str,
    )

    required.add_argument(
        "-hp2",
        "--HyperParamsR2",
        help="Selected hyper parameters r2 filename.",
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
        "-it",
        "--IterationID",
        help="Iteration ID as int",
        required=True,
        type=int,
    )

    required.add_argument(
        "-o",
        "--OutputDir",
        help="Output Directory",
        required=True,
        type=str,
    )

    required.add_argument(
        "-j",
        "--NumJobs",
        help="Number of jobs to run in paralle.",
        required=True,
        type=int,
    )
    
    return vars(parser.parse_args())

def load_hyperparameters_en(h_params_filename:str):
    hp_frame = pd.read_csv(h_params_filename, index_col=0)
    alpha = float(hp_frame.loc["alpha"][0])
    l1_ratio = float(hp_frame.loc["l1_ratio"][0])

    return alpha, l1_ratio

def get_output_filename(test_id:int, date_str:str, 
    iteration_id:int, output_dir:str, 
):
    output_filename = os.path.join(
        output_dir, 
        'enr2_shuffled_train_validate_score.{iteration_id}.{test_id}.{date_str}.csv'.format(
            test_id = test_id, 
            date_str=date_str,
            iteration_id=iteration_id
        )
    )

    return output_filename

def preprocess_data(phenos:pd.DataFrame, scores:pd.DataFrame, 
        dependent_var:str, data_sci_mgr:object, 
        impute:bool, standardise:bool, 
        normalise:bool, strategy:str
):
    scores = scores[dependent_var].values.reshape(-1,1)
    if len(phenos.values.shape) == 1:
        phenos = phenos.values.reshape(-1,1)
    else:
        phenos = phenos.values[:, 0:]

    phenos, scores = data_sci_mgr.data_mgr.preprocess_data(
        X=phenos, Y=scores, impute = impute, standardise = standardise, 
        normalise = normalise, strategy=strategy
    )

    scores = scores.ravel()

    return phenos.astype(float), scores.astype(float)

def shuffle_dependent_var(scores:np.array, seed:int):
    shuffled_scores = np.copy(scores)
    np.random.seed(seed=seed)
    np.random.shuffle(shuffled_scores)
    return shuffled_scores

def get_feature_subset_en_r1(
    phenos:np.array, scores:np.array, 
    alpha:float, l1_ratio:float, 
    phenos_subset:list
):
    model = ElasticNet(alpha=alpha, 
        l1_ratio=l1_ratio
    )

    n_splits = 5
    n_repeats = 4
    random_state_2 = 21


    cv = RepeatedKFold(
        n_splits=n_splits, 
        n_repeats=n_repeats, 
        random_state=random_state_2
    )

    splits_gen = cv.split(phenos)

    results = []
    feature_coefs = pd.DataFrame()
    for i in range(0, n_repeats+1):
        split = next(splits_gen)
        train_indeces = split[0]
        test_indeces = split[1]

        model.fit(phenos[train_indeces, :], scores[train_indeces])

        name = 'coef_{}'.format(i)
        if feature_coefs.empty:
            feature_coefs = pd.DataFrame(model.coef_, index=phenos_subset, columns=[name])
        else:
            feature_coefs[name] = model.coef_

        # Computer Predictions and Summary Stats
        y_hat = model.predict(phenos[test_indeces, :])
        neg_mae = -1*mean_absolute_error(scores[test_indeces], y_hat)
        results.append(neg_mae)

    feature_coefs['mean'] = feature_coefs.values.mean(axis=1)
    feature_coefs = feature_coefs[feature_coefs['mean'] > 0].copy()
    feature_coefs['abs(mean)'] = np.abs(feature_coefs['mean'])
    feature_coefs = feature_coefs.sort_values(by='abs(mean)', ascending=False)
    revised_phenos_subset = list(feature_coefs.loc[np.count_nonzero(feature_coefs.values[:, 0:5], axis=1) >= 3].index)

    return revised_phenos_subset

def get_feature_subset_fs_bs_en_r2(
    phenos:np.array, scores:np.array, 
    phenos_subset:list, 
    alpha:float, l1_ratio:float, 
    scoring:str, fs_bs_filter:int,
    n_jobs:int
):

    n_splits = 4
    n_repeats = 10
    random_state = 42
    tolerance = None

    cv = RepeatedKFold(
        n_splits=n_splits, 
        n_repeats=n_repeats, 
        random_state=random_state
    )

    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)

    sfs_for = SequentialFeatureSelector(
            model, direction='forward', 
            n_features_to_select='auto', 
            scoring=scoring, 
            tol=tolerance, cv=cv, n_jobs=n_jobs
    )

    sfs_for.fit(phenos, scores)
    for_selected_features = sfs_for.get_support()

    sfs_bac = SequentialFeatureSelector(
        model, direction='backward', 
        n_features_to_select='auto', 
        scoring=scoring, 
        tol=tolerance, cv=cv, n_jobs=n_jobs
    )
    sfs_bac.fit(phenos, scores)
    bac_selected_features = sfs_bac.get_support()

    summary = [[phenos_subset[i], for_selected_features[i], bac_selected_features[i]] for i in range(0, len(phenos_subset), 1)]
    summary_df = pd.DataFrame(summary, columns=['label', 'forward_selected', 'backward_selected'])

    indeces = summary_df.values[:,1:3].sum(axis=1)
    indeces = np.where(indeces >= fs_bs_filter)
    reduced_phenos_subset = list(summary_df.iloc[indeces]['label'].values)
    return reduced_phenos_subset

def partition_training_data(
    phenos:np.array, scores:np.array, 
    n_splits:float, n_repeats:int = 1, ith_repeat:int = 1, 
    iteration_id:int = 42
):

    random_state = int(np.random.random_sample(size=(1,1))[0,0]*iteration_id)

    cv = RepeatedKFold(
        n_splits=n_splits, 
        n_repeats=n_repeats, 
        random_state=random_state
    )

    splits_gen = cv.split(phenos)
        
    split = None
    for i in range(0, ith_repeat,1):
        split = next(splits_gen)
        
    train_indeces = split[0]
    test_indeces = split[1]

    phenos_t = np.take(phenos, indices=train_indeces,axis=0)
    scores_t = np.take(scores, indices=train_indeces,axis=0)

    phenos_v = np.take(phenos, indices=test_indeces,axis=0)
    scores_v = np.take(scores, indices=test_indeces,axis=0)

    return phenos_t, scores_t, phenos_v, scores_v

def get_train_validation_score_en_r2(
    phenos_t:np.array,
    scores_t:np.array,
    phenos_v:np.array,
    scores_v:np.array,
    alpha:float, 
    l1_ratio:float
):

    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)

    model.fit(phenos_t, scores_t)
        
    # Computer Predictions and Summary Stats
    y_hat = model.predict(phenos_v)
    neg_mae = -1*mean_absolute_error(scores_v, y_hat)

    return neg_mae

#Instantiate Controllers
use_full_dataset = True
use_database = False

data_sci_mgr = dsm.DataScienceManager(
    use_full_dataset=use_full_dataset,
    use_database=use_database
)

dependent_var = 'f_kir_score'
scoring = 'neg_mean_absolute_error'
fs_bs_filter = 2

# Parse Input Params
args = parse_arguments()
test_id = args["TestID"]
test_plan_filename = args["TestPlan"]
h_params_filename_r1 = args["HyperParamsR1"]
h_params_filename_r2 = args["HyperParamsR2"]
date_str = args["DateStr"]
iteration_id = args["IterationID"]
output_dir = args["OutputDir"]
n_jobs = args["NumJobs"]

# Load and Parse Ancillary Inputs
test_plan_atts = pd.read_csv(test_plan_filename, index_col=None)
test_plan_atts = test_plan_atts[test_plan_atts['test_id'] == test_id].copy()

source_filename = test_plan_atts['source'].values[0]
phenos_subset_r0 = list(pd.read_csv(source_filename, index_col=0).values[:, 0])

impute = bool(test_plan_atts['impute'].values[0])
strategy = test_plan_atts['strategy'].values[0]
normalise = bool(test_plan_atts['normalise'].values[0])
standardise = bool(test_plan_atts['standardise'].values[0])

alpha_r1, l1_ratio_r1 = load_hyperparameters_en(h_params_filename_r1)
alpha_r2, l1_ratio_r2 = load_hyperparameters_en(h_params_filename_r2)

# Load Data
phenos = data_sci_mgr.data_mgr.outcomes(fill_na=False, fill_na_value=None, partition='training')
scores = data_sci_mgr.data_mgr.features(fill_na=False, fill_na_value=None, partition='training')
phenos = phenos[phenos_subset_r0]

phenos, scores = preprocess_data(phenos=phenos, scores=scores, 
    dependent_var=dependent_var, data_sci_mgr=data_sci_mgr, 
    impute=impute, standardise=standardise, 
    normalise=normalise, strategy=strategy
)

# Shuffle Dependent Variable
shuffled_scores = shuffle_dependent_var(scores, seed=iteration_id)

# Perform First Feature Reduction
phenos_subset_r1 = get_feature_subset_en_r1(
    phenos = phenos, 
    scores = shuffled_scores, 
    alpha = alpha_r1, 
    l1_ratio = l1_ratio_r1, 
    phenos_subset = phenos_subset_r0
)

# Load Reduced Featureset & Data
phenos_r1 = data_sci_mgr.data_mgr.outcomes(fill_na=False, fill_na_value=None, partition='training')
scores_r1 = data_sci_mgr.data_mgr.features(fill_na=False, fill_na_value=None, partition='training')
phenos_r1 = phenos_r1[phenos_subset_r1]

# scores_r1 is not used, will continue to use shuffled_scores
phenos_r1, scores_r1 = preprocess_data(phenos=phenos_r1, scores=scores_r1, 
    dependent_var=dependent_var, data_sci_mgr=data_sci_mgr, 
    impute=impute, standardise=standardise, 
    normalise=normalise, strategy=strategy
)

phenos_subset_r2 = get_feature_subset_fs_bs_en_r2(
    phenos=phenos_r1, scores=shuffled_scores, 
    phenos_subset=phenos_subset_r1, alpha=alpha_r1, 
    l1_ratio=l1_ratio_r1, scoring=scoring, 
    fs_bs_filter=fs_bs_filter, n_jobs=n_jobs
)

# Load Reduced Featureset & Data
phenos_r2 = data_sci_mgr.data_mgr.outcomes(fill_na=False, fill_na_value=None, partition='training')
scores_r2 = data_sci_mgr.data_mgr.features(fill_na=False, fill_na_value=None, partition='training')
phenos_r2 = phenos_r2[phenos_subset_r2]

# Load Reduced Featureset & Data - Validation
phenos_r2_v= data_sci_mgr.data_mgr.outcomes(fill_na=False, fill_na_value=None, partition='validation')
scores_r2_v = data_sci_mgr.data_mgr.features(fill_na=False, fill_na_value=None, partition='validation')
phenos_r2_v = phenos_r2_v[phenos_subset_r2]

# scores_r2 is not used, will continue to use shuffled_scores
phenos_r2, scores_r2 = preprocess_data(phenos=phenos_r2, scores=scores_r2, 
    dependent_var=dependent_var, data_sci_mgr=data_sci_mgr, 
    impute=impute, standardise=standardise, 
    normalise=normalise, strategy=strategy
)

# scores_r2 is not used, will continue to use shuffled_scores - Validation
phenos_r2_v, scores_r2_v = preprocess_data(phenos=phenos_r2_v, scores=scores_r2_v, 
    dependent_var=dependent_var, data_sci_mgr=data_sci_mgr, 
    impute=impute, standardise=standardise, 
    normalise=normalise, strategy=strategy
)
shuffled_scores_v = shuffle_dependent_var(scores_r2_v, seed=iteration_id * iteration_id)

# Get Train Tesst Score
train_validate_score = get_train_validation_score_en_r2(
    phenos_t=phenos_r2, 
    scores_t=shuffled_scores,
    phenos_v=phenos_r2_v,
    scores_v=shuffled_scores_v, 
    alpha=alpha_r2, 
    l1_ratio=l1_ratio_r2
)

# Export Summary
output_filename = get_output_filename(
    test_id=test_id, 
    date_str=date_str, 
    iteration_id=iteration_id,
    output_dir=output_dir
)

output = {}
output['data_source'] = source_filename
output['test_plan'] = test_plan_filename
output['h_parmas_r1'] = h_params_filename_r1
output['h_parmas_r2'] = h_params_filename_r2
output['test_id'] = test_id
output['dependent_var'] = "shuffled_"+dependent_var
output['partition_training_dataset'] = False
output['fs_bs_filter'] = fs_bs_filter
output['impute'] = impute
output['standardise'] = standardise
output['normalise'] = normalise
output['strategy'] = strategy
output['feautres'] = phenos_subset_r2
output['features_count'] = len(phenos_subset_r2)
output['mean_train_test_score'] = train_validate_score

output = pd.Series(output)
print(output)
output.to_csv(output_filename)
