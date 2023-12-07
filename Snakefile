import json, os, sys
def load_pipeline_config():
    try:
        full_path = os.path.join(os.getcwd(), "pipeline.config.json")
        f = open(full_path)
        config = json.load(f)
        return config
    except Exception as e:
        print('Failed to load pipeline config. Exiting...')
        sys.exit(0)

config = load_pipeline_config()

PROJECT = config["project"]
DATE_STR = config["date_str"]
TEST_IDS = config["test_ids"]
EN_TEST_PLAN = config["en_test_plan_details"]

RF_TEST_PLAN = config["rf_test_plan_details"]
GS_INDEX_MIN = config["gs_index_min"] 
GS_INDEX_MAX = config["gs_index_max"] 
GS_INDEX_STEP = config["gs_index_step"] 
GS_INDECES = range(GS_INDEX_MIN, (GS_INDEX_MAX+1)*GS_INDEX_STEP, GS_INDEX_STEP)
FS_BS_INCLUSION_THRESHOLD = config["fs_bs_inclusion_threshold"]
RF_H_PARAMS_R1 = config["rf_selected_h_params_r1"]
RF_H_PARAMS_R2 = config["rf_selected_h_params_r2"]

PERF_PERM_TEST_IT_COUNT = config["perf_perm_iter_count"]
PERF_PERM_TEST_IT_STEP = config["perf_perm_iter_step"]

rule all_complete:
    input:
        rf_complete = "Output/{project}/RandomForest/{date_str}/rf_complete.{date_str}.txt".format(
            project = PROJECT, 
            date_str = DATE_STR
        ),

        en_complete = "Output/{project}/ElasticNet/{date_str}/en_complete.{date_str}.txt".format(
            project = PROJECT, 
            date_str = DATE_STR
        )

rule rf_complete:
    input:
        perm_results = expand(
            "Output/{project}/RandomForest/{date_str}/Test{test_id}/rf_feature_importance_perm_values.{test_id}.{date_str}.csv",
            project=PROJECT,
            test_id=TEST_IDS,
            date_str=DATE_STR
        ),

        train_test_results = expand(
            "Output/{project}/RandomForest/{date_str}/Test{test_id}/rf_train_test_score.{test_id}.{date_str}.csv",
            project=PROJECT,
            test_id=TEST_IDS,
            date_str=DATE_STR
        ),

        plot_r1 = expand(
            "Output/{project}/RandomForest/{date_str}/Test{test_id}/rf_gs_results_r1_line_plot.min_samples_split.{test_id}.{date_str}.png",
            project=PROJECT,
            test_id=TEST_IDS,
            date_str=DATE_STR
        ),

        plot_r2 = expand(
            "Output/{project}/RandomForest/{date_str}/Test{test_id}/rf_gs_results_r2_line_plot.min_samples_split.{test_id}.{date_str}.png",
            project=PROJECT,
            test_id=TEST_IDS,
            date_str=DATE_STR
        ),

        perf_hist = expand(
            "Output/{project}/RandomForest/{date_str}/Test{test_id}/rf_model_performance_perm_hist.{test_id}.{date_str}.png",
            project=PROJECT,
            test_id=TEST_IDS,
            date_str=DATE_STR
        ),

        validation_results = expand(
            "Output/{project}/RandomForest/{date_str}/Test{test_id}/Validation/rf_final_score.{test_id}.{date_str}.csv",
            project=PROJECT,
            test_id=TEST_IDS,
            date_str=DATE_STR
        ),

        optimised_candidates = expand(
            "Output/{project}/RandomForest/{date_str}/Test{test_id}/Optimised/optimised_model_candidates.{test_id}.{date_str}.csv",
            project=PROJECT,
            test_id=TEST_IDS,
            date_str=DATE_STR
        ),

        optimised_trn_test_results = expand(
            "Output/{project}/RandomForest/{date_str}/Test{test_id}/Optimised/rf_train_test_score.optimised.{test_id}.{date_str}.csv",
            project=PROJECT,
            test_id=TEST_IDS,
            date_str=DATE_STR
        ),

        univar_scores_v = expand(
            "Output/{project}/RandomForest/{date_str}/Test{test_id}/Validation/univar_final_scores.{test_id}.{date_str}.csv",
            project=PROJECT,
            test_id=TEST_IDS,
            date_str=DATE_STR
        ),

        perf_hist_v = expand(
            "Output/{project}/RandomForest/{date_str}/Test{test_id}/Validation/rf_model_performance_perm_hist.{test_id}.{date_str}.png",
            project=PROJECT,
            test_id=TEST_IDS,
            date_str=DATE_STR
        ),

        univar_scores_opt = expand(
            "Output/{project}/RandomForest/{date_str}/Test{test_id}/Optimised/univar_final_scores.optimised.{test_id}.{date_str}.csv",
            project=PROJECT,
            test_id=TEST_IDS,
            date_str=DATE_STR
        ),

        perf_hist_opt = expand(
            "Output/{project}/RandomForest/{date_str}/Test{test_id}/Optimised/rf_model_performance_perm_hist.optimised.{test_id}.{date_str}.png",
            project=PROJECT,
            test_id=TEST_IDS,
            date_str=DATE_STR
        )

    output:
        file = "Output/{project}/RandomForest/{date_str}/rf_complete.{date_str}.txt"
    params:
        project = "{project}",
        date_str = "{date_str}"
    shell:
        """
            echo rf_complete! > Output/{params.project}/RandomForest/{params.date_str}/rf_complete.{params.date_str}.txt
        """

rule en_complete:
    input: 
        perm_results = expand(
            "Output/{project}/ElasticNet/{date_str}/Test{test_id}/mv_feature_importance_perm_values.{test_id}.{date_str}.csv",
            project=PROJECT,
            date_str=DATE_STR,
            test_id=TEST_IDS
        ),

        train_test_results = expand(
            "Output/{project}/ElasticNet/{date_str}/Test{test_id}/mv_train_test_score.{test_id}.{date_str}.csv",
            project=PROJECT,
            date_str=DATE_STR,
            test_id=TEST_IDS
        ),

        alpha_plot = expand("Output/{project}/ElasticNet/{date_str}/Test{test_id}/en_gs_results_line_plot.{h_param}.{test_id}.{date_str}.png",
            project=PROJECT,
            h_param="alpha",
            test_id=TEST_IDS,
            date_str=DATE_STR
        ),

        l1_plot = expand("Output/{project}/ElasticNet/{date_str}/Test{test_id}/en_gs_results_line_plot.{h_param}.{test_id}.{date_str}.png",
            project=PROJECT,
            h_param="l1_ratio",
            test_id=TEST_IDS,
            date_str=DATE_STR
        ),

        perf_hist = expand(
            "Output/{project}/ElasticNet/{date_str}/Test{test_id}/mv_model_performance_perm_hist.{test_id}.{date_str}.png",
            project=PROJECT,
            test_id=TEST_IDS,
            date_str=DATE_STR
        ),

        validation_results = expand(
            "Output/{project}/ElasticNet/{date_str}/Test{test_id}/Validation/mv_final_score.{test_id}.{date_str}.csv",
            project=PROJECT,
            test_id=TEST_IDS,
            date_str=DATE_STR
        ),

        optimised_candidates = expand(
            "Output/{project}/ElasticNet/{date_str}/Test{test_id}/Optimised/mv_train_test_score.optimised.{test_id}.{date_str}.csv",
            project=PROJECT,
            test_id=TEST_IDS,
            date_str=DATE_STR
        ),

        optimised_train_test_results = expand(
            "Output/{project}/ElasticNet/{date_str}/Test{test_id}/Optimised/optimised_model_candidates.{test_id}.{date_str}.csv",
            project=PROJECT,
            test_id=TEST_IDS,
            date_str=DATE_STR
        ),

        univar_scores_v = expand(
            "Output/{project}/ElasticNet/{date_str}/Test{test_id}/Validation/univar_final_scores.{test_id}.{date_str}.csv",
            project=PROJECT,
            test_id=TEST_IDS,
            date_str=DATE_STR
        ),

        perf_hist_v = expand(
            "Output/{project}/ElasticNet/{date_str}/Test{test_id}/Validation/mv_model_performance_perm_hist.{test_id}.{date_str}.png",
            project=PROJECT,
            test_id=TEST_IDS,
            date_str=DATE_STR
        ),

        univar_scores_opt = expand(
            "Output/{project}/ElasticNet/{date_str}/Test{test_id}/Optimised/univar_final_scores.optimised.{test_id}.{date_str}.csv",
            project=PROJECT,
            test_id=TEST_IDS,
            date_str=DATE_STR
        ),

        perf_hist_opt = expand(
            "Output/{project}/ElasticNet/{date_str}/Test{test_id}/Optimised/mv_model_performance_perm_hist.optimised.{test_id}.{date_str}.png",
            project=PROJECT,
            test_id=TEST_IDS,
            date_str=DATE_STR
        ),
        
        enr2_complete = "Output/{project}/ElasticNet/{date_str}/enr2_complete.{date_str}.txt".format(
            project = PROJECT, 
            date_str = DATE_STR
        )

    output:
        file = "Output/{project}/ElasticNet/{date_str}/en_complete.{date_str}.txt"
    params:
        project = "{project}",
        date_str = "{date_str}"
    shell:
        """
            echo en_complete! > Output/{params.project}/ElasticNet/{params.date_str}/en_complete.{params.date_str}.txt
        """
rule enr2_complete:
    input:
        perm_results = expand(
            "Output/{project}/ElasticNet/{date_str}/Test{test_id}/ElasticNetR2/enr2_feature_importance_perm_values.{test_id}.{date_str}.csv",
            project=PROJECT,
            test_id=TEST_IDS,
            date_str=DATE_STR
        ),

        train_test_results = expand(
            "Output/{project}/ElasticNet/{date_str}/Test{test_id}/ElasticNetR2/enr2_train_test_score.{test_id}.{date_str}.csv",
            project=PROJECT,
            test_id=TEST_IDS,
            date_str=DATE_STR
        ),

        plot_r2 = expand(
            "Output/{project}/ElasticNet/{date_str}/Test{test_id}/ElasticNetR2/enr2_gs_results_line_plot.alpha.{test_id}.{date_str}.png",
            project=PROJECT,
            test_id=TEST_IDS,
            date_str=DATE_STR
        ),

        perf_hist = expand(
            "Output/{project}/ElasticNet/{date_str}/Test{test_id}/ElasticNetR2/enr2_model_performance_perm_hist.{test_id}.{date_str}.png",
            project=PROJECT,
            test_id=TEST_IDS,
            date_str=DATE_STR
        ),

        validation_results = expand(
            "Output/{project}/ElasticNet/{date_str}/Test{test_id}/ElasticNetR2/Validation/enr2_final_score.{test_id}.{date_str}.csv",
            project=PROJECT,
            test_id=TEST_IDS,
            date_str=DATE_STR
        ),

        optimised_candidates = expand(
            "Output/{project}/ElasticNet/{date_str}/Test{test_id}/ElasticNetR2/Optimised/optimised_model_candidates.{test_id}.{date_str}.csv",
            project=PROJECT,
            test_id=TEST_IDS,
            date_str=DATE_STR
        ),

        optimised_trn_test_results = expand(
            "Output/{project}/ElasticNet/{date_str}/Test{test_id}/ElasticNetR2/Optimised/enr2_train_test_score.optimised.{test_id}.{date_str}.csv",
            project=PROJECT,
            test_id=TEST_IDS,
            date_str=DATE_STR
        ),

        univar_scores_v = expand(
            "Output/{project}/ElasticNet/{date_str}/Test{test_id}/ElasticNetR2/Validation/univar_final_scores.{test_id}.{date_str}.csv",
            project=PROJECT,
            test_id=TEST_IDS,
            date_str=DATE_STR
        ),

        perf_hist_v = expand(
            "Output/{project}/ElasticNet/{date_str}/Test{test_id}/ElasticNetR2/Validation/enr2_model_performance_perm_hist.{test_id}.{date_str}.png",
            project=PROJECT,
            test_id=TEST_IDS,
            date_str=DATE_STR
        ),

        univar_scores_opt = expand(
            "Output/{project}/ElasticNet/{date_str}/Test{test_id}/ElasticNetR2/Optimised/univar_final_scores.optimised.{test_id}.{date_str}.csv",
            project=PROJECT,
            test_id=TEST_IDS,
            date_str=DATE_STR
        ),

        perf_hist_opt = expand(
            "Output/{project}/ElasticNet/{date_str}/Test{test_id}/ElasticNetR2/Optimised/enr2_model_performance_perm_hist.optimised.{test_id}.{date_str}.png",
            project=PROJECT,
            test_id=TEST_IDS,
            date_str=DATE_STR
        )
        
    output:
        file = "Output/{project}/ElasticNet/{date_str}/enr2_complete.{date_str}.txt"
    params:
        project = "{project}",
        date_str = "{date_str}"
    shell:
        """
            echo enr2_complete! > Output/{params.project}/ElasticNet/{params.date_str}/enr2_complete.{params.date_str}.txt
        """
#Random Forest Rules
include: "Rules/RandomForest/get_rf_h_params_combinations.smk"
include: "Rules/RandomForest/run_rf_gs_r1_parallel.smk"
include: "Rules/RandomForest/run_rf_gs_r2_parallel.smk"
include: "Rules/RandomForest/run_rf_gs_r1_wo_cv_parallel.smk"
include: "Rules/RandomForest/run_rf_gs_r2_wo_cv_parallel.smk"
include: "Rules/RandomForest/gen_rf_gs_results_r1_line_plot.smk"
include: "Rules/RandomForest/gen_rf_gs_results_r2_line_plot.smk"
include: "Rules/RandomForest/run_rf_r1_w_cv.smk"
include: "Rules/RandomForest/run_rf_qc_fs_bs.smk"
include: "Rules/RandomForest/run_rf_perm_test.smk"

include: "Rules/RandomForest/run_rf_train_test.smk"
include: "Rules/RandomForest/run_rf_perf_perm_test.smk"
include: "Rules/RandomForest/gen_rf_perf_test_hist.smk"
include: "Rules/RandomForest/get_rf_optimised_features.smk"

include: "Rules/RandomForest/gen_rf_perf_test_hist.opt.smk"
include: "Rules/RandomForest/run_rf_meff_calc.opt.smk"
include: "Rules/RandomForest/run_rf_perf_perm_test.opt.smk"
include: "Rules/RandomForest/run_rf_perm_test.opt.smk"
include: "Rules/RandomForest/run_rf_univar_validation.opt.smk"
include: "Rules/RandomForest/run_rf_validation.opt.smk"

include: "Rules/RandomForest/run_rf_validation.smk"
include: "Rules/RandomForest/run_rf_train_test.opt.smk"

include: "Rules/RandomForest/run_rf_perm_test.v.smk"
include: "Rules/RandomForest/run_rf_meff_calc.smk"
include: "Rules/RandomForest/gen_rf_perf_test_hist.v.smk"
include: "Rules/RandomForest/run_rf_univar_validation.v.smk"

#ElasticNet Rules
include: "Rules/ElasticNet/gen_en_gs_line_plots.smk"
include: "Rules/ElasticNet/get_en_h_params_combinations.smk"
include: "Rules/ElasticNet/run_en_gs.smk"
include: "Rules/ElasticNet/run_en_gs_wo_cv.smk"
include: "Rules/ElasticNet/run_mv_perm_test.smk"
include: "Rules/ElasticNet/run_mv_qc_fs_bs.smk"

include: "Rules/ElasticNet/run_mv_train_test.smk"
include: "Rules/ElasticNet/run_mv_perf_perm_test.smk"
include: "Rules/ElasticNet/gen_mv_perf_test_hist.smk"
include: "Rules/ElasticNet/get_mv_optimised_features.smk"

include: "Rules/ElasticNet/gen_mv_perf_test_hist.opt.smk"
include: "Rules/ElasticNet/run_mv_meff_calc.opt.smk"
include: "Rules/ElasticNet/run_mv_perf_perm_test.opt.smk"
include: "Rules/ElasticNet/run_mv_perm_test.opt.smk"
include: "Rules/ElasticNet/run_mv_univar_validation.opt.smk"
include: "Rules/ElasticNet/run_mv_validation.opt.smk"

include: "Rules/ElasticNet/run_mv_validation.smk"
include: "Rules/ElasticNet/run_mv_train_test.opt.smk"

include: "Rules/ElasticNet/run_mv_perm_test.v.smk"
include: "Rules/ElasticNet/run_mv_meff_calc.smk"
include: "Rules/ElasticNet/gen_mv_perf_test_hist.v.smk"
include: "Rules/ElasticNet/run_mv_univar_validation.v.smk"

#ElasticNetR2 Rules
include: "Rules/ElasticNetR2/gen_enr2_gs_line_plots.smk"
include: "Rules/ElasticNetR2/gen_enr2_perf_test_hist.opt.smk"
include: "Rules/ElasticNetR2/gen_enr2_perf_test_hist.smk"
include: "Rules/ElasticNetR2/gen_enr2_perf_test_hist.v.smk"
include: "Rules/ElasticNetR2/get_enr2_optimised_features.smk"
include: "Rules/ElasticNetR2/run_enr2_gs_wo_cv.smk"
include: "Rules/ElasticNetR2/run_enr2_gs.smk"
include: "Rules/ElasticNetR2/run_enr2_meff_calc.opt.smk"
include: "Rules/ElasticNetR2/run_enr2_meff_calc.smk"
include: "Rules/ElasticNetR2/run_enr2_perf_perm_test.opt.smk"
include: "Rules/ElasticNetR2/run_enr2_perf_perm_test.smk"
include: "Rules/ElasticNetR2/run_enr2_perm_test.opt.smk"
include: "Rules/ElasticNetR2/run_enr2_perm_test.smk"
include: "Rules/ElasticNetR2/run_enr2_perm_test.v.smk"
include: "Rules/ElasticNetR2/run_enr2_qc_fs_bs.smk"
include: "Rules/ElasticNetR2/run_enr2_train_test.opt.smk"
include: "Rules/ElasticNetR2/run_enr2_train_test.smk"
include: "Rules/ElasticNetR2/run_enr2_univar_validation.opt.smk"
include: "Rules/ElasticNetR2/run_enr2_univar_validation.v.smk"
include: "Rules/ElasticNetR2/run_enr2_validation.opt.smk"
include: "Rules/ElasticNetR2/run_enr2_validation.smk"