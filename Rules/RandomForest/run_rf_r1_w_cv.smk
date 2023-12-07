rule run_rf_r1_w_cv:
    input:
        script = "Scripts/RandomForest/run_rf_cv.py",
        test_plan = RF_TEST_PLAN,
        h_params = RF_H_PARAMS_R1,
        gs_data = "Output/{project}/RandomForest/{date_str}/Test{test_id}/rf_gs_results_r1_w_cv.{test_id}.{date_str}.csv"
    params:
        date_str = "{date_str}",
        test_id = "{test_id}",
        out_dir = "Output/{project}/RandomForest/{date_str}/Test{test_id}",
        num_jobs = 1
    output:
        summary = "Output/{project}/RandomForest/{date_str}/Test{test_id}/rf_results_r1_w_cv.{test_id}.{date_str}.csv",
        details = "Output/{project}/RandomForest/{date_str}/Test{test_id}/rf_r1_feature_importance_impurity_rankings.{test_id}.{date_str}.csv"
    log:
        file = "Output/log/run_rf_r1_w_cv.{project}.{test_id}.{date_str}.log.txt"
    threads: 1
    resources:
        mem_mb=1000
    conda: "../../Envs/kir_hla_ml_env.yml"
    shell: 
        """
            python {input.script} \
            --TestPlan {input.test_plan} \
            --HyperParams {input.h_params} \
            --TestID {params.test_id} \
            --DateStr {params.date_str} \
            --OutputDir {params.out_dir} \
            --NumJobs {params.num_jobs} \
            > {log.file}
        """
