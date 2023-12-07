rule run_rf_perf_perm_test_opt:
    input:
        script = "Scripts/RandomForest/rf_perf_perm_test.py",
        test_plan = RF_TEST_PLAN,
        h_params = RF_H_PARAMS_R2,
        features = "Output/{project}/RandomForest/{date_str}/Test{test_id}/Optimised/optimised_model_candidates.{test_id}.{date_str}.csv"
    params:
        date_str = "{date_str}",
        test_id = "{test_id}",
        iteration_id = "{iteration_id}",
        iteration_step = PERF_PERM_TEST_IT_STEP,
        out_dir = "Output/{project}/RandomForest/{date_str}/Test{test_id}/Optimised/ParallelisedData",
        n_jobs = 15
    output:
        results = "Output/{project}/RandomForest/{date_str}/Test{test_id}/Optimised/ParallelisedData/rf_model_performance_perm_values.optimised.{iteration_id}.{test_id}.{date_str}.csv",
    log:         
        file = "Output/log/rf_perm_test.{project}.{iteration_id}.{test_id}.{date_str}.log.txt"
    threads: 16
    resources:
        mem_mb=1000
    conda: "../../Envs/kir_hla_ml_env.yml"
    shell: 
        """
            python {input.script} \
            --TestID {params.test_id} \
            --TestPlan {input.test_plan} \
            --HyperParams {input.h_params} \
            --Input {input.features} \
            --IterationID {params.iteration_id} \
            --IterationStep {params.iteration_step} \
            --DateStr {params.date_str} \
            --OutputDir {params.out_dir} \
            --NumJobs {params.n_jobs} \
            > {log.file}
        """