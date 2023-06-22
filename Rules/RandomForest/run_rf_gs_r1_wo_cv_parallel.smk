rule run_rf_gs_r1_wo_cv_parallel:
    input:
        script = "Scripts/RandomForest/run_rf_parallel_gs_no_cv_r1.py",
        test_plan = RF_TEST_PLAN,
        h_params = "Output/{project}/RandomForest/MetaData/rf_gs_h_parmas.{date_str}.csv"
    params:
        date_str = "{date_str}",
        test_id = "{test_id}",
        start_index = "{start_index}",
        step = "{step}",
        out_dir = "Output/{project}/RandomForest/{date_str}/Test{test_id}/ParallelisedData",
        num_jobs = 1
    output:
        result = "Output/{project}/RandomForest/{date_str}/Test{test_id}/ParallelisedData/rf_parallel_gs_results_r1_wo_cv.{start_index}.{step}.{test_id}.{date_str}.csv"
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
            --StartIndex {params.start_index} \
            --Step {params.step} \
            --OutputDir {params.out_dir} \
            --NumJobs {params.num_jobs} \
        """