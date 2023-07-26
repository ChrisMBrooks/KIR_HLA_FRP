rule run_mv_perf_perm_test:
    input:
        script = "Scripts/Multivariate/mv_perf_perm_test.py",
        test_plan = EN_TEST_PLAN,
        features = "Output/{project}/ElasticNet/{date_str}/Test{test_id}/mv_qc_fs_bs_candidate_features.{test_id}.{date_str}.csv"
    params:
        date_str = "{date_str}",
        test_id = "{test_id}",
        iteration_id = "{iteration_id}",
        out_dir = "Output/{project}/ElasticNet/{date_str}/Test{test_id}/ParallelisedData"
    output:
        results = "Output/{project}/ElasticNet/{date_str}/Test{test_id}/ParallelisedData/mv_model_performance_perm_values.{iteration_id}.{test_id}.{date_str}.csv",
    log:         
        file = "Output/log/mv_perm_test.{project}.{iteration_id}.{test_id}.{date_str}.log.txt"
    threads: 16
    resources:
        mem_mb=1000
    conda: "../../Envs/kir_hla_ml_env.yml"
    shell: 
        """
            python {input.script} \
            --OutputDir {params.out_dir} \
            --TestID {params.test_id} \
            --TestPlan {input.test_plan} \
            --DateStr {params.date_str} \
            --Input {input.features} \
            --IterationID {params.iteration_id}
            > {log.file}
        """