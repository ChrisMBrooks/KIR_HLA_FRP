rule run_enr2_perf_perm_test_opt:
    input:
        script = "Scripts/ElasticNetR2/run_enr2_perf_perm_test.py",
        test_plan = EN_TEST_PLAN,
        h_params = "Output/{project}/ElasticNet/{date_str}/Test{test_id}/ElasticNetR2/enr2_gs_run_summary_results.{test_id}.{date_str}.csv",
        features = "Output/{project}/ElasticNet/{date_str}/Test{test_id}/ElasticNetR2/enr2_seq_selection_candidates.{test_id}.{date_str}.csv"
    params:
        date_str = "{date_str}",
        test_id = "{test_id}",
        iteration_id = "{iteration_id}",
        out_dir = "Output/{project}/ElasticNet/{date_str}/Test{test_id}/ElasticNetR2/Optimised/ParallelisedData"
    output:
        results = "Output/{project}/ElasticNet/{date_str}/Test{test_id}/ElasticNetR2/Optimised/ParallelisedData/enr2_model_performance_perm_values.optimised.{iteration_id}.{test_id}.{date_str}.csv",
    log:         
        file = "Output/log/enr2_perm_test.{project}.{iteration_id}.{test_id}.{date_str}.log.txt"
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
            --HyperParams {input.h_params} \
            --DateStr {params.date_str} \
            --Input {input.features} \
            --IterationID {params.iteration_id}
            > {log.file}
        """