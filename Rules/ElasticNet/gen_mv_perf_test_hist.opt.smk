rule gen_mv_perf_test_hist_opt:
    input:
        script = "Scripts/Multivariate/mv_plot_perf_perm_test_hist.py",
        test_results = [
            "Output/{{project}}/ElasticNet/{{date_str}}/Test{{test_id}}/Optimised/ParallelisedData/mv_model_performance_perm_values.optimised.{it_id}.{{test_id}}.{{date_str}}.csv".format(
            it_id=it_id) for it_id in range(0, PERF_PERM_TEST_IT_COUNT, 1)
        ],
        validation_results = "Output/{project}/ElasticNet/{date_str}/Test{test_id}/Optimised/mv_final_score.optimised.{test_id}.{date_str}.csv"
    params: 
        date_str = "{date_str}",
        test_id = "{test_id}",
        iter_count = PERF_PERM_TEST_IT_COUNT,
        in_dir = "Output/{project}/ElasticNet/{date_str}/Test{test_id}/Optimised/ParallelisedData",
        out_dir = "Output/{project}/ElasticNet/{date_str}/Test{test_id}/Optimised"
    output:
        values = "Output/{project}/ElasticNet/{date_str}/Test{test_id}/Optimised/mv_model_performance_perm_values.optimised.{test_id}.{date_str}.csv",
        plot = "Output/{project}/ElasticNet/{date_str}/Test{test_id}/Optimised/mv_model_performance_perm_hist.optimised.{test_id}.{date_str}.png",
        p_val = "Output/{project}/ElasticNet/{date_str}/Test{test_id}/Optimised/mv_model_performance_cdf_results.optimised.{test_id}.{date_str}.csv"
    log:         
        file = "Output/log/mv_plot_perf_perm_test_hist.optimised.{project}.{test_id}.{date_str}.log.txt"
    threads: 1
    resources:
        mem_mb=1000
    conda: "../../Envs/kir_hla_ml_env.yml"
    shell: 
        """
            python {input.script} \
            --TestID {params.test_id} \
            --InputDir {params.in_dir} \
            --IterationCount {params.iter_count} \
            --ModelResults {input.validation_results} \
            --DateStr {params.date_str} \
            --OutputDir {params.out_dir} \
            > {log.file}
        """