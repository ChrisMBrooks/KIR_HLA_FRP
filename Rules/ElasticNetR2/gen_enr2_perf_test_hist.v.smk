rule gen_enr2_perf_test_hist_v:
    input:
        script = "Scripts/ElasticNetR2/enr2_plot_perf_perm_test_hist.py",
        test_results = [
            "Output/{{project}}/ElasticNet/{{date_str}}/Test{{test_id}}/ElasticNetR2/ParallelisedData/enr2_model_performance_perm_values.{it_id}.{{test_id}}.{{date_str}}.csv".format(
            it_id=it_id) for it_id in range(0, PERF_PERM_TEST_IT_COUNT, 1)
        ],
        validation_results = "Output/{project}/ElasticNet/{date_str}/Test{test_id}/ElasticNetR2/Validation/enr2_final_score.{test_id}.{date_str}.csv"
    params: 
        date_str = "{date_str}",
        test_id = "{test_id}",
        iter_count = PERF_PERM_TEST_IT_COUNT,
        in_dir = "Output/{project}/ElasticNet/{date_str}/Test{test_id}/ElasticNetR2/ParallelisedData",
        out_dir = "Output/{project}/ElasticNet/{date_str}/Test{test_id}/ElasticNetR2/Validation"
    output:
        values = "Output/{project}/ElasticNet/{date_str}/Test{test_id}/ElasticNetR2/Validation/enr2_model_performance_perm_values.{test_id}.{date_str}.csv",
        plot = "Output/{project}/ElasticNet/{date_str}/Test{test_id}/ElasticNetR2/Validation/enr2_model_performance_perm_hist.{test_id}.{date_str}.png",
        p_val = "Output/{project}/ElasticNet/{date_str}/Test{test_id}/ElasticNetR2/Validation/enr2_model_performance_cdf_results.{test_id}.{date_str}.csv"
    log:         
        file = "Output/log/enr2_plot_perf_perm_test_hist.v.{project}.{test_id}.{date_str}.log.txt"
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