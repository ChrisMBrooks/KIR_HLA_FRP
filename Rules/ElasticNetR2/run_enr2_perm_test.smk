rule run_enr2_perm_test:
    input:
        script = "Scripts/ElasticNetR2/run_enr2_perm_test.py",
        test_plan = EN_TEST_PLAN,
        h_params = "Output/{project}/ElasticNet/{date_str}/Test{test_id}/ElasticNetR2/enr2_gs_run_summary_results.{test_id}.{date_str}.csv",
        features = "Output/{project}/ElasticNet/{date_str}/Test{test_id}/ElasticNetR2/enr2_seq_selection_candidates.{test_id}.{date_str}.csv"
    params:
        date_str = "{date_str}",
        test_id = "{test_id}",
        sigma_cut_off = 0,
        out_dir = "Output/{project}/ElasticNet/{date_str}/Test{test_id}/ElasticNetR2"
    output:
        results = "Output/{project}/ElasticNet/{date_str}/Test{test_id}/ElasticNetR2/enr2_feature_importance_perm_values.{test_id}.{date_str}.csv",
        plot = "Output/{project}/ElasticNet/{date_str}/Test{test_id}/ElasticNetR2/enr2_feature_importance_box_plot.{test_id}.{date_str}.png"
    log:         
        file = "Output/log/enr2_perm_test.{project}.{test_id}.{date_str}.log.txt"
    threads: 1
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
            --SeqSelectionResults {input.features} \
            --SigmaCutOff {params.sigma_cut_off}
            > {log.file}
        """