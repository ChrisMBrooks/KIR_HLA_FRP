rule run_mv_perm_test:
    input:
        script = "Scripts/Multivariate/mv_perm_test.py",
        test_plan = EN_TEST_PLAN,
        features = "Output/{project}/ElasticNet/{date_str}/Test{test_id}/mv_qc_fs_bs_candidate_features.{test_id}.{date_str}.csv"
    params:
        date_str = "{date_str}",
        test_id = "{test_id}",
        out_dir = "Output/{project}/ElasticNet/{date_str}/Test{test_id}"
    output:
        results = "Output/{project}/ElasticNet/{date_str}/Test{test_id}/mv_feature_importance_perm_values.{test_id}.{date_str}.csv",
        plot = "Output/{project}/ElasticNet/{date_str}/Test{test_id}/mv_feature_import_box_plot.{test_id}.{date_str}.png"
    log:         
        file = "Output/log/mv_perm_test.{project}.{test_id}.{date_str}.log.txt"
    threads: 1
    conda: "../../Envs/kir_hla_ml_env.yml"
    shell: 
        """
            python {input.script} \
            --OutputDir {params.out_dir} \
            --TestID {params.test_id} \
            --TestPlan {input.test_plan} \
            --DateStr {params.date_str} \
            --Input {input.features}
            > {log.file}
        """