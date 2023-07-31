rule run_mv_perm_test_v:
    input:
        script = "Scripts/Multivariate/mv_perm_test.py",
        test_plan = EN_TEST_PLAN,
        features = "Output/{project}/ElasticNet/{date_str}/Test{test_id}/mv_qc_fs_bs_candidate_features.{test_id}.{date_str}.csv"
    params:
        date_str = "{date_str}",
        test_id = "{test_id}",
        sigma_cut_off = 2,
        out_dir = "Output/{project}/ElasticNet/{date_str}/Test{test_id}/Validation"
    output:
        results = "Output/{project}/ElasticNet/{date_str}/Test{test_id}/Validation/mv_feature_importance_perm_values_2sig.{test_id}.{date_str}.csv",
        plot = "Output/{project}/ElasticNet/{date_str}/Test{test_id}/Validation/mv_feature_importance_box_plot_2sig.{test_id}.{date_str}.png"
    log:         
        file = "Output/log/mv_perm_test.v.{project}.{test_id}.{date_str}.log.txt"
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
            --DateStr {params.date_str} \
            --SeqSelectionResults {input.features} \
            --SigmaCutOff {params.sigma_cut_off}
            > {log.file}
        """