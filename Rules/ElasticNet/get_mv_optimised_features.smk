rule get_mv_optimised_features:
    input:
        script = "Scripts/General/optimise_model_candidates.py",
        test_plan = EN_TEST_PLAN,
        fs_bs_results = "Output/{project}/ElasticNet/{date_str}/Test{test_id}/mv_qc_fs_bs_candidate_features.{test_id}.{date_str}.csv"
    params:
        date_str = "{date_str}",
        test_id = "{test_id}",
        out_dir = "Output/{project}/ElasticNet/{date_str}/Test{test_id}/Optimised",
    output:
        results = "Output/{project}/ElasticNet/{date_str}/Test{test_id}/Optimised/optimised_model_candidates.{test_id}.{date_str}.csv"
    log:
        file =  "Output/log/en_optimised_model_candidates.{project}.{test_id}.{date_str}.log.txt"
    threads:1
    resources:
        mem_mb=1000
    conda: "../../Envs/kir_hla_ml_env.yml"
    shell: 
        """
            python {input.script} \
            --TestID {params.test_id} \
            --TestPlan {input.test_plan} \
            --SeqSelectionResults {input.fs_bs_results} \
            --DateStr {params.date_str} \
            --OutputDir {params.out_dir} \
            > {log.file}
        """