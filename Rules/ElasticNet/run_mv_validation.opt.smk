rule run_mv_validation_opt:
    input:
        script = "Scripts/Multivariate/mv_validation.py",
        test_plan = EN_TEST_PLAN,
        features = "Output/{project}/ElasticNet/{date_str}/Test{test_id}/Optimised/optimised_model_candidates.{test_id}.{date_str}.csv"
    params:
        date_str = "{date_str}",
        test_id = "{test_id}",
        trn_test = False,
        out_dir = "Output/{project}/ElasticNet/{date_str}/Test{test_id}/Optimised"
    output:
        results = "Output/{project}/ElasticNet/{date_str}/Test{test_id}/Optimised/mv_final_score.optimised.{test_id}.{date_str}.csv"
    log:
        file =  "Output/log/mv_validation.optimised.{project}.{test_id}.{date_str}.log.txt"
    threads: 1
    resources:
        mem_mb=1000
    conda: "../../Envs/kir_hla_ml_env.yml"
    shell: 
        """
            python {input.script} \
            --TestID {params.test_id} \
            --DateStr {params.date_str} \
            --Input {input.features} \
            --TrainTestPartitioning {params.trn_test} \
            --TestPlan {input.test_plan} \
            --OutputDir {params.out_dir} \
            > {log.file}
        """