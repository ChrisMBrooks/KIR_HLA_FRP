rule run_enr2_validation_opt:
    input:
        script = "Scripts/ElasticNetR2/run_enr2_validation.py",
        test_plan = EN_TEST_PLAN,
        h_params = "Output/{project}/ElasticNet/{date_str}/Test{test_id}/ElasticNetR2/enr2_gs_run_summary_results.{test_id}.{date_str}.csv",
        features = "Output/{project}/ElasticNet/{date_str}/Test{test_id}/ElasticNetR2/Optimised/optimised_model_candidates.{test_id}.{date_str}.csv"
    params:
        date_str = "{date_str}",
        test_id = "{test_id}",
        trn_test = False,
        out_dir = "Output/{project}/ElasticNet/{date_str}/Test{test_id}/ElasticNetR2/Optimised"
    output:
        results = "Output/{project}/ElasticNet/{date_str}/Test{test_id}/ElasticNetR2/Optimised/enr2_final_score.optimised.{test_id}.{date_str}.csv"
    log:
        file =  "Output/log/enr2_validation.optimised.{project}.{test_id}.{date_str}.log.txt"
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
            --HyperParams {input.h_params} \
            --OutputDir {params.out_dir} \
            > {log.file}
        """