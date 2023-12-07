rule run_rf_train_test_r2_opt:
    input:
        script = "Scripts/RandomForest/run_rf_validation.py",
        test_plan = RF_TEST_PLAN,
        h_params = RF_H_PARAMS_R2,
        features = "Output/{project}/RandomForest/{date_str}/Test{test_id}/Optimised/optimised_model_candidates.{test_id}.{date_str}.csv",
    params:
        date_str = "{date_str}",
        test_id = "{test_id}",
        trn_test = True,
        out_dir = "Output/{project}/RandomForest/{date_str}/Test{test_id}/Optimised",
        n_jobs = 1
    output:
        results = "Output/{project}/RandomForest/{date_str}/Test{test_id}/Optimised/rf_train_test_score.optimised.{test_id}.{date_str}.csv"
    log:
        file =  "Output/log/rf_train_test_r2.optimised.{project}.{test_id}.{date_str}.log.txt"
    threads:1
    resources:
        mem_mb=1000
    conda: "../../Envs/kir_hla_ml_env.yml"
    shell: 
        """
            python {input.script} \
            --TestPlan {input.test_plan} \
            --HyperParams {input.h_params} \
            --TestID {params.test_id} \
            --SeqSelectionResults {input.features} \
            --TrainTestPartitioning {params.trn_test} \
            --DateStr {params.date_str} \
            --OutputDir {params.out_dir} \
            --NumJobs {params.n_jobs} \
            > {log.file}
        """