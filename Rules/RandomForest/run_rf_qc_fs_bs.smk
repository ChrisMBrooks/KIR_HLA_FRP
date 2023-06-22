rule run_rf_qc_fs_bs:
    input:
        script = "Scripts/RandomForest/rf_qc_fs_bs.py",
        test_plan = RF_TEST_PLAN,
        h_params = RF_H_PARAMS_R1,
        importances = "Output/{project}/RandomForest/{date_str}/Test{test_id}/rf_r1_feature_importance_impurity_rankings.{test_id}.{date_str}.csv"
    params:
        date_str = "{date_str}",
        test_id = "{test_id}",
        s_threshold = FS_BS_INCLUSION_THRESHOLD,
        f_selection = 1,
        b_selection = 1,
        out_dir = "Output/{project}/RandomForest/{date_str}/Test{test_id}",
        n_jobs = 16
    output:
        results = "Output/{{project}}/RandomForest/{{date_str}}/Test{{test_id}}/rf_seq_selection_candidates.fs_bs.{thresh}.{{test_id}}.{{date_str}}.csv".format(
            thresh=FS_BS_INCLUSION_THRESHOLD
        ), 
        run_summary = "Output/{{project}}/RandomForest/{{date_str}}/Test{{test_id}}/rf_seq_selection_summary.fs_bs.{thresh}.{{test_id}}.{{date_str}}.csv".format(
            thresh=FS_BS_INCLUSION_THRESHOLD
        ),
    log:
        file = "Output/log/run_rf_qc_fs_bs.{project}.{test_id}.{date_str}.log.txt",
    conda: "../../Envs/kir_hla_ml_env.yml"
    threads: 16
    shell: 
        """
            python {input.script} \
            --TestPlan {input.test_plan} \
            --HyperParams {input.h_params} \
            --TestID {params.test_id} \
            --DateStr {params.date_str} \
            --FeatureImportances {input.importances} \
            --SelectionThreshold {params.s_threshold} \
            --ForwardSelection {params.f_selection} \
            --BackwardSelection {params.b_selection} \
            --OutputDir {params.out_dir} \
            --NumJobs {params.n_jobs} \
            > {log.file}
        """