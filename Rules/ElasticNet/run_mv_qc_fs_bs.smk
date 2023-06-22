rule run_mv_qc_fs_bs:
    input:
        script = "Scripts/Multivariate/mv_qc_fs_bs.py",
        test_plan = EN_TEST_PLAN,
        features = "Output/{project}/ElasticNet/{date_str}/Test{test_id}/en_feature_coefs.{test_id}.{date_str}.csv"
    params:
        date_str = "{date_str}",
        test_id = "{test_id}",
        out_dir = "Output/{project}/ElasticNet/{date_str}/Test{test_id}"
    output:
        results = "Output/{project}/ElasticNet/{date_str}/Test{test_id}/mv_qc_fs_bs_candidate_features.{test_id}.{date_str}.csv", 
        run_summary = "Output/{project}/ElasticNet/{date_str}/Test{test_id}/mv_qc_fs_bs_summary.{test_id}.{date_str}.csv",
    log:
        file = "Output/log/run_mv_qc_fs_bs.{project}.{test_id}.{date_str}.log.txt",
    conda: "../../Envs/kir_hla_ml_env.yml"
    threads: 16
    shell: 
        """
            export MKL_NUM_THREADS=1
            export NUMEXPR_NUM_THREADS=1
            export OMP_NUM_THREADS=1
            export OPENBLAS_NUM_THREADS=1
            export VECLIB_MAXIMUM_THREADS=1

            python {input.script} \
            --TestID {params.test_id} \
            --DateStr {params.date_str} \
            --Input {input.features} \
            --TestPlan {input.test_plan} \
            --OutputDir {params.out_dir} \
            > {log.file}
        """