rule run_enr2_qc_fs_bs:
    input:
        script = "Scripts/ElasticNetR2/run_enr2_qc_fs_bs.py",
        test_plan = EN_TEST_PLAN,
        features = "Output/{project}/ElasticNet/{date_str}/Test{test_id}/en_feature_coefs.{test_id}.{date_str}.csv",
        h_params = "Output/{project}/ElasticNet/{date_str}/Test{test_id}/en_gs_run_summary_results.{test_id}.{date_str}.csv"
    params:
        date_str = "{date_str}",
        test_id = "{test_id}",
        out_dir = "Output/{project}/ElasticNet/{date_str}/Test{test_id}/ElasticNetR2",
        num_jobs = 32
    output:
        results = "Output/{project}/ElasticNet/{date_str}/Test{test_id}/ElasticNetR2/enr2_seq_selection_candidates.{test_id}.{date_str}.csv", 
        run_summary = "Output/{project}/ElasticNet/{date_str}/Test{test_id}/ElasticNetR2/enr2_seq_selection_summary.{test_id}.{date_str}.csv",
    log:
        file = "Output/log/run_enr2_qc_fs_bs.{project}.{test_id}.{date_str}.log.txt",
    conda: "../../Envs/kir_hla_ml_env.yml"
    threads: 32
    resources:
        mem_mb=6000
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
            --HyperParams {input.h_params} \
            --OutputDir {params.out_dir} \
            --NumJobs {params.num_jobs} \
            > {log.file}
        """