rule run_enr2_gs:
    input:
        script = "Scripts/ElasticNetR2/run_enr2_gs.py",
        test_plan = EN_TEST_PLAN,
        features = "Output/{project}/ElasticNet/{date_str}/Test{test_id}/ElasticNetR2/enr2_seq_selection_candidates.{test_id}.{date_str}.csv"
    params:
        date_str = "{date_str}",
        test_id = "{test_id}",
        out_dir = "Output/{project}/ElasticNet/{date_str}/Test{test_id}/ElasticNetR2",
        num_jobs = 32
    output:
        feature_coefs = "Output/{project}/ElasticNet/{date_str}/Test{test_id}/ElasticNetR2/enr2_feature_coefs.{test_id}.{date_str}.csv",
        summary = "Output/{project}/ElasticNet/{date_str}/Test{test_id}/ElasticNetR2/enr2_gs_run_summary_results.{test_id}.{date_str}.csv",

        gs_details = "Output/{project}/ElasticNet/{date_str}/Test{test_id}/ElasticNetR2/enr2_gs_details.{test_id}.{date_str}.csv",
        plot = "Output/{project}/ElasticNet/{date_str}/Test{test_id}/ElasticNetR2/enr2_gs_heat_map.{test_id}.{date_str}.png"
    log:         
        file = "Output/log/run_enr2_gs.{project}.{test_id}.{date_str}.log.txt"
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
            --OutputDir {params.out_dir} \
            --TestID {params.test_id} \
            --TestPlan {input.test_plan} \
            --Input {input.features} \
            --DateStr {params.date_str} \
            --NumJobs {params.num_jobs} \
            > {log.file}
        """