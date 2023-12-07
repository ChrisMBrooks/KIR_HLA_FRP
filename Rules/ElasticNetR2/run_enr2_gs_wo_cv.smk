rule run_enr2_gs_wo_cv:
    input:
        script = "Scripts/ElasticNetR2/run_enr2_gs_wo_cv.py",
        test_plan = EN_TEST_PLAN,
        h_params = "Output/{project}/ElasticNet/MetaData/en_gs_h_parmas.{date_str}.csv",
        features = "Output/{project}/ElasticNet/{date_str}/Test{test_id}/ElasticNetR2/enr2_seq_selection_candidates.{test_id}.{date_str}.csv"
    params:
        date_str = "{date_str}",
        test_id = "{test_id}",
        out_dir = "Output/{project}/ElasticNet/{date_str}/Test{test_id}/ElasticNetR2",
    output:
        summary = "Output/{project}/ElasticNet/{date_str}/Test{test_id}/ElasticNetR2/enr2_gs_run_summary_results_no_cv.{test_id}.{date_str}.csv",
        gs_details = "Output/{project}/ElasticNet/{date_str}/Test{test_id}/ElasticNetR2/enr2_gs_details_no_cv.{test_id}.{date_str}.csv"
    log:         
        file = "Output/log/run_enr2_gs.{project}.{test_id}.{date_str}.log.txt"
    conda: "../../Envs/kir_hla_ml_env.yml"
    threads: 16
    resources:
        mem_mb=8000
    shell: 
        """
            python {input.script} \
            --OutputDir {params.out_dir} \
            --TestID {params.test_id} \
            --TestPlan {input.test_plan} \
            --DateStr {params.date_str} \
            --Input {input.features} \
            --HyperParams {input.h_params}
            > {log.file}
        """