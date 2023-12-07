rule run_rf_meff_calc:
    input:
        script = "Scripts/General/run_poolr_meff_calculation.py",
        features = "Output/{project}/RandomForest/{date_str}/Test{test_id}/Validation/rf_feature_importance_perm_values_2sig.{test_id}.{date_str}.csv"
    params:
        date_str = "{date_str}",
        test_id = "{test_id}",
        out_dir = "Output/{project}/RandomForest/{date_str}/Test{test_id}/Validation"
    output:
        results = "Output/{project}/RandomForest/{date_str}/Test{test_id}/Validation/rf_m_effective_results.{test_id}.{date_str}.csv"
    log:
        file =  "Output/log/rf_m_effective_results.v.{project}.{test_id}.{date_str}.log.txt"
    threads: 1
    resources:
        mem_mb=1000
    conda: "../../Envs/rpy2_env.yml"
    shell: 
        """
            python {input.script} \
            --TestID {params.test_id} \
            --Input {input.features} \
            --OutputDir {params.out_dir} \
            --DateStr {params.date_str} \
            > {log.file}
        """