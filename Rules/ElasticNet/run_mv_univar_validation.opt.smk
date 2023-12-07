rule run_mv_univar_validation_opt:
    input:
        script = "Scripts/General/final_univar_validation.py",
        features = "Output/{project}/ElasticNet/{date_str}/Test{test_id}/Optimised/mv_feature_importance_perm_values_2sig.optimised.{test_id}.{date_str}.csv",
        m_effectives = "Output/{project}/ElasticNet/{date_str}/Test{test_id}/Optimised/mv_m_effective_results.{test_id}.{date_str}.csv",
    params:
        date_str = "{date_str}",
        test_id = "{test_id}",
        trn_test = False,
        out_dir = "Output/{project}/ElasticNet/{date_str}/Test{test_id}/Optimised"
    output:
        results = "Output/{project}/ElasticNet/{date_str}/Test{test_id}/Optimised/univar_final_scores.optimised.{test_id}.{date_str}.csv"
    log:
        file =  "Output/log/en_univar_final_scores.optimsed.{project}.{test_id}.{date_str}.log.txt"
    threads:1
    resources:
        mem_mb=1000
    conda: "../../Envs/kir_hla_ml_env.yml"
    shell: 
        """
            python {input.script} \
            --TestID {params.test_id} \
            --Input {input.features} \
            --MeffFile {input.m_effectives} \
            --DateStr {params.date_str} \
            --OutputDir {params.out_dir} \
            > {log.file}
        """