rule get_rf_h_params_combinations:
    input:
        script = "Scripts/RandomForest/rf_gen_gs_params.py"
    output:
        csv = "Output/{project}/RandomForest/MetaData/rf_gs_h_parmas.{date_str}.csv"
    log:
        file = "Output/log/rf_get_h_params_combinations.{project}.{date_str}.log.txt"
    threads:1
    resources:
        mem_mb=1000
    conda: "../../Envs/kir_hla_ml_env.yml"
    shell: 
        """
            python {input.script} --Output {output.csv} > {log.file}
        """
