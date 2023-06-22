rule get_en_h_params_combinations:
    input:
        script = "Scripts/ElasticNet/en_gen_gs_params.py"
    output:
        csv = "Output/{project}/ElasticNet/MetaData/en_gs_h_parmas.{date_str}.csv"
    log:
        file = "Output/log/en_get_h_params_combinations.{project}.{date_str}.log.txt"
    threads: 1
    conda: "../../Envs/kir_hla_ml_env.yml"
    shell: 
        """
            python {input.script} --Output {output.csv} > {log.file}
        """
