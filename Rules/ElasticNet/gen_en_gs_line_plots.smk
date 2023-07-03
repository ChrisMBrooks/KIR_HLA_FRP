rule gen_en_gs_line_plots:
    input:
        script = "Scripts/ElasticNet/gen_en_gs_line_plots.py",
        gs_details = "Output/{project}/ElasticNet/{date_str}/Test{test_id}/en_gs_details.{test_id}.{date_str}.csv",
        gs_details_no_cv = "Output/{project}/ElasticNet/{date_str}/Test{test_id}/en_gs_details_no_cv.{test_id}.{date_str}.csv"
    params: 
        date_str = "{date_str}",
        test_id = "{test_id}",
        out_dir = "Output/{project}/ElasticNet/{date_str}/Test{test_id}"
    output:
        alpha_plot = "Output/{project}/ElasticNet/{date_str}/Test{test_id}/en_gs_results_line_plot.alpha.{test_id}.{date_str}.png",
        l1_plot = "Output/{project}/ElasticNet/{date_str}/Test{test_id}/en_gs_results_line_plot.l1_ratio.{test_id}.{date_str}.png"
    log:         
        file = "Output/log/gen_en_gs_line_plots.{project}.{test_id}.{date_str}.log.txt"
    threads: 1
    resources:
        mem_mb=1000
    conda: "../../Envs/kir_hla_ml_env.yml"
    shell: 
        """
            python {input.script} \
            --TestID {params.test_id} \
            --DateStr {params.date_str} \
            --FileWCV {input.gs_details} \
            --FileWOCV {input.gs_details_no_cv} \
            --OutputDir {params.out_dir} \
            > {log.file}
        """