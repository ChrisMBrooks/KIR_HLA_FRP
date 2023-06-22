rule gen_rf_gs_results_r2_line_plot:
    input:
        script = "Scripts/RandomForest/rf_consolidate_plot_gs_results.py",
        gs_results_w_cv = [
            "Output/{{project}}/RandomForest/{{date_str}}/Test{{test_id}}/ParallelisedData/rf_parallel_gs_results_r2_w_cv.{start_index}.{step}.{{test_id}}.{{date_str}}.csv".format(
                start_index=si, step=GS_INDEX_STEP) for si in GS_INDECES],
        gs_results_wo_cv = [
            "Output/{{project}}/RandomForest/{{date_str}}/Test{{test_id}}/ParallelisedData/rf_parallel_gs_results_r2_wo_cv.{start_index}.{step}.{{test_id}}.{{date_str}}.csv".format(
           start_index=si, step=GS_INDEX_STEP) for si in GS_INDECES]
    params:
        date_str = "{date_str}",
        test_id = "{test_id}",
        start_index = GS_INDECES[0],
        stop_index = GS_INDECES[-1],
        step = GS_INDEX_STEP,
        in_dir = "Output/{project}/RandomForest/{date_str}/Test{test_id}/ParallelisedData",
        out_dir = "Output/{project}/RandomForest/{date_str}/Test{test_id}",
    output:
        plot = "Output/{project}/RandomForest/{date_str}/Test{test_id}/rf_gs_results_r2_line_plot.min_samples_split.{test_id}.{date_str}.png",
        gs_data_w_cv = "Output/{project}/RandomForest/{date_str}/Test{test_id}/rf_gs_results_r2_w_cv.{test_id}.{date_str}.csv"
    log:
        file = "Output/log/rf_get_gs_line_plots_r2.{project}.{test_id}.{date_str}.log.txt"
    threads: 1
    conda: "../../Envs/kir_hla_ml_env.yml"
    shell: 
        """
            python {input.script} \
            --TestID {params.test_id} \
            --DateStr {params.date_str} \
            --StartIndex {params.start_index} \
            --StopIndex {params.stop_index} \
            --Step {params.step} \
            --Iteration 2 \
            --InputDir {params.in_dir} \
            --OutputDir {params.out_dir} \
            > {log.file}
        """
