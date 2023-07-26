import os

test_id = 26
filename_template = "/Volumes/cmb22/home/KIR_HLA_STUDY_TWINS_MM_05/KIR_HLA_FRP/Output/KIR_HLA_TWINS_MM05/RandomForest/22062023/Test{test_id}/ParallelisedData/rf_parallel_gs_results_r1_w_cv.{it_id}.1000.{test_id}.22062023.csv"

count = 0
for i in range(0, 380, 1):
    it_id = 1000*i
    filename = filename_template.format(it_id=it_id, test_id=test_id)
    if not os.path.isfile(filename):
        print(it_id)
        count+=1
    
print('missing count:', count)