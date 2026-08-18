from Inference_Board import RunInference
import time


def Test(X_test , y_test , models ,args):
    
    all_per_sample_results = []
    

    

  
    total_inference_duration = 0.0 
    
    # Now, loop through each data type for inference and add type-specific metrics
    for w in range(len(X_test)):
        
        start_time_infernce = time.time()
        inference_obj = RunInference(X_test=X_test[w], y_test=y_test[w], models=models, stages=args.proportions , window_num= w)

        sub_forest_entropy, prediction = inference_obj.predict_proba()

        _, _, _, per_sample_results_for_window = inference_obj.check_exit(sub_forest_entropy, args.th_combination, prediction, y_test[w], start_time_infernce)
        
        # all_stage_exit_accuracies = inference_obj.ExitAtAllStage()
        
       
    
        all_per_sample_results.extend(per_sample_results_for_window)


    return all_per_sample_results