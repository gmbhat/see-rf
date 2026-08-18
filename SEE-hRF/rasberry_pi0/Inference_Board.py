import numpy as np
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import accuracy_score
from ECE import ECE_computation , calculate_confidence_max_prob
import concurrent.futures , time


def _predict_for_tree(args):
    tree_batch, X_test_temp, n_samples, classes_ = args
    scores_per_batch = np.zeros((n_samples, len(classes_)))
    X_test_temp=np.atleast_2d(X_test_temp)
    for tree in tree_batch:
        predictions = tree.predict(X_test_temp)
        for sample_idx in range(n_samples):
            predicted_class = predictions[sample_idx]
        
            if np.isin(predicted_class, classes_): 
                class_idx = np.where(classes_ == predicted_class)[0][0]
                scores_per_batch[sample_idx, class_idx] += 1   
    return scores_per_batch


class RunInference:

    def __init__(self, X_test, y_test, models , stages, window_num):
        
        self.X_test = X_test 
        self.y_test = y_test 
        self.trees_ = None
        self.n_samples = 1 #self.X_test.shape[0] # it should be 1
        self.classes_ = None
        self.models = models
        self.stages = stages
        self.srf_entropy = None 
        self._final_passed_indices_after_check_exit = [] 
        self._all_exited_indices_after_check_exit = set() 
        self.window_num = window_num 
        self.stage_durations = []
        self.indices_in = []
        self.indices_out = []
        self.absolute_time_per_stage = []
        
    def entropy(self , probabilities):  
        epsilon = 1e-5  
        return -np.sum(probabilities * np.log(probabilities + epsilon), axis=1)
    
    def predict_proba(self):
        
        y_processed = np.array(self.y_test).ravel().astype(int)
        self.classes_ = unique_labels(y_processed)
        self.absolute_time_per_stage = []
        self.n_classes_ = len(self.classes_)
        n_features_total = self.X_test.shape[0]
        self.all_scores = np.zeros((self.n_samples, self.n_classes_))
        self.all_entropy = []
        self.all_predict = []
        self.connected_prediction = []
        self.list_of_Prob = []
        
        self.stage_durations = [] # Reset time tracking for each call
        cumulative_time = 0.0 # No longer needed, but good for local tracking
        BATCH_SIZE = 10
        for j in range(len(self.models)):
            t_stage_start = time.time()
            n_features_to_select = int(np.ceil(n_features_total * self.stages[j])) # Use ceil to ensure at least 1
            n_features_to_select = min(n_features_to_select, n_features_total) # Cap
            selected_feature_indices = np.arange(n_features_to_select)
        
            X_test_temp = self.X_test[:n_features_to_select]
            X_test_temp=np.atleast_2d(X_test_temp)
            
            srf = self.models[j]['model']
            self.trees_ = srf.trees_ 
            scores = np.zeros((self.n_samples, self.n_classes_))  
            
            # do the prediction parallel
            tree_batches = [self.trees_[i:i + BATCH_SIZE] for i in range(0, len(self.trees_), BATCH_SIZE)]
            list_args = [(batch, X_test_temp, 1, self.classes_) for batch in tree_batches]
                
            max_workers = min(len(list_args), 4)   
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                outputs = list(executor.map(_predict_for_tree, list_args))
            
            for scores_per_tree in outputs:
                scores += scores_per_tree

            self.all_scores += scores
             
            self.connected_prediction.append(np.argmax(self.all_scores , axis=1)[0])
            
            denominator = len(self.trees_) * (j + 1)
            
            proba = self.all_scores / denominator
            self.list_of_Prob.append(proba)
            self.srf_entropy =  self.entropy(proba)
            
            self.all_entropy.append( self.srf_entropy[0])
            self.all_predict.append(np.argmax(scores , axis=1)[0])
            t_stage_end = time.time()
            time_spent_at_stage = t_stage_end - t_stage_start
        
        # Store the individual duration (this is the tN)
            self.absolute_time_per_stage.append(t_stage_end)
        
        return self.all_entropy , self.all_predict  
    
 
    def ExitAtAllStage(self):
 
        accuracies_exit_all = {}
        
        for j , prediction_per_stage in enumerate(self.connected_prediction):
            
            accuracy = accuracy_score(self.y_test, prediction_per_stage)
            accuracies_exit_all[f"accuracy_exit_all_Sample_RF {j+1}"] = f"{accuracy:.4f}"
        
        return accuracies_exit_all



    # In Inference_Board.py, inside class RunInference:

    # In Inference_Board.py, inside class RunInference:

# FIX: Added stage_durations and start_time to the arguments
    def check_exit(self , sub_rf_entropy , threshold_list_of_keys , predictions , y_test, start_time):
    
        t_start_absolute = start_time
        
        num_total_stages = len(self.models) 
        num_thresholds = len(threshold_list_of_keys) 
        
        # --- Per-Sample Result Initialization ---
        per_sample_result = {
            't_start': -1.0, # Will be set to t_start_absolute later
            'exit_level': -1, 
            'total': -1.0, 
            'prediction': -1, 
            'true_label': y_test, 
            'correctness': False,
            'window_num': self.window_num,
            'data%': self.stages[-1]
        }
        # Initialize ALL individual stage time tracking fields to -1.0
        for i in range(1, num_total_stages + 1):
            per_sample_result[f't{i}'] = -1.0 
        # ---------------------------------------

        exited_stage = -1
        
        # Loop for stages 1 to N-1
        for stage_idx in range(num_total_stages - 1):
            
            stage_num = stage_idx + 1 
        
            # FIX 3: Get the absolute time when this stage's prediction finished
            t_exit_absolute = self.absolute_time_per_stage[stage_idx]
            
            # FIX 4: Record the absolute time (tN)
            per_sample_result[f't{stage_num}'] = t_exit_absolute

            sample_entropy_value = sub_rf_entropy[stage_idx] 
            
            if sample_entropy_value < threshold_list_of_keys[stage_idx]:
                # Sample Exits

                t_exit_absolute = time.time() # FIX 2: Capture absolute exit time

                exited_stage = stage_num
                
                # Record final exit metrics
                per_sample_result['exit_level'] = stage_num
                per_sample_result['prediction'] = predictions[stage_idx] 
                
                # FIX 3: Total time is (Absolute Exit time - Absolute Start time)
                per_sample_result['total'] = t_exit_absolute - t_start_absolute 
                
                per_sample_result['correctness'] = (per_sample_result['true_label'] == per_sample_result['prediction'])
                per_sample_result['data%'] = self.stages[stage_num - 1]
                break 
                
        # --------------------------------------------------------------------------------
        # 2. FORCED EXIT AT THE FINAL RF STAGE (RF N)
        # --------------------------------------------------------------------------------

        final_rf_idx = num_total_stages - 1 
        final_stage_num = num_total_stages 

        if exited_stage == -1: # Only proceed if the sample has not exited yet
            
            # FIX 6: Get the absolute time for the final stage
            t_exit_absolute = self.absolute_time_per_stage[final_rf_idx]
            
            final_prediction = predictions[final_rf_idx]
            
            exited_stage = final_stage_num
            
            final_prediction = predictions[final_rf_idx]
            
            
            # Record exit metrics
            per_sample_result['exit_level'] = final_stage_num
            per_sample_result['prediction'] = final_prediction
            
            # FIX 5: Total time is (Absolute Exit time - Absolute Start time)
            per_sample_result['total'] = t_exit_absolute - t_start_absolute
            
            # Record the individual stage time for the final stage (tN)
            per_sample_result[f't{final_stage_num}'] = t_exit_absolute 
            
            per_sample_result['correctness'] = (per_sample_result['true_label'] == per_sample_result['prediction'])
            per_sample_result['data%'] = self.stages[stage_num]
        # --------------------------------------------------------------------------------
        # 3. Finalize and Return
        # --------------------------------------------------------------------------------
        
        # FIX 6: Ensure t_start in the dict is the absolute start time
        per_sample_result['t_start'] = t_start_absolute 
        
        indices_in = []
        indices_out = [self.window_num]
        all_keys_inference_metrics = [] 
        per_sample_results_list = [per_sample_result]
        
        return indices_in, indices_out, all_keys_inference_metrics, per_sample_results_list
        # def check_exit(self , sub_rf_entropy , threshold_list_of_keys , predictions , y_test):
            

        #     t_start = time.time()
        #     # Pre-process threshold_list_of_keys if it's in the tuple format ([...],)
        #     if isinstance(threshold_list_of_keys, tuple) and len(threshold_list_of_keys) == 1:
        #         threshold_list_of_keys = threshold_list_of_keys[0]
        #     ece_per_exit = []         
        #     all_confidences = []
        #     all_corrects = []
            
        #     if sub_rf_entropy is None:
        #         print("Error in check_exit: sub_rf_entropy is not available.")
        #         # Added a return value for per_sample_results
        #         return [], [], [], [] 

    #     # N = Total number of Random Forests (stages)
    #     num_total_stages = len(self.models) 
    #     # N-1 = Number of thresholds provided
    #     num_thresholds = len(threshold_list_of_keys) 
        
    #     if num_thresholds != num_total_stages - 1:
    #         print(f"Error: Expected {num_total_stages - 1} thresholds, got {num_thresholds}. Exiting.")
    #         return [], [], [], []

    #     if len(sub_rf_entropy) != num_total_stages:
    #         print(f"Error in check_exit: Expected {num_total_stages} entropy arrays, got {len(sub_rf_entropy)}.")
    #         return [], [], [], []

    #     num_total_samples = len(sub_rf_entropy[0])
    #     all_original_indices = list(range(num_total_samples))
        
    #     # --- NEW: Per-Sample Result Tracking ---
    #     per_sample_results = [
    #         {
    #             't_start': t_start, 
    #             'exit_level': -1, 
    #             'total': -1.0, 
    #             'prediction': -1, 
    #             'true_label': y_test[i],
    #             # Add window number for later aggregation
    #             'window_num': self.window_num 
    #         }
    #         for i in range(num_total_samples)
    #     ]
    #     # Initialize time tracking fields for all stages (t1, t2, ...)
    #     for i in range(1, num_total_stages + 1):
    #         for sample_dict in per_sample_results:
    #             sample_dict[f't{i}'] = -1.0 
    #             sample_dict[f'data%'] = self.stages[i-1]
    #     # ---------------------------------------

    #     print(f"check_exit: Initial number of samples: {num_total_samples}")
    #     print(f"--- Processing Threshold Configuration: {threshold_list_of_keys} ---")

    #     indices_being_processed_for_this_key = list(all_original_indices) 
    #     exited_samples_this_key_cumulative = set()

    #     current_key_inference_metrics = {}
    #     current_key_inference_metrics["Threshold_Configuration"] = str(threshold_list_of_keys) 

    #     EnergyUsed_sum = []
    #     Total_acc_per_config = []
        
    #     # --------------------------------------------------------------------------------
    #     # 1. PROCESS INTERMEDIATE, THRESHOLDED EXITS (RF 1 to RF N-1)
    #     # --------------------------------------------------------------------------------
    #     for threshold_j_idx, threshold_value in enumerate(threshold_list_of_keys):
            
    #         stage_num = threshold_j_idx + 1 # 1-based index for stage/exit
    #         t_stage_start = time.time() # Start time measurement for this stage's *processing*

    #         # Check if there are any samples left from the previous stage
    #         if not indices_being_processed_for_this_key:
    #             print(f"Exit stage {stage_num}: No samples left to process. Skipping.")
    #             # Fill placeholder metrics for remaining unvisited stages 
    #             for k in range(threshold_j_idx, num_thresholds):
    #                 # ... (Placeholder code remains the same)
    #                 pass # Keep placeholder logic as in your original code
    #             break 

    #         exited_at_this_stage = []
    #         passed_this_stage = []
            
    #         # Check samples against the current threshold
    #         for original_sample_index in indices_being_processed_for_this_key:
    #             sample_entropy_value = sub_rf_entropy[threshold_j_idx][original_sample_index]
                
    #             if sample_entropy_value < threshold_value:
    #                 exited_at_this_stage.append(original_sample_index)
    #                 exited_samples_this_key_cumulative.add(original_sample_index) 
                    
    #                 # --- NEW: Record Exit Info for this Sample ---
    #                 per_sample_results[original_sample_index]['exit_level'] = stage_num
    #                 # Use connected_prediction for the final prediction at this stage
    #                 per_sample_results[original_sample_index]['prediction'] = self.connected_prediction[threshold_j_idx][original_sample_index]

    #             else:
    #                 passed_this_stage.append(original_sample_index)

    #         # --- NEW: Record Time for Exiting Samples ---
    #         t_stage_end = time.time()
    #         t_elapsed_current_stage = t_stage_end - t_stage_start # Time spent on THIS stage
            
    #         # Calculate total accumulated time for the exiting samples
    #         for original_sample_index in exited_at_this_stage:
    #              # Sum time spent in previous stages (-1.0 for non-visited stages) + current stage time
    #              total_time_so_far = sum(per_sample_results[original_sample_index][f't{k}'] for k in range(1, stage_num))
                 
    #              # Only count stages visited (time > 0)
    #              # However, since we don't track time for samples that PASS, we'll keep it simple: 
    #              # t{stage_num} is the time to process this stage, and 'total' is the sum of stages
    #              per_sample_results[original_sample_index][f't{stage_num}'] = t_elapsed_current_stage 
    #              per_sample_results[original_sample_index]['total'] = total_time_so_far + t_elapsed_current_stage 

    #         indices_being_processed_for_this_key = passed_this_stage 
            
    #         # --- Calculate Metrics for this EXIT (RF_{j+1}) ---
    #         # ... (Existing metric calculation and storage remains the same)
            
    #         print(f"Exit stage {stage_num} (Threshold Value: {threshold_value:.4f}): samples exited: {len(exited_at_this_stage)}")
            
    #         exit_percentage = float(len(exited_at_this_stage) / num_total_samples)
    #         EnergyUsed_sum.append(float(exit_percentage * self.stages[threshold_j_idx]))

    #         subset_predictions = self.connected_prediction[threshold_j_idx][exited_at_this_stage] # Use connected_prediction
    #         subset_true_labels = y_test[exited_at_this_stage]
    #         # ... (Remaining metric calculations for Accuracy, ECE, Total_acc_per_config)
            
            
    #         # Store metrics for this specific RF stage (RF_{j+1})
    #         current_key_inference_metrics[f"Threshold_Value_RF_{stage_num}"] = f"{threshold_value:.4f}"
    #         current_key_inference_metrics[f"Samples_Exited_RF_{stage_num}"] = len(exited_at_this_stage)
    #         current_key_inference_metrics[f"Samples_Remaining_RF_{stage_num}"] = len(indices_being_processed_for_this_key)
    #         # ... (Remaining metric fields for the stage)

    #         # NOTE: accuracy_score and ECE calculation must be here for the per-stage metrics to be correct

    #         if len(subset_predictions) == 0:
    #             sklearn_accuracy = -1
    #             ece = -1
    #             acc_str = f"{-1.0000:.4f}"
    #             ece_str = f"{-1.0000:.4f}"
    #         else:
    #             sklearn_accuracy = accuracy_score(subset_true_labels, subset_predictions)
    #             corrects = (subset_true_labels == subset_predictions).astype(int).tolist()
    #             confidences = calculate_confidence_max_prob(self.list_of_Prob[threshold_j_idx][exited_at_this_stage])
    #             ece = ECE_computation(confidences, corrects, 10)
    #             ece_per_exit.append(ece)
    #             all_confidences += confidences
    #             all_corrects += corrects
    #             acc_str = f"{sklearn_accuracy:.4f}"
    #             ece_str = f"{ece:.4f}"
                    
    #         if sklearn_accuracy == -1:
    #             Total_acc_per_config.append(0)
    #         else:
    #             Total_acc_per_config.append(sklearn_accuracy * exit_percentage)
            
    #         current_key_inference_metrics[f"Accuracy_RF_{stage_num}"] = acc_str
    #         current_key_inference_metrics[f"Exit_Percentage_RF_{stage_num}"] = f"{exit_percentage:.4f}"
    #         current_key_inference_metrics[f"ECE_RF{stage_num}"] = ece_str
            
    #     # --------------------------------------------------------------------------------
    #     # 2. FORCED EXIT AT THE FINAL RF STAGE (RF N)
    #     # --------------------------------------------------------------------------------

    #     final_rf_idx = num_total_stages - 1 
    #     final_stage_num = num_total_stages 
    #     t_stage_start_final = time.time() # Start time measurement for the final stage
        
    #     exited_at_final_stage = indices_being_processed_for_this_key 
        
    #     if exited_at_final_stage:
            
    #         subset_predictions = self.connected_prediction[final_rf_idx][exited_at_final_stage] # Use connected_prediction
    #         subset_true_labels = y_test[exited_at_final_stage]
            
    #         self._all_exited_indices_after_check_exit.update(exited_at_final_stage)
    #         exited_samples_this_key_cumulative.update(exited_at_final_stage) 
            
    #         # --- NEW: Record Final Exit Info for Samples ---
    #         t_stage_end_final = time.time()
    #         t_elapsed_final_stage = t_stage_end_final - t_stage_start_final # Time spent on THIS stage

    #         for original_sample_index in exited_at_final_stage:
    #             per_sample_results[original_sample_index]['exit_level'] = final_stage_num
    #             # Find the corresponding prediction/label
    #             idx_in_subset = exited_at_final_stage.index(original_sample_index)
    #             per_sample_results[original_sample_index]['prediction'] = subset_predictions[idx_in_subset]
                
    #             # Calculate total accumulated time
    #             total_time_so_far = sum(per_sample_results[original_sample_index][f't{k}'] for k in range(1, final_stage_num))
                
    #             per_sample_results[original_sample_index][f't{final_stage_num}'] = t_elapsed_final_stage
    #             per_sample_results[original_sample_index]['total'] = total_time_so_far + t_elapsed_final_stage
    #         # ---------------------------------------------
            
    #         exit_percentage = float(len(exited_at_final_stage) / num_total_samples)
    #         EnergyUsed_sum.append(float(exit_percentage * self.stages[final_rf_idx]))
            
    #         # --- Calculate Metrics for the FINAL EXIT (RF_N) ---
    #         if len(subset_predictions) == 0:
    #             sklearn_accuracy = -1
    #             ece = -1
    #             acc_str = f"{-1.0000:.4f}"
    #             ece_str = f"{-1.0000:.4f}"
    #         else:
    #             sklearn_accuracy = accuracy_score(subset_true_labels, subset_predictions)
    #             corrects = (subset_true_labels == subset_predictions).astype(int).tolist()
    #             confidences = calculate_confidence_max_prob(self.list_of_Prob[final_rf_idx][exited_at_final_stage])
    #             ece = ECE_computation(confidences, corrects, 10)
    #             ece_per_exit.append(ece)
    #             all_confidences += confidences
    #             all_corrects += corrects
    #             acc_str = f"{sklearn_accuracy:.4f}"
    #             ece_str = f"{ece:.4f}"
            
    #         if sklearn_accuracy == -1:
    #             Total_acc_per_config.append(0)
    #         else:
    #             Total_acc_per_config.append(sklearn_accuracy * exit_percentage)
                
    #         # Store metrics for the FINAL stage
    #         current_key_inference_metrics[f"Threshold_Value_RF_{final_stage_num}"] = "FORCED EXIT" 
    #         current_key_inference_metrics[f"Samples_Exited_RF_{final_stage_num}"] = len(exited_at_final_stage)
    #         current_key_inference_metrics[f"Samples_Remaining_RF_{final_stage_num}"] = 0 
    #         current_key_inference_metrics[f"Accuracy_RF_{final_stage_num}"] = acc_str
    #         current_key_inference_metrics[f"Exit_Percentage_RF_{final_stage_num}"] = f"{exit_percentage:.4f}"
    #         current_key_inference_metrics[f"ECE_RF{final_stage_num}"] = ece_str
            
    #         print(f"**FORCED FINAL EXIT** stage {final_stage_num}: samples exited: {len(exited_at_final_stage)}")
            
    #         indices_being_processed_for_this_key = [] 
    #     else:
    #         # If no samples reached the final stage (already exited)
    #         # ... (Existing metric storage for 0 samples remains the same)
    #         pass

    #     # --------------------------------------------------------------------------------
    #     # 3. CONSOLIDATE METRICS AND PER-SAMPLE RESULTS
    #     # --------------------------------------------------------------------------------
    #     if all_corrects:
    #         model_ece = ECE_computation(all_confidences, all_corrects) 
    #         current_key_inference_metrics[f"Model_ECE"] = f"{model_ece:.4f}" 
    #     else:
    #         current_key_inference_metrics[f"Model_ECE"] = f"{0.0000:.4f}" 

    #     current_key_inference_metrics[f"Energy_USED"] = f"{sum(EnergyUsed_sum):.4f}" 
    #     current_key_inference_metrics[f"Total_acc"]= f"{sum(Total_acc_per_config):.4f}"
        
    #     # Finalize per-sample results
    #     for sample_dict in per_sample_results:
    #         sample_dict['correctness'] = (sample_dict['true_label'] == sample_dict['prediction'])

    #     all_keys_inference_metrics = [current_key_inference_metrics]

    #     # ... (Indices logic remains the same)
    #     final_passed_set = set(indices_being_processed_for_this_key) 
    #     self._final_passed_indices_after_check_exit = sorted(list(final_passed_set))
    #     self.indices_in = self._final_passed_indices_after_check_exit
    #     self.indices_out = sorted(list(self._all_exited_indices_after_check_exit))

    #     print("="*30)
    #     print("Overall Results from check_exit:")
    #     print(f"Total unique samples that exited: {len(self.indices_out)}")
    #     print(f"Total samples that passed all stages ('in'): {len(self.indices_in)}")
    #     print("="*30)
        
    #     # UPDATED RETURN: includes per_sample_results
    #     return self.indices_in, self.indices_out, all_keys_inference_metrics, per_sample_results