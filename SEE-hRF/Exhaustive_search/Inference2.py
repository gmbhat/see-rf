import numpy as np
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import accuracy_score
from sklearn.utils.multiclass import unique_labels 
from ECE import ECE_computation , calculate_confidence_max_prob
import concurrent.futures , time

def _predict_for_tree(args):
    tree_batch, X_test_temp, n_samples, classes_ = args
    scores_per_batch = np.zeros((n_samples, len(classes_)))
    
    for tree in tree_batch:
        predictions = tree.predict(X_test_temp)
        for sample_idx in range(n_samples):
            predicted_class = predictions[sample_idx]
            if predicted_class in classes_:
                class_idx = np.where(classes_ == predicted_class)[0][0]
                scores_per_batch[sample_idx, class_idx] += 1   
    return scores_per_batch



class RunInference:
    def __init__(self, X_test, y_test, models , stages):
        
        self.X_test = X_test
        self.y_test = y_test
        self.trees_ = None
        self.n_samples = self.X_test.shape[0]
        self.classes_ = None
        self.models = models
        self.stages = stages
        self.srf_entropy = None 
        self._final_passed_indices_after_check_exit = [] 
        self._all_exited_indices_after_check_exit = set() 

        self.indices_in = []
        self.indices_out = []
    
    def entropy(self , probabilities):  
        epsilon = 1e-5  # to avoid taking the logarithm of zero
        return -np.sum(probabilities * np.log(probabilities + epsilon), axis=1)
    
    def predict_proba(self):

        self.classes_ = unique_labels(self.y_test)
        self.n_classes_ = len(self.classes_)
        n_features_total = self.X_test.shape[1]
        self.all_scores = np.zeros((self.n_samples, self.n_classes_))
        self.all_entropy = []
        self.all_predict = []
        self.connected_prediction = []
        self.list_of_Prob = []
        
        BATCH_SIZE = 10
        for j in range(len(self.models)):
        
            n_features_to_select = int(np.ceil(n_features_total * self.stages[j])) # Use ceil to ensure at least 1
            n_features_to_select = min(n_features_to_select, n_features_total) # Cap
            selected_feature_indices = np.arange(n_features_to_select)
        
            X_test_temp = self.X_test[:, selected_feature_indices]
        
            srf = self.models[j]['model']
            self.trees_ = srf.trees_ # Assuming srf.trees_ is populated by srf.fit()
            
            scores = np.zeros((self.n_samples, self.n_classes_))  
            
            #this the old version which do the prediction sequentialy
            
            # for i, tree in enumerate(self.trees_):
            #     # Assuming tree.predict is available and works on X_test_temp
            #     predictions = tree.predict(X_test_temp) 
            #     for sample_idx in range(self.n_samples):
            #         predicted_class = predictions[sample_idx]
            #         if predicted_class in self.classes_:
            #             class_idx = np.where(self.classes_ == predicted_class)[0][0]
            #             scores[sample_idx, class_idx] += 1

            # self.all_scores += scores
             
            
            # do the prediction parallel  --> new version
            # Divide the trees into batches
            tree_batches = [self.trees_[i:i + BATCH_SIZE] for i in range(0, len(self.trees_), BATCH_SIZE)]
            
            # Prepare arguments for each batch
            list_args = [(batch, X_test_temp, self.n_samples, self.classes_) for batch in tree_batches]

            # tree_scores_list = Parallel(n_jobs=-1, verbose=10)(delayed(_predict_for_tree)(tree, X_test_temp, self.n_samples, self.classes_) for tree in self.trees_ )
            
            
            
                
            max_workers = min(len(list_args), 4)   
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                outputs = list(executor.map(_predict_for_tree, list_args))
            
              
                
            # with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(list_args), os.cpu_count() or 4)) as executor:
            #         outputs = list(executor.map(_predict_for_tree, list_args)) 
            for scores_per_tree in outputs:
                scores += scores_per_tree

            self.all_scores += scores
             
            self.connected_prediction.append(np.argmax(self.all_scores , axis=1))
            
            denominator = len(self.trees_) * (j + 1)
            
            proba = self.all_scores / denominator
            self.list_of_Prob.append(proba)
            self.srf_entropy =  self.entropy(proba)
            
            self.all_entropy.append( self.srf_entropy)
            self.all_predict.append(np.argmax(scores , axis=1))
        
        return self.all_entropy , self.all_predict  
    
    
    
    def ExitAtAllStage(self):
        
        accuracies_exit_all = {}
        
        for j , prediction_per_stage in enumerate(self.connected_prediction):
            
            accuracy = accuracy_score(self.y_test, prediction_per_stage)
            accuracies_exit_all[f"accuracy_exit_all_Sample_RF {j+1}"] = f"{accuracy:.4f}"
        
        return accuracies_exit_all


    def check_exit(self , sub_rf_entropy , threshold_list_of_keys , predictions , y_test):
        
       
        # threshold_list_of_keys is the ONE list of N-1 threshold values
        
        ece_per_exit = []         
        all_confidences = []
        all_corrects = []
        
        if sub_rf_entropy is None:
            print("Error in check_exit: sub_rf_entropy is not available.")
            return [], [], []

        # N = Total number of Random Forests (stages)
        num_total_stages = len(self.models) 
        # N-1 = Number of thresholds provided
        num_thresholds = len(threshold_list_of_keys) 
        
        if num_thresholds != num_total_stages - 1:
            print(f"Error: Expected {num_total_stages - 1} thresholds, got {num_thresholds}. Exiting.")
            return [], [], []

        if len(sub_rf_entropy) != num_total_stages:
            print(f"Error in check_exit: Expected {num_total_stages} entropy arrays, got {len(sub_rf_entropy)}.")
            return [], [], []

        num_total_samples = len(sub_rf_entropy[0])
        all_original_indices = list(range(num_total_samples))

        print(f"check_exit: Initial number of samples: {num_total_samples}")
        print(f"--- Processing Threshold Configuration: {threshold_list_of_keys} ---")

        indices_being_processed_for_this_key = list(all_original_indices) 
        exited_samples_this_key_cumulative = set()

        current_key_inference_metrics = {}
        current_key_inference_metrics["Threshold_Configuration"] = str(threshold_list_of_keys) 

        EnergyUsed_sum = []
        Total_acc_per_config = []
        
        # --------------------------------------------------------------------------------
        # 1. PROCESS INTERMEDIATE, THRESHOLDED EXITS (RF 1 to RF N-1)
        # The loop runs N-1 times, once for each threshold value provided.
        # threshold_list_of_keys is used directly here.
        # --------------------------------------------------------------------------------
        for threshold_j_idx, threshold_value in enumerate(threshold_list_of_keys):
            
            # Check if there are any samples left from the previous stage
            if not indices_being_processed_for_this_key:
                print(f"Exit stage {threshold_j_idx + 1} (Threshold: {threshold_value:.4f}): No samples left to process. Skipping.")
                
                # Fill placeholder metrics for remaining unvisited stages (RF_{j+1} through RF_{N-1})
                for k in range(threshold_j_idx, num_thresholds):
                    stage_num = k + 1
                    current_key_inference_metrics[f"Threshold_Value_RF_{stage_num}"] = f"{threshold_list_of_keys[k]:.4f}"
                    current_key_inference_metrics[f"Samples_Exited_RF_{stage_num}"] = 0
                    current_key_inference_metrics[f"Samples_Remaining_RF_{stage_num}"] = 0
                    current_key_inference_metrics[f"Accuracy_RF_{stage_num}"] = f"{-1.0000:.4f}"
                    current_key_inference_metrics[f"Exit_Percentage_RF_{stage_num}"] = f"{0.0000:.4f}"
                    current_key_inference_metrics[f"ECE_RF{stage_num}"] = f"{-1.0000:.4f}"
                break 

            exited_at_this_stage = []
            passed_this_stage = []
            
            # Check samples against the current threshold
            for original_sample_index in indices_being_processed_for_this_key:
                # sub_rf_entropy[threshold_j_idx] corresponds to RF index j, i.e., RF_{j+1}
                sample_entropy_value = sub_rf_entropy[threshold_j_idx][original_sample_index]
                
                if sample_entropy_value < threshold_value:
                    exited_at_this_stage.append(original_sample_index)
                    exited_samples_this_key_cumulative.add(original_sample_index) 
                else:
                    passed_this_stage.append(original_sample_index)

            # Samples that pass this stage move to the next iteration
            indices_being_processed_for_this_key = passed_this_stage 
            
            # --- Calculate Metrics for this EXIT (RF_{j+1}) ---
            stage_num = threshold_j_idx + 1
            exit_percentage = float(len(exited_at_this_stage) / num_total_samples)
            EnergyUsed_sum.append(float(exit_percentage * self.stages[threshold_j_idx]))
            
            subset_predictions = predictions[threshold_j_idx][exited_at_this_stage]
            subset_true_labels = y_test[exited_at_this_stage]
            
            print(f"Exit stage {stage_num} (Threshold Value: {threshold_value:.4f}): samples exited: {len(exited_at_this_stage)}")
    
            if len(subset_predictions) == 0:
                sklearn_accuracy = -1
                ece = -1
                acc_str = f"{-1.0000:.4f}"
                ece_str = f"{-1.0000:.4f}"
            else:
                sklearn_accuracy = accuracy_score(subset_true_labels, subset_predictions)
                corrects = (subset_true_labels == subset_predictions).astype(int).tolist()
                confidences = calculate_confidence_max_prob(self.list_of_Prob[threshold_j_idx][exited_at_this_stage])
                ece = ECE_computation(confidences, corrects, 10)
                ece_per_exit.append(ece)
                all_confidences += confidences
                all_corrects += corrects
                acc_str = f"{sklearn_accuracy:.4f}"
                ece_str = f"{ece:.4f}"
                    
            if sklearn_accuracy == -1:
                Total_acc_per_config.append(0)
            else:
                Total_acc_per_config.append(sklearn_accuracy * exit_percentage)
                    
            # Store metrics for this specific RF stage (RF_{j+1})
            current_key_inference_metrics[f"Threshold_Value_RF_{stage_num}"] = f"{threshold_value:.4f}"
            current_key_inference_metrics[f"Samples_Exited_RF_{stage_num}"] = len(exited_at_this_stage)
            current_key_inference_metrics[f"Samples_Remaining_RF_{stage_num}"] = len(indices_being_processed_for_this_key)
            current_key_inference_metrics[f"Accuracy_RF_{stage_num}"] = acc_str
            current_key_inference_metrics[f"Exit_Percentage_RF_{stage_num}"] = f"{exit_percentage:.4f}"
            current_key_inference_metrics[f"ECE_RF{stage_num}"] = ece_str
            
        # --------------------------------------------------------------------------------
        # 2. FORCED EXIT AT THE FINAL RF STAGE (RF N)
        # This block handles the last RF (index N-1) where remaining samples are forced to exit.
        # --------------------------------------------------------------------------------

        final_rf_idx = num_total_stages - 1 
        final_stage_num = num_total_stages 
        
        exited_at_final_stage = indices_being_processed_for_this_key 
        
        if exited_at_final_stage:
            
            subset_predictions = predictions[final_rf_idx][exited_at_final_stage]
            subset_true_labels = y_test[exited_at_final_stage]
            
            self._all_exited_indices_after_check_exit.update(exited_at_final_stage)
            exited_samples_this_key_cumulative.update(exited_at_final_stage) 
            
            exit_percentage = float(len(exited_at_final_stage) / num_total_samples)
            EnergyUsed_sum.append(float(exit_percentage * self.stages[final_rf_idx]))
            
            # --- Calculate Metrics for the FINAL EXIT (RF_N) ---
            if len(subset_predictions) == 0:
                sklearn_accuracy = -1
                ece = -1
                acc_str = f"{-1.0000:.4f}"
                ece_str = f"{-1.0000:.4f}"
            else:
                sklearn_accuracy = accuracy_score(subset_true_labels, subset_predictions)
                corrects = (subset_true_labels == subset_predictions).astype(int).tolist()
                confidences = calculate_confidence_max_prob(self.list_of_Prob[final_rf_idx][exited_at_final_stage])
                ece = ECE_computation(confidences, corrects, 10)
                ece_per_exit.append(ece)
                all_confidences += confidences
                all_corrects += corrects
                acc_str = f"{sklearn_accuracy:.4f}"
                ece_str = f"{ece:.4f}"
            
            if sklearn_accuracy == -1:
                Total_acc_per_config.append(0)
            else:
                Total_acc_per_config.append(sklearn_accuracy * exit_percentage)
                
            # Store metrics for the FINAL stage
            current_key_inference_metrics[f"Threshold_Value_RF_{final_stage_num}"] = "FORCED EXIT" 
            current_key_inference_metrics[f"Samples_Exited_RF_{final_stage_num}"] = len(exited_at_final_stage)
            current_key_inference_metrics[f"Samples_Remaining_RF_{final_stage_num}"] = 0 
            current_key_inference_metrics[f"Accuracy_RF_{final_stage_num}"] = acc_str
            current_key_inference_metrics[f"Exit_Percentage_RF_{final_stage_num}"] = f"{exit_percentage:.4f}"
            current_key_inference_metrics[f"ECE_RF{final_stage_num}"] = ece_str
            
            print(f"**FORCED FINAL EXIT** stage {final_stage_num}: samples exited: {len(exited_at_final_stage)}")
            
            indices_being_processed_for_this_key = [] 
        else:
            # If no samples reached the final stage (already exited)
            current_key_inference_metrics[f"Threshold_Value_RF_{final_stage_num}"] = "FORCED EXIT (0 samples)"
            current_key_inference_metrics[f"Samples_Exited_RF_{final_stage_num}"] = 0
            current_key_inference_metrics[f"Samples_Remaining_RF_{final_stage_num}"] = 0
            current_key_inference_metrics[f"Accuracy_RF_{final_stage_num}"] = f"{0.0000:.4f}"
            current_key_inference_metrics[f"Exit_Percentage_RF_{final_stage_num}"] = f"{0.0000:.4f}"
            current_key_inference_metrics[f"ECE_RF{final_stage_num}"] = f"{0.0000:.4f}"
            print(f"**FORCED FINAL EXIT** stage {final_stage_num}: No samples remaining.")


        # --------------------------------------------------------------------------------
        # 3. CONSOLIDATE METRICS 
        # --------------------------------------------------------------------------------
        if all_corrects:
            model_ece = ECE_computation(all_confidences, all_corrects) 
            current_key_inference_metrics[f"Model_ECE"] = f"{model_ece:.4f}" 
        else:
            current_key_inference_metrics[f"Model_ECE"] = f"{0.0000:.4f}" 

        current_key_inference_metrics[f"Energy_USED"] = f"{sum(EnergyUsed_sum):.4f}" 
        current_key_inference_metrics[f"Total_acc"]= f"{sum(Total_acc_per_config):.4f}"

        # Since we removed the outer loop, all_keys_inference_metrics will be a list containing one dictionary
        all_keys_inference_metrics = [current_key_inference_metrics]

        # Indices that passed all stages (should be an empty set here)
        final_passed_set = set(indices_being_processed_for_this_key) 
        
        self._final_passed_indices_after_check_exit = sorted(list(final_passed_set))
        
        self.indices_in = self._final_passed_indices_after_check_exit
        # The set of exited samples is cumulative across the single configuration
        self.indices_out = sorted(list(self._all_exited_indices_after_check_exit))

        print("="*30)
        print("Overall Results from check_exit:")
        print(f"Total unique samples that exited: {len(self.indices_out)}")
        print(f"Total samples that passed all stages ('in'): {len(self.indices_in)}")
        print("="*30)
        
        return self.indices_in, self.indices_out, all_keys_inference_metrics
       
    # def check_exit(self , sub_rf_entropy , threshold_list_of_keys , predictions , y_test):
        
    #     ece_per_exit = []         
    #     overall_ece = []
    #     all_confidences = []
    #     all_corrects = []
        
    #     if sub_rf_entropy is None:
    #         print("Error in check_exit: sub_rf_entropy is not available.")
    #         print("Please ensure predict_proba() has been called and successfully populated sub_rf_entropy.")
    #         self.indices_in = []
    #         self.indices_out = []
    #         return [], [], [] # Return empty list for metrics

    #     if len(sub_rf_entropy) == 0:
    #         print("Warning in check_exit: sub_rf_entropy is empty. No samples to check.")
    #         self.indices_in = []
    #         self.indices_out = []
    #         return [], [], [] # Return empty list for metrics

    #     num_total_samples = len(sub_rf_entropy[0])
    #     all_original_indices = list(range(num_total_samples))

    #     print(f"check_exit: Initial number of samples: {num_total_samples}\n")

    #     passed_indices_for_each_key = [] 
    #     self._all_exited_indices_after_check_exit = set()
        
    #     # This will store a list of dictionaries, one for each threshold key
    #     all_keys_inference_metrics = [] 

    #     for key_idx, key_specific_thresholds in enumerate(threshold_list_of_keys):
    #         threshold_strings =  key_specific_thresholds
    #         print(f"--- Processing Key {key_idx + 1} (Thresholds: {threshold_strings}) ---")
    #         print(f"    (Starting with initial {num_total_samples} samples for this key)")

    #         indices_being_processed_for_this_key = list(all_original_indices) 
    #         exited_samples_this_key_cumulative = set()

    #         # Dictionary to store inference metrics for the CURRENT key
    #         current_key_inference_metrics = {}
    #         current_key_inference_metrics["Threshold_Configuration"] = str(key_specific_thresholds) # No Key_X suffix here

    #         EnergyUsed_sum = []
    #         Total_acc_per_config = []
            
    #         for threshold_j_idx, threshold_value in enumerate(key_specific_thresholds):
                
    #             is_last = (threshold_j_idx ==(len(key_specific_thresholds) - 1))
                
    #             # Add the threshold value for this specific RF stage within this key
    #             current_key_inference_metrics[f"Threshold_Value_RF_{threshold_j_idx+1}"] = f"{threshold_value:.4f}" # No Key_X suffix

    #             if is_last:  # FIX THE CORRECT INDIXES
    #                 sklearn_accuracy = accuracy_score(y_test[passed_this_stage], predictions[threshold_j_idx][passed_this_stage])  #????????????
    #                 print(f"Accuracy for the last subset {threshold_j_idx+1}: {sklearn_accuracy:.4f}")
                    
    #                 exit_percentage = float( len(passed_this_stage)/ num_total_samples)
    #                 if exit_percentage == 0:
    #                     Total_acc_per_config.append(0) 
    #                     sklearn_accuracy = -1
    #                     ece_per_exit.append(-1)
    #                     ece = -1
    #                     current_key_inference_metrics[f"ECE_RF{threshold_j_idx+1}"] = -1
                        
    #                 else:
    #                     Total_acc_per_config.append(sklearn_accuracy * exit_percentage)   
    #                     #calculate ECE per exit:
    #                     corrects = (y_test[passed_this_stage] == predictions[threshold_j_idx][passed_this_stage]).astype(int).tolist()
    #                     confidences = calculate_confidence_max_prob(self.list_of_Prob[threshold_j_idx][passed_this_stage]) # a list of list
    #                     ece = ECE_computation(confidences, corrects, 10)
    #                     ece_per_exit.append(ece)
    #                     all_confidences += confidences
    #                     all_corrects += corrects
                                            
                    
    #                 EnergyUsed_sum.append(float(exit_percentage * self.stages[threshold_j_idx]))
                    
    #                 # Capture metrics for the last stage as well
    #                 current_key_inference_metrics[f"Accuracy_RF_{threshold_j_idx+1}"] = f"{sklearn_accuracy:.4f}"
    #                 current_key_inference_metrics[f"Samples_Exited_RF_{threshold_j_idx+1}"] = len(indices_being_processed_for_this_key) # All remaining exit here
    #                 current_key_inference_metrics[f"Samples_Remaining_RF_{threshold_j_idx+1}"] = 0
    #                 current_key_inference_metrics[f"Exit_Percentage_RF_{threshold_j_idx+1}"] = f"{exit_percentage:.4f}"
    #                 current_key_inference_metrics[f"ECE_RF{threshold_j_idx+1}"] = f"{ece:.4f}"
                    
                    
                    
    #                 # Entropy values for the last stage would be for the remaining samples
    #                 # entropy_values_for_last_stage = sub_rf_entropy[threshold_j_idx][indices_being_processed_for_this_key]
    #                 # current_key_inference_metrics[f"Entropy_Values_RF_{threshold_j_idx+1}"] = ';'.join(map(str, entropy_values_for_last_stage)) if len(entropy_values_for_last_stage) > 0 else ''
                    
    #             else:
    #                 if not indices_being_processed_for_this_key:
    #                     exit_percentage = float(len(exited_at_this_stage)/ num_total_samples)

    #                     print(f"  Key {key_idx+1}, exit stage {threshold_j_idx + 1} (Threshold: {threshold_value:.4f}):")
    #                     print(f"    No samples left to process from previous stage within this key. Skipping.")
    #                     # Still add placeholder entries for consistency
    #                     current_key_inference_metrics[f"Samples_Exited_RF_{threshold_j_idx+1}"] = 0
    #                     current_key_inference_metrics[f"Samples_Remaining_RF_{threshold_j_idx+1}"] = 0
    #                     current_key_inference_metrics[f"Accuracy_RF_{threshold_j_idx+1}"] = ''
    #                     current_key_inference_metrics[f"Exit_Percentage_RF_{threshold_j_idx+1}"] = f"{exit_percentage:.4f}"
    #                     current_key_inference_metrics[f"ECE_RF{threshold_j_idx+1}"] = ''
    #                     # current_key_inference_metrics[f"Entropy_Values_RF_{threshold_j_idx+1}"] = ''
    #                     break 

    #                 exited_at_this_stage = []
    #                 passed_this_stage = []
                    
    #                 for original_sample_index in indices_being_processed_for_this_key:
    #                     sample_entropy_value = sub_rf_entropy[threshold_j_idx][original_sample_index]
                        
    #                     if sample_entropy_value < threshold_value:
    #                         exited_at_this_stage.append(original_sample_index)
    #                         self._all_exited_indices_after_check_exit.add(original_sample_index) 
    #                         exited_samples_this_key_cumulative.add(original_sample_index) 
    #                     else:
    #                         passed_this_stage.append(original_sample_index)

    #                 subset_predictions = predictions[threshold_j_idx][exited_at_this_stage]
    #                 subset_true_labels = y_test[exited_at_this_stage]
                                    
    #                 temp_indices_processed = indices_being_processed_for_this_key
    #                 indices_being_processed_for_this_key = passed_this_stage 
    #                 exit_percentage = float(len(exited_at_this_stage)/ num_total_samples)
                    
                    
    #                 EnergyUsed_sum.append(float(exit_percentage * self.stages[threshold_j_idx]))
                    
    #                 print(f"  Key {key_idx+1}, exit stage {threshold_j_idx+1} (Threshold Value: {threshold_value:.4f}):")
    #                 print(f"    samples were exit : {len(exited_at_this_stage)}")
    #                 print(f"    samples were passed to next (stage/forest within this key) : {len(indices_being_processed_for_this_key)}")
             
                    
    #                 if len(subset_predictions) == 0:
    #                     sklearn_accuracy = -1
    #                     print(f"\nAccuracy for the subset: No samples in sub rf {threshold_j_idx+1} to calculate accuracy.")
    #                     ece = -1
    #                 else:
    #                     sklearn_accuracy = accuracy_score(subset_true_labels, subset_predictions)
    #                     print(f"Accuracy for the subset {threshold_j_idx+1}: {sklearn_accuracy:.4f}")
                        
    #                     #calculate ECE
                        
    #                     corrects = (subset_true_labels == subset_predictions).astype(int).tolist()
    #                     confidences = calculate_confidence_max_prob(self.list_of_Prob[threshold_j_idx][exited_at_this_stage]) # a list of list
    #                     ece = ECE_computation(confidences, corrects, 10)
    #                     ece_per_exit.append(ece)
    #                     all_confidences += confidences
    #                     all_corrects += corrects
                        
                        
                        
    #                 if  sklearn_accuracy == -1:
                        
    #                     Total_acc_per_config.append(0)
    #                 else:
    #                     Total_acc_per_config.append(sklearn_accuracy * exit_percentage)
                        
    #                 # Store metrics for this specific RF stage within this key
    #                 current_key_inference_metrics[f"Samples_Exited_RF_{threshold_j_idx+1}"] = len(exited_at_this_stage)
    #                 current_key_inference_metrics[f"Samples_Remaining_RF_{threshold_j_idx+1}"] = len(indices_being_processed_for_this_key)
    #                 current_key_inference_metrics[f"Accuracy_RF_{threshold_j_idx+1}"] = f"{sklearn_accuracy:.4f}"
    #                 current_key_inference_metrics[f"Exit_Percentage_RF_{threshold_j_idx+1}"] = f"{exit_percentage:.4f}"
    #                 current_key_inference_metrics[f"ECE_RF{threshold_j_idx+1}"] = f"{ece:.4f}"
    #                 # entropy_values_for_stage = sub_rf_entropy[threshold_j_idx][exited_at_this_stage]
    #                 # current_key_inference_metrics[f"Entropy_Values_RF_{threshold_j_idx+1}"] = ';'.join(map(str, entropy_values_for_stage)) if len(entropy_values_for_stage) > 0 else ''
                    
                    
    #         model_ece = ECE_computation(all_confidences, all_corrects)    
    #         current_key_inference_metrics[f"Model_ECE"] = model_ece 
    #         current_key_inference_metrics[f"Energy_USED"] = (sum(EnergyUsed_sum))       
    #         current_key_inference_metrics[f"Total_acc"]  = (sum(Total_acc_per_config))
    #         # After all thresholds for the current key are processed, add this key's metrics to the list
    #         all_keys_inference_metrics.append(current_key_inference_metrics)

    #         passed_indices_for_each_key.append(set(indices_being_processed_for_this_key))
            
    #         print(f"--- Finished Key {key_idx + 1} ---")
    #         print(f"  Samples passed ALL thresholds of this Key: {len(indices_being_processed_for_this_key)}")
    #         print(f"  Total samples exited during this Key's processing: {len(exited_samples_this_key_cumulative)}\n")
            
    #         if key_idx < len(threshold_list_of_keys) -1 :
    #              print("finished one threshold list (key), processing next list (key) with initial samples...")
    #         else:
    #              print("finished all threshold lists (keys).")
                 
    #     # After all keys are processed, determine final 'in' and 'out'
    #     if not passed_indices_for_each_key:
    #         self._final_passed_indices_after_check_exit = []
    #     else:
    #         final_passed_set = set(passed_indices_for_each_key[0])
    #         for i in range(1, len(passed_indices_for_each_key)):
    #             final_passed_set.intersection_update(passed_indices_for_each_key[i])
    #         self._final_passed_indices_after_check_exit = sorted(list(final_passed_set))
        
    #     self.indices_in = self._final_passed_indices_after_check_exit
    #     self.indices_out = sorted(list(self._all_exited_indices_after_check_exit))

    #     print("="*30)
    #     print("Overall Results from check_exit:")
    #     print(f"Total unique samples that exited at any point (final 'out'): {len(self.indices_out)}")
    #     print(f"Total samples that passed all keys (final 'in'): {len(self.indices_in)}")
    #     print("="*30)
        
    #     return self.indices_in, self.indices_out, all_keys_inference_metrics

