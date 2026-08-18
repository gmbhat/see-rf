import numpy as np
import pandas as pd
from ReadFile import LoadData
from sklearn.metrics import accuracy_score
import time
from Seq_RandomForest import SequentialRandomForest
from Inference_BO import RunInference
import os


def run_staged_srf(X_train, y_train, X_VAL, y_VAL, config_index,
                   overall_max_depth, overall_n_estimators, stages, tree_splits, random_state=42, logger=None):


    TIME_LOG_FILE = "Boosting_trainingTime\\training_times_NonBoosting.txt"
    current_config_training_metrics = {}
    final_weights = None
    n_features_total = X_train.shape[1]
    last_stage_model = None

    models_info = []

    if logger: logger.info(f"\n--- Starting Staged sRF Learning for Config {config_index} ---")
    start_time_total = time.time()


    number_of_tress_per_forest = []
    allocated_trees = 0
    for k in range(len(tree_splits)):
        # For the last forest, assign all remaining trees
        if k == len(tree_splits) - 1:
            num_estimators = overall_n_estimators - allocated_trees
        else:
            # Calculate trees for the current forest and update the count of allocated trees
            num_estimators = max(1,int(tree_splits[k] * overall_n_estimators)- allocated_trees)
            allocated_trees += num_estimators

        number_of_tress_per_forest.append(num_estimators)


    for i, percentage in enumerate(stages):
        stage_num = i + 1
        if logger: logger.info(f"\n--- Stage {stage_num} ({int(percentage*100)}% Initial Features) ---")
        start_time_stage = time.time()

        # --- Select Features: Take the first N% ---
        n_features_to_select = int(np.ceil(n_features_total * percentage))
        n_features_to_select = min(n_features_to_select, n_features_total)

        selected_feature_indices = np.arange(n_features_to_select)
        if logger: logger.info(f"  Using first {n_features_to_select} features (Indices 0 to {n_features_to_select-1}).")

        current_X_train_stage = X_train[:, selected_feature_indices]
        current_X_VAL_stage = X_VAL[:, selected_feature_indices]
        if logger: logger.info(f"  Training data shape for stage: {current_X_train_stage.shape}")

        srf_stage = SequentialRandomForest(
            n_estimators= number_of_tress_per_forest[i],
            max_depth = overall_max_depth,
            random_state = random_state,
            tree_splits = tree_splits,
            initial_class_weights=final_weights
        )

        # Train sRF for this stage
        if logger: logger.info(f"  Training sRF with {overall_n_estimators} estimators...")
        srf_stage.fit(current_X_train_stage, y_train)
        if logger: logger.info(f" *** number of trees: {len(srf_stage.trees_)}")

        # Evaluate
        y_pred_stage  = srf_stage.predict(current_X_VAL_stage)
        acc_stage = accuracy_score(y_VAL, y_pred_stage)
        if logger: logger.info(f"  Stage {stage_num} Test Accuracy: {acc_stage:.4f}")
     
        
        
        # Get final weights for the next stage
        final_weights = srf_stage.get_final_class_weights()
        if logger: logger.info(f"  Final class weights for next stage: { {k: round(v, 3) for k, v in final_weights.items()} }")

        models_info.append({
            'model': srf_stage,
            'features_indices': selected_feature_indices
        })
        end_time_stage = time.time()
        stage_duration = end_time_stage - start_time_stage
        if logger: logger.info(f"  Stage {stage_num} duration: {end_time_stage - start_time_stage:.2f} seconds")
        
        with open(TIME_LOG_FILE, "a") as f:
            f.write(f"Config {config_index}, Stage Random forest {stage_num}, Duration: {stage_duration:.2f} seconds\n")
 
        
        # Collect training metrics for the current stage into the local dictionary
        current_config_training_metrics[f"data_splits"] = stages
        current_config_training_metrics[f"tree_splits"] = tree_splits
        current_config_training_metrics[f"num_of_exits"] = len(stages)
        current_config_training_metrics[f"Num_Trees_per_RF-{stage_num}"] = number_of_tress_per_forest[i]
        current_config_training_metrics[f"train-acc-{stage_num}"] = f"{acc_stage:.4f}"
        current_config_training_metrics[f"Trainnig Duration-{stage_num}"] = f"{stage_duration:.2f}"

    end_time_total = time.time()
    if logger: logger.info(f"\n--- Staged Learning Complete for Config {config_index} ---")
    total_duration = end_time_total - start_time_total
    if logger: logger.info(f"Total duration for Config {config_index}: {end_time_total - start_time_total:.2f} seconds")
    current_config_training_metrics["Model_Size"] = sum(srf_stage.node_counts)
    with open(TIME_LOG_FILE, "a") as f:
        f.write(f"Config {config_index}, Total Duration for all the forests : {total_duration:.2f} seconds\n\n")
    current_config_training_metrics[f"Total Duration Trainig for all forest"] = f"{total_duration:.2f}"
   

    return models_info, current_config_training_metrics


def Run_orchestrator(dataset_name, max_depth, overall_n_estimators , split_points, config_index, logger , th_combination , tree_splits , BO_time):
    """
    Orchestrates a full experiment run (data loading, training, inference)
    for a single configuration and dataset.
    Returns a list of dictionaries, where each dictionary is a row of combined metrics.
    """
    if isinstance(th_combination, list) and len(th_combination) == 1 and isinstance(th_combination[0], list):
        th_combinations = th_combination[0]
    else:
        th_combinations = th_combination
    
    # Load data for the current dataset
    classData = LoadData()
    classData.Read(dataset_name)
    classData.SplitData()

    # Calculate overall dataset length and split percentages once
    total_dataset_length = classData.GetWindow()
    len_train = len(classData.GetTrainX())
    len_val = len(classData.GetValX())
    len_test = len(classData.GetTestX())

    # Calculate weights based on dataset split sizes
    if total_dataset_length == 0:
        weight_train = 0.0
        weight_val = 0.0
        weight_test = 0.0
        if logger: logger.warning("Total dataset length is zero. Weights for overall accuracy/energy will be zero.")
    else:
        weight_train = len_train / total_dataset_length
        weight_val = len_val / total_dataset_length
        weight_test = len_test / total_dataset_length

    split_weights = {
        "Train": weight_train,
        "Val": weight_val,
        "Test": weight_test
    }
    if logger: logger.info(f"Calculated split weights: {split_weights}")

    # Run staged sRF training
    all_model, current_config_training_metrics = run_staged_srf(
        classData.GetTrainX(), classData.GetYtrain(), classData.GetValX(), classData.GetYval(),
        config_index=config_index,
        overall_max_depth=max_depth,
        overall_n_estimators= overall_n_estimators,
        stages=split_points,
        tree_splits = tree_splits,
        random_state=42,
        logger=logger
    )

    # Create a SINGLE row for this configuration
    combined_config_row = {
        "Config_Index": config_index,
        "Overall_Max_Depth": max_depth,
        "Num_Trees_per_RF": overall_n_estimators,
        "Overall_Trees": overall_n_estimators * len(split_points),
        **current_config_training_metrics,
        "Threshold_Configuration": th_combinations,
        **{f"Threshold_Value_RF_{j+1}": f"{t_val:.4f}" for j, t_val in enumerate(th_combinations)}
    }

    # Run inference for each data type and add metrics to the single row
    datasets_for_inference = {
        "Train": (classData.GetTrainX(), classData.GetYtrain()),
        "Val": (classData.GetValX(), classData.GetYval()),
        "Test": (classData.GetTestX(), classData.GetYtest())
    }

    for data_type, (X_data, y_data) in datasets_for_inference.items():
        if logger: logger.info(f"\n--- Running Inference for {data_type} Data with Thresholds: {th_combinations} ---")
        inference_obj = RunInference(X_test=X_data, y_test=y_data, models=all_model, stages=split_points)

        sub_forest_entropy, prediction = inference_obj.predict_proba()

        _, _, all_keys_inference_metrics_list_from_check_exit = inference_obj.check_exit(
            sub_forest_entropy, th_combinations, prediction, y_data
        )
        
        # Take only the first (and should be only) result from check_exit
        key_inference_data_for_this_data_type = all_keys_inference_metrics_list_from_check_exit[0] if all_keys_inference_metrics_list_from_check_exit else {}

        all_stage_exit_accuracies = inference_obj.ExitAtAllStage()

        # Add data-type specific inference metrics
        for key, value in key_inference_data_for_this_data_type.items():
            if not (key == "Threshold_Configuration" or key.startswith("Threshold_Value_RF_")):
                combined_config_row[f"{key}_{data_type}"] = value

        for key, value in all_stage_exit_accuracies.items():
            combined_config_row[f"{key}_{data_type}"] = value

    # --- Calculate Weighted Overall Exit Percentage for each RF stage across all data types ---
    for stage_num in range(1, len(split_points) + 1):
        weighted_overall_exit_percentage_for_stage = 0.0
        total_weight_applied_percentage = 0.0
        
        for data_type, weight in split_weights.items():
            percentage_key = f"Exit_Percentage_RF_{stage_num}_{data_type}"
            
            exit_pct_val = combined_config_row.get(percentage_key)
            
            if exit_pct_val is not None and exit_pct_val != '':
                try:
                    exit_pct_float = float(exit_pct_val)
                    weighted_overall_exit_percentage_for_stage += (exit_pct_float * weight)
                    total_weight_applied_percentage += weight
                except ValueError:
                    if logger: logger.warning(f"Could not convert '{exit_pct_val}' for {percentage_key} to float. Skipping for weighted overall percentage.")
                    pass
        
        if total_weight_applied_percentage > 1e-9:
            combined_config_row[f"Weighted_Overall_Exit_Percentage_RF_{stage_num}"] = f"{weighted_overall_exit_percentage_for_stage:.4f}"
        else:
            combined_config_row[f"Weighted_Overall_Exit_Percentage_RF_{stage_num}"] = ''

    # --- Calculate Total Weighted Energy Used ---
    total_weighted_energy = 0.0
    total_weight_applied_energy = 0.0
    total_weighted_acc = 0.0
    total_weight_applied_acc = 0.0

    for data_type, weight in split_weights.items():
        energy_key = f"Energy_USED_{data_type}"
        accuracy_key = f"Total_acc_{data_type}"
        energy_val = combined_config_row.get(energy_key)
        accuracy_val = combined_config_row.get(accuracy_key)
        
        if energy_val is not None and energy_val != '':
            try:
                energy_float = float(energy_val)
                total_weighted_energy += (energy_float * weight)
                total_weight_applied_energy += weight
            except ValueError:
                if logger: logger.warning(f"Could not convert '{energy_val}' for {energy_key} to float. Skipping for total weighted energy.")
                pass
    
        if accuracy_val is not None and accuracy_val != '':
            try:
                accuracy_float = float(accuracy_val)
                total_weighted_acc += (accuracy_float * weight)
                total_weight_applied_acc += weight
            except ValueError:
                if logger: logger.warning(f"Could not convert '{accuracy_val}' for {accuracy_key} to float. Skipping for total weighted accuracy.")
                pass

    # Set final weighted metrics
    if total_weight_applied_energy > 1e-9:
        combined_config_row["data_total_Energy"] = f"{total_weighted_energy:.4f}"
    else:
        combined_config_row["data_total_Energy"] = f"{float('inf'):.4f}"

    if total_weight_applied_acc > 1e-9:
        combined_config_row["total_accurcy"] = f"{total_weighted_acc:.4f}"
    else:
        combined_config_row["total_accurcy"] = f"{float('-inf'):.4f}"
    
    combined_config_row["BO_time"] = BO_time

    if logger: logger.info(f"============================================================")
    if logger: logger.info(f"Finished model for Configuration #{config_index}\n\n")
    if logger: logger.info(f"============================================================ \n\n")

    # Return a list with just ONE row
    return all_model, [combined_config_row]


























# def Run_orchestrator(dataset_name, max_depth, overall_n_estimators , split_points, config_index, logger , th_combination , tree_splits , BO_time):
#     """
#     Orchestrates a full experiment run (data loading, training, inference)
#     for a single configuration and dataset.
#     Returns a list of dictionaries, where each dictionary is a row of combined metrics.
#     """
#     if isinstance(th_combination, list) and len(th_combination) == 1 and isinstance(th_combination[0], list):
#         th_combinations = th_combination[0]
#     else:
#         th_combinations = th_combination
    
#     # Load data for the current dataset
#     classData = LoadData()
#     classData.Read(dataset_name)
#     classData.SplitData()

#     # Calculate overall dataset length and split percentages once
#     total_dataset_length = classData.GetWindow()
#     len_train = len(classData.GetTrainX())
#     len_val = len(classData.GetValX())
#     len_test = len(classData.GetTestX())

#     # Calculate weights based on dataset split sizes
#     if total_dataset_length == 0:
#         weight_train = 0.0
#         weight_val = 0.0
#         weight_test = 0.0
#         if logger: logger.warning("Total dataset length is zero. Weights for overall accuracy/energy will be zero.")
#     else:
#         weight_train = len_train / total_dataset_length
#         weight_val = len_val / total_dataset_length
#         weight_test = len_test / total_dataset_length

#     split_weights = {
#         "Train": weight_train,
#         "Val": weight_val,
#         "Test": weight_test
#     }
#     if logger: logger.info(f"Calculated split weights: {split_weights}")

#     # Run staged sRF training
#     all_model, current_config_training_metrics = run_staged_srf(
#         classData.GetTrainX(), classData.GetYtrain(), classData.GetValX(), classData.GetYval(),
#         config_index=config_index,
#         overall_max_depth=max_depth,
#         overall_n_estimators= overall_n_estimators,
#         stages=split_points,
#         tree_splits = tree_splits,
#         random_state=42,
#         logger=logger
#     )

#     # Create a SINGLE row for this configuration
#     combined_config_row = {
#         "Config_Index": config_index,
#         "Overall_Max_Depth": max_depth,
#         "Num_Trees_per_RF": overall_n_estimators,
#         "Overall_Trees": overall_n_estimators * len(split_points),
#         **current_config_training_metrics,
#         "Threshold_Configuration": th_combinations,
#         **{f"Threshold_Value_RF_{j+1}": f"{t_val:.4f}" for j, t_val in enumerate(th_combinations)}
#     }

#     # Run inference for each data type and add metrics to the single row
#     datasets_for_inference = {
#         "Train": (classData.GetTrainX(), classData.GetYtrain()),
#         "Val": (classData.GetValX(), classData.GetYval()),
#         "Test": (classData.GetTestX(), classData.GetYtest())
#     }

#     for data_type, (X_data, y_data) in datasets_for_inference.items():
#         if logger: logger.info(f"\n--- Running Inference for {data_type} Data with Thresholds: {th_combinations} ---")
#         inference_obj = RunInference(X_test=X_data, y_test=y_data, models=all_model, stages=split_points)

#         sub_forest_entropy, prediction = inference_obj.predict_proba()

#         _, _, all_keys_inference_metrics_list_from_check_exit = inference_obj.check_exit(
#             sub_forest_entropy, th_combinations, prediction, y_data
#         )
        
#         # Take only the first (and should be only) result from check_exit
#         key_inference_data_for_this_data_type = all_keys_inference_metrics_list_from_check_exit[0] if all_keys_inference_metrics_list_from_check_exit else {}

#         all_stage_exit_accuracies = inference_obj.ExitAtAllStage()

#         # Add data-type specific inference metrics
#         for key, value in key_inference_data_for_this_data_type.items():
#             if not (key == "Threshold_Configuration" or key.startswith("Threshold_Value_RF_")):
#                 combined_config_row[f"{key}_{data_type}"] = value

#         for key, value in all_stage_exit_accuracies.items():
#             combined_config_row[f"{key}_{data_type}"] = value

#     # --- Calculate Weighted Overall Exit Percentage for each RF stage across all data types ---
#     for stage_num in range(1, len(split_points) + 1):
#         weighted_overall_exit_percentage_for_stage = 0.0
#         total_weight_applied_percentage = 0.0
        
#         for data_type, weight in split_weights.items():
#             percentage_key = f"Exit_Percentage_RF_{stage_num}_{data_type}"
            
#             exit_pct_val = combined_config_row.get(percentage_key)
            
#             if exit_pct_val is not None and exit_pct_val != '':
#                 try:
#                     exit_pct_float = float(exit_pct_val)
#                     weighted_overall_exit_percentage_for_stage += (exit_pct_float * weight)
#                     total_weight_applied_percentage += weight
#                 except ValueError:
#                     if logger: logger.warning(f"Could not convert '{exit_pct_val}' for {percentage_key} to float. Skipping for weighted overall percentage.")
#                     pass
        
#         if total_weight_applied_percentage > 1e-9:
#             combined_config_row[f"Weighted_Overall_Exit_Percentage_RF_{stage_num}"] = f"{weighted_overall_exit_percentage_for_stage:.4f}"
#         else:
#             combined_config_row[f"Weighted_Overall_Exit_Percentage_RF_{stage_num}"] = ''

#     # --- Calculate Total Weighted Energy Used ---
#     total_weighted_energy = 0.0
#     total_weight_applied_energy = 0.0
#     total_weighted_acc = 0.0
#     total_weight_applied_acc = 0.0

#     for data_type, weight in split_weights.items():
#         energy_key = f"Energy_USED_{data_type}"
#         accuracy_key = f"Total_acc_{data_type}"
#         energy_val = combined_config_row.get(energy_key)
#         accuracy_val = combined_config_row.get(accuracy_key)
        
#         if energy_val is not None and energy_val != '':
#             try:
#                 energy_float = float(energy_val)
#                 total_weighted_energy += (energy_float * weight)
#                 total_weight_applied_energy += weight
#             except ValueError:
#                 if logger: logger.warning(f"Could not convert '{energy_val}' for {energy_key} to float. Skipping for total weighted energy.")
#                 pass
    
#         if accuracy_val is not None and accuracy_val != '':
#             try:
#                 accuracy_float = float(accuracy_val)
#                 total_weighted_acc += (accuracy_float * weight)
#                 total_weight_applied_acc += weight
#             except ValueError:
#                 if logger: logger.warning(f"Could not convert '{accuracy_val}' for {accuracy_key} to float. Skipping for total weighted accuracy.")
#                 pass

#     # Set final weighted metrics
#     if total_weight_applied_energy > 1e-9:
#         combined_config_row["data_total_Energy"] = f"{total_weighted_energy:.4f}"
#     else:
#         combined_config_row["data_total_Energy"] = f"{float('inf'):.4f}"

#     if total_weight_applied_acc > 1e-9:
#         combined_config_row["total_accurcy"] = f"{total_weighted_acc:.4f}"
#     else:
#         combined_config_row["total_accurcy"] = f"{float('-inf'):.4f}"
    
#     combined_config_row["BO_time"] = BO_time

#     if logger: logger.info(f"============================================================")
#     if logger: logger.info(f"Finished model for Configuration #{config_index}\n\n")
#     if logger: logger.info(f"============================================================ \n\n")

#     # Return a list with just ONE row
#     return all_model, [combined_config_row]

























# # def Run_orchestrator(dataset_name, max_depth, overall_n_estimators , split_points, config_index, logger , th_combination , tree_splits , BO_time):
# #     """
# #     Orchestrates a full experiment run (data loading, training, inference)
# #     for a single configuration and dataset.
# #     Returns a list of dictionaries, where each dictionary is a row of combined metrics.
# #     """
# #     if isinstance(th_combination, list) and len(th_combination) == 1 and isinstance(th_combination[0], list):
# #         th_combinations = th_combination[0]
# #     else:
# #         # If it's already a single list of floats, or something unexpected, use it as is.
# #         # This depends on your exact intent for th_combination parameter.
# #         th_combinations = th_combination
# #     # Load data for the current dataset
# #     classData = LoadData()
# #     classData.Read(dataset_name)
# #     classData.SplitData()

# #     # Calculate overall dataset length and split percentages once
# #     total_dataset_length = classData.GetWindow()
# #     len_train = len(classData.GetTrainX())
# #     len_val = len(classData.GetValX())
# #     len_test = len(classData.GetTestX())

# #     # Calculate weights based on dataset split sizes
# #     if total_dataset_length == 0:
# #         weight_train = 0.0
# #         weight_val = 0.0
# #         weight_test = 0.0
# #         if logger: logger.warning("Total dataset length is zero. Weights for overall accuracy/energy will be zero.")
# #     else:
# #         weight_train = len_train / total_dataset_length
# #         weight_val = len_val / total_dataset_length
# #         weight_test = len_test / total_dataset_length

# #     split_weights = {
# #         "Train": weight_train,
# #         "Val": weight_val,
# #         "Test": weight_test
# #     }
# #     if logger: logger.info(f"Calculated split weights: {split_weights}")

# #     # Run staged sRF training
# #     all_model, current_config_training_metrics = run_staged_srf(
# #         classData.GetTrainX(), classData.GetYtrain(), classData.GetValX(), classData.GetYval(),
# #         config_index=config_index,
# #         overall_max_depth=max_depth,
# #         overall_n_estimators= overall_n_estimators,
# #         stages=split_points,
# #         tree_splits = tree_splits,
# #         random_state=42,
# #         logger=logger
# #     )

# #     results_to_return = []
# #     datasets_for_inference = {
# #         "Train": (classData.GetTrainX(), classData.GetYtrain()),
# #         "Val": (classData.GetValX(), classData.GetYval()),
# #         "Test": (classData.GetTestX(), classData.GetYtest())
# #     }

# #     # Generate threshold combinations once, as they are part of the configuration
# #     # inference_obj_for_thresholds = RunInference(X_test=classData.GetValX(), y_test=classData.GetYval(), models=all_model, stages=split_points)  #changed test to val for BO
    

# #     # Loop through each threshold key's inference metrics and prepare a row for each
# #     for key_idx, key_specific_thresholds in enumerate(th_combinations):
# #         # Create a new base row dictionary for each threshold configuration (key)
# #         combined_config_row = {
# #             "Config_Index": config_index,
# #             "Overall_Max_Depth": max_depth,
# #             "Num_Trees_per_RF": overall_n_estimators,
# #             "Overall_Trees" : overall_n_estimators * len(split_points) ,
# #             **current_config_training_metrics,
# #             "Threshold_Configuration":th_combinations, #str(key_specific_thresholds),
# #             # The outer loop already gives you 'key_specific_thresholds'
# #             # You want to iterate over *those specific thresholds* for the Threshold_Value_RF_ keys
# #             **{f"Threshold_Value_RF_{j+1}": f"{t_val:.4f}" for j, t_val in enumerate(th_combinations)}
# #         }

# #         # Now, loop through each data type for inference and add type-specific metrics
# #         for data_type, (X_data, y_data) in datasets_for_inference.items():
# #             if logger: logger.info(f"\n--- Running Inference for {data_type} Data with Thresholds: {key_specific_thresholds} ---")
# #             inference_obj = RunInference(X_test=X_data, y_test=y_data, models=all_model, stages=split_points)

# #             sub_forest_entropy, prediction = inference_obj.predict_proba()

# #             _, _, all_keys_inference_metrics_list_from_check_exit = inference_obj.check_exit(
# #                 sub_forest_entropy, th_combinations, prediction, y_data
# #             )
# #             key_inference_data_for_this_data_type = all_keys_inference_metrics_list_from_check_exit[0] if all_keys_inference_metrics_list_from_check_exit else {}

# #             all_stage_exit_accuracies = inference_obj.ExitAtAllStage()

# #             # Add *only* the data-type specific inference metrics, with data_type suffix
# #             for key, value in key_inference_data_for_this_data_type.items():
# #                 if not (key == "Threshold_Configuration" or key.startswith("Threshold_Value_RF_")):
# #                     combined_config_row[f"{key}_{data_type}"] = value

# #             for key, value in all_stage_exit_accuracies.items():
# #                 combined_config_row[f"{key}_{data_type}"] = value

# #         # --- Calculate Weighted Overall Exit Percentage for each RF stage across all data types ---
# #         for stage_num in range(1, len(split_points) + 1):
# #             weighted_overall_exit_percentage_for_stage = 0.0
# #             total_weight_applied_percentage = 0.0
            
# #             for data_type, weight in split_weights.items():
# #                 percentage_key = f"Exit_Percentage_RF_{stage_num}_{data_type}"
                
# #                 exit_pct_val = combined_config_row.get(percentage_key)
                
# #                 if exit_pct_val is not None and exit_pct_val != '':
# #                     try:
#                         exit_pct_float = float(exit_pct_val)
#                         weighted_overall_exit_percentage_for_stage += (exit_pct_float * weight)
#                         total_weight_applied_percentage += weight
#                     except ValueError:
#                         if logger: logger.warning(f"Could not convert '{exit_pct_val}' for {percentage_key} to float. Skipping for weighted overall percentage.")
#                         pass
            
#             if total_weight_applied_percentage > 1e-9:
#                 combined_config_row[f"Weighted_Overall_Exit_Percentage_RF_{stage_num}"] = f"{weighted_overall_exit_percentage_for_stage:.4f}"
#             else:
#                 combined_config_row[f"Weighted_Overall_Exit_Percentage_RF_{stage_num}"] = ''

#         # --- Calculate Total Weighted Energy Used ---
#         total_weighted_energy = 0.0
#         total_weight_applied_energy = 0.0

#         total_weighted_acc = 0.0
#         total_weight_applied_acc = 0.0

#         for data_type, weight in split_weights.items():
#             energy_key = f"Energy_USED_{data_type}"
#             accuracy_key =f"Total_acc_{data_type}"
#             energy_val = combined_config_row.get(energy_key)
#             accuracy_val = combined_config_row.get(accuracy_key)
            
#             if energy_val is not None and energy_val != '':
#                 try:
#                     energy_float = float(energy_val)
#                     total_weighted_energy += (energy_float * weight)
#                     total_weight_applied_energy += weight
#                 except ValueError:
#                     if logger: logger.warning(f"Could not convert '{energy_val}' for {energy_key} to float. Skipping for total weighted energy.")
#                     pass
        
        
#             if accuracy_val is not None and accuracy_val != '':
#                 try:
#                     accuarcy_float = float(accuracy_val)
#                     total_weighted_acc += (accuarcy_float * weight)
#                     total_weight_applied_acc += weight
#                 except ValueError:
#                     if logger: logger.warning(f"Could not convert '{accuracy_val}' for {accuracy_key} to float. Skipping for total weighted energy.")
#                     pass
        
        
        
        
#         # Ensure these keys are always present, even if with default values
#         if total_weight_applied_energy > 1e-9:
#             combined_config_row["data_total_Energy"] = f"{total_weighted_energy:.4f}" # Changed to Energy_USED_Val to match BysianOptimization.py
#         else:
#             combined_config_row["data_total_Energy"] = f"{float('inf'):.4f}" # Or some other large number indicating bad energy usage

#         if total_weight_applied_acc > 1e-9:
#             combined_config_row["total_accurcy"] = f"{total_weighted_acc:.4f}" # Changed to Total_acc_Val to match BysianOptimization.py
#         else:
#             combined_config_row["total_accurcy"] = f"{float('-inf'):.4f}" # Or 0.0 indicating very poor accuracy
        
        
        
#         combined_config_row["inference_time"] = BO_time
     



#         results_to_return.append(combined_config_row)

#     if logger: logger.info(f"============================================================")
#     if logger: logger.info(f"Finished model for Configuration #{config_index}\n\n")
#     if logger: logger.info(f"============================================================ \n\n")

#     return all_model , results_to_return


