import concurrent.futures
import itertools
import csv
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import argparse
import logging
import pprint
import ast
import time
import datetime
import math
import itertools
import random
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical, Space
from skopt.utils import use_named_args
import copy
import pickle
from sklearn.model_selection import train_test_split
from ax.service.ax_client import AxClient
from ax.service.utils.instantiation import ObjectiveProperties
from SaveData_BO import SaveToCSV
import os
from updateClassWeight_lastTree_BO import Run_orchestrator
from Inference_BO import RunInference
from ReadFile import LoadData

RF_cache = {}

pp = pprint.PrettyPrinter(indent=2)
device = torch.device('cpu')
torch.set_num_threads(1)
random.seed(42)
np.random.seed(42)

# --- Setup logging ---
LOG_FILE = f"LogFolder\\parallelized_experiment_run_{time.strftime('%Y%m%d-%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s]: %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()]
)
def run_and_record_experiment(config):
    """
    Runs a single experiment for a given configuration and dataset.
    This function will be executed in parallel processes.
    It calls the Run_orchestrator and returns its results.
    """
    dataset_name = config['Dataset']
    split_points = config['Proportions']
    max_depth = config['Tree_depth']
    n_estimators = config['n_estimators'] # Although Run_orchestrator currently uses a fixed 100EpilepsyEpilepsy
    clf = config['clf']
    current_config_training_metrics=config['current_config_training_metrics'] 
    th_combinations = config['th']
    
    
    # Use a unique index that spans all configurations, not just the batch
    # unique_config_idx = config_index_in_batch 

    # It's good practice to get a logger specific to the process
    # This prevents multiprocessing issues with shared logging handlers if not configured carefully.
    # For simple logging setup as above, this might not be strictly necessary, but good for robustness.
    logger = logging.getLogger(f"Process-{os.getpid()}")
    logger.setLevel(logging.INFO) # Ensure logger level is set for this process
    
    logger.info(f"Starting experiment for Dataset: {dataset_name}")
    logger.info(f"  Split Points: {split_points}")
    logger.info(f"  Max Depth: {max_depth}")
    logger.info(f"  Number of Trees: {n_estimators}")

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
        "Test": weight_test}



    try:
    
        # all_model ,experiment_results_rows = Run_orchestrator(
        #     dataset_name=dataset_name,
        #     max_depth=max_depth,
        #     overall_n_estimators = n_estimators,
        #     split_points=split_points,
        #     config_index=1,
        #     logger=process_logger ,# Pass the logger instance to the orchestrator
        #     th_combination = config['th'],
        #     tree_splits= config['Tree_splits'],
        #     BO_time = config['BO_time']
            
        # )
        # process_logger.info(f"Finished experiment for Dataset: {dataset_name}\n")
        # return {'dataset_name': dataset_name, 'rows': experiment_results_rows}

            # Create a SINGLE row for this configuration
        combined_config_row = {
            "Config_Index": 1,
            "Overall_Max_Depth": max_depth,
            "Num_Trees_per_RF": n_estimators,
            "Overall_Trees": n_estimators * len(split_points),
            **current_config_training_metrics[0],
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
            inference_obj = RunInference(X_test=X_data, y_test=y_data, models=clf, stages=split_points)

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
        
        combined_config_row["BO_time"] = config['BO_time']

        if logger: logger.info(f"============================================================")
        if logger: logger.info(f"Finished model for Configuration #{1}\n\n")
        if logger: logger.info(f"============================================================ \n\n")

        # Return a list with just ONE row
        # return all_model, [combined_config_row]
        logger.info(f"Finished experiment for Dataset: {dataset_name}\n")
        return {'dataset_name': dataset_name, 'rows': [combined_config_row]}



    except Exception as e:
        logger.error(f"Error processing Dataset: {dataset_name}: {e}", exc_info=True)
        return {'dataset_name': dataset_name, 'rows': [], 'error': str(e)}


def max_entropy(num_classes):
    probabilities = np.ones(num_classes) / num_classes
    return -np.sum(probabilities * np.log(probabilities))





def evaluate(params, n_ex, RF_cache, train_time_cache, validation_acc, validation_energy, train_flatten, y_train, val_flatten, y_val, n_classes, models_dir , dataset_name):
    # features percentages
    
    
    energy_val = 100000.0 # Default to a very high energy (bad)
    acc_val = 0.0 # Default to a very low accuracy (bad)
    experiment_results_rows = None
    
    process_logger = logging.getLogger(f"Process-{os.getpid()}")
    proportions = []
    current = round(params['p_0'],2)
    proportions.append(current)
    for i in range(1,n_ex-1):
        current = current + params[f'p_{i}']
        proportions.append(round(current,2))
    proportions.append(1)

    # tree splits
    t_s = []
    current = round(params['t_0'],2)
    t_s.append(current)
    for i in range(1,n_ex-1):
        current = current + params[f't_{i}']
        t_s.append(round(current,2))
    t_s.append(1)

    # entropy thresholds
    th_combination = []
    for i in range(n_ex-1):
        th_combination.append(round(params[f'th_{i}'],2))

    

    RF_clf_param = (params['n_estimators'],params['max_depth'], n_ex, tuple(proportions), tuple(t_s))
    RF_param_with_th = (params['n_estimators'],params['max_depth'], n_ex, tuple(proportions), tuple(t_s), tuple(th_combination))

    F = f"{params['n_estimators']}_{params['max_depth']}_{n_ex}_" \
    f"{'_'.join(f'{x:.2f}' for x in proportions)}_" \
    f"{'_'.join(f'{x:.2f}' for x in t_s)}.pkl"

    model_file_path = os.path.join(models_dir, F)

    # If we build the RF previously, then we reuse it
    if os.path.isfile(model_file_path):
        with open(model_file_path, 'rb') as f:
            loaded_data = pickle.load(f)
            
        clf = loaded_data.get('model')
        experiment_results_rows = loaded_data.get('results')
            
    
     
    else: 
        t1 = time.time()
        # clf = RandomForest(n_trees=params['n_estimators'], max_depth=params['max_depth'])
        # clf.fit(train_flatten, y_train, proportions, t_s) 
        
        
        clf, experiment_results_rows = Run_orchestrator(
        dataset_name=dataset_name,
        max_depth=params['max_depth'],
        overall_n_estimators = params['n_estimators'],
        split_points=proportions,
        config_index= 1 ,
        logger=process_logger ,
        th_combination = th_combination,
        tree_splits=t_s,
        BO_time= None
        )
        
        energy_val = 100000.0 # Default to a very high energy (bad)
        acc_val = 0.0 # Default to a very low accuracy (bad)
        
        if experiment_results_rows: 
            row_dict = experiment_results_rows[0]
        
            energy_str = row_dict.get('Energy_USED_Val')
            acc_str = row_dict.get('Total_acc_Val')
            
            try:
                if energy_str is not None and energy_str != '':
                    energy_val = float(energy_str)
                else:
                    process_logger.warning(f"Energy_USED_Val was missing or empty for a trial.")
            except ValueError:
                process_logger.error(f"Could not convert Energy_USED_Val '{energy_str}' to float. Setting to infinity.")

            try:
                if acc_str !='nan' and acc_str != '':
                    acc_val = float(acc_str)
                else:
                    process_logger.warning(f"Total_acc_Val was missing or empty for a trial.")
            except ValueError:
                process_logger.error(f"Could not convert Total_acc_Val '{acc_str}' to float. Setting to -infinity.")
        else:
            process_logger.warning("Run_orchestrator returned no results for a trial.") 
        
        t2 = time.time()
        train_time_cache[RF_clf_param] = t2 - t1 
     
        data_to_save = {
            'model': clf,
            'results': experiment_results_rows
        }
        with open(model_file_path, 'wb') as f:
            pickle.dump(data_to_save, f)  
            
       

    # Store the corresponding validation accuracy and energy
    if RF_param_with_th not in validation_acc:
        validation_acc[RF_param_with_th] = acc_val
        validation_energy[RF_param_with_th] = energy_val

    return  {"accuracy": (acc_val, 0.0), "energy": (energy_val, 0.0)} , experiment_results_rows
    




def perform_BO(args):
    dataset_name, num_exits, entropy_min, entropy_max, train_flatten, y_train, val_flatten, y_val, n_classes, models_dir = args
    list_configs = []

    RF_cache = {}
    train_time_cache = {}
    validation_acc = {}
    validation_energy = {}

    # Search space
    if dataset_name == 'Shoaib':
            param = [
                    {"name": "n_estimators", "type": "range", "value_type": "int", "bounds": [60, 80]},
                    {"name": "max_depth", "type": "range", "value_type": "int", "bounds": [50, 70]},
            ]
    if dataset_name == 'Epilepsy':
            param = [
                    {"name": "n_estimators", "type": "range", "value_type": "int", "bounds": [80, 90]},
                    {"name": "max_depth", "type": "range", "value_type": "int", "bounds": [10, 25]},
            ]
    if dataset_name == 'EMGPhysical':
            param = [
                    {"name": "n_estimators", "type": "range", "value_type": "int", "bounds": [30, 40]},
                    {"name": "max_depth", "type": "range", "value_type": "int", "bounds": [25, 40]},
            ]   
    if dataset_name == 'SelfRegulationSCP1':
            param = [
                    {"name": "n_estimators", "type": "range", "value_type": "int", "bounds": [15, 20]},
                    {"name": "max_depth", "type": "range", "value_type": "int", "bounds": [10, 30]},
            ]
    if dataset_name == 'WESADchest':
            param = [
                    {"name": "n_estimators", "type": "range", "value_type": "int", "bounds": [4, 8]},
                    {"name": "max_depth", "type": "range", "value_type": "int", "bounds": [20, 60]},
            ]
    if dataset_name == 'PAMAP2':
            param = [
                    {"name": "n_estimators", "type": "range", "value_type": "int", "bounds": [10, 30]},
                    {"name": "max_depth", "type": "range", "value_type": "int", "bounds": [15, 30]},
            ]

    #wid = 0.9/num_exits

    if num_exits == 2: # the first percentage should be in [0.25,0.5], then we will keep adding percentages, so for the rest of percentages (num_exits-1)* wid < 0.95-0.5 
        wid = 0.5
    elif num_exits == 3:
        wid = 0.25
    else:
        wid = 0.16

    # the first percentage    
    param.append({"name": f"p_{0}", "type": "range", "value_type": "float", "bounds": [0.25, 0.5]}) # features percentages
    param.append({"name": f"t_{0}", "type": "range", "value_type": "float", "bounds": [0.25, 0.5]}) # tree splits 

    # next percentages
    for i in range(1,num_exits - 1):
      param.append({"name": f"p_{i}", "type": "range", "value_type": "float", "bounds": [0.1, wid]}) # features percentages
      param.append({"name": f"t_{i}", "type": "range", "value_type": "float", "bounds": [0.1, wid]}) # tree splits 

    # thresholds
    for i in range(num_exits-1):
        param.append({"name": f"th_{i}", "type": "range", "value_type": "float", "bounds": [entropy_min, entropy_max]})


    BO_t1 = time.time()
    ax_client = AxClient()
    ax_client.create_experiment(
        name="multi_objective_optimization",
        parameters=param,
        objectives={
            "accuracy": ObjectiveProperties(minimize=False),
            "energy": ObjectiveProperties(minimize=True)
        },
        overwrite_existing_experiment=True
    )

    # Bayesian optimization
    NUM_TRIALS = 3
    for _ in range(NUM_TRIALS):
        parameters, trial_index = ax_client.get_next_trial()
        evaluation ,experiment_results_rows= evaluate(parameters, num_exits, RF_cache, train_time_cache, validation_acc, validation_energy, train_flatten, y_train, val_flatten, y_val, n_classes, models_dir,dataset_name)
        ax_client.complete_trial(trial_index=trial_index, raw_data=evaluation)
    BO_t2 = time.time()

    # Pareto optimal points
    pareto_dict = ax_client.get_pareto_optimal_parameters()
    for trial_index, params in pareto_dict.items():
        # For one single trial and parameter configuration
        trial = ax_client.experiment.trials[trial_index]
        (param_dict,_) = params 
        n_estimators = param_dict["n_estimators"]
        max_depth = param_dict["max_depth"]

        p_list = []
        t_list = []

        for i in range(num_exits - 1):
            p_list.append(param_dict[f'p_{i}'])
            t_list.append(param_dict[f't_{i}'])

        th_combinations = []
        for i in range(num_exits-1):
            th_combinations.append(round(param_dict[f'th_{i}'], 2))

        
        current = round(p_list[0], 2)
        proportions = [current]
        for i in range(1, num_exits - 1):
            current = current + p_list[i]
            proportions.append(round(current, 2))
        proportions.append(1)

       
        current = round(t_list[0], 2) 
        t_splits = [current]
        for i in range(1, num_exits - 1):
            current = current + t_list[i]
            t_splits.append(round(current, 2))
        t_splits.append(1)

        RF_clf_param = (param_dict['n_estimators'],param_dict['max_depth'], num_exits, tuple(proportions), tuple(t_splits))
        RF_param_with_th = (param_dict['n_estimators'],param_dict['max_depth'], num_exits, tuple(proportions), tuple(t_splits), tuple(th_combinations))

        F = f"{param_dict['n_estimators']}_{param_dict['max_depth']}_{num_exits}_" \
        f"{'_'.join(f'{x:.2f}' for x in proportions)}_" \
        f"{'_'.join(f'{x:.2f}' for x in t_splits)}.pkl"

        model_file_path = os.path.join(models_dir, F)
        
        with open(model_file_path, 'rb') as f:
            loaded_data = pickle.load(f)
            
        clf = loaded_data.get('model')
        experiment_results_rows = loaded_data.get('results')

        BO_time = BO_t2 -  BO_t1

        # Shut down the executor
        if hasattr(clf,"executor") and clf.executor:
            clf.executor.shutdown(wait=True)
            clf.executor = None

        dictionary = {
            'Dataset': dataset_name,
            'Num_exits': num_exits,
            'Tree_depth': max_depth,
            'n_estimators': n_estimators,
            'Tree_splits': t_splits,
            'Proportions': proportions,
            'th': th_combinations ,#[th_combinations],
            'clf': clf,
            'train_time': train_time_cache[RF_clf_param],
            'validation_acc': validation_acc[RF_param_with_th],
            'validation_energy': validation_energy[RF_param_with_th],
            'BO_time': BO_time,
            'current_config_training_metrics' : experiment_results_rows
        }
    
        list_configs.append(dictionary)
    return list_configs


def generate_all_configurations(dataset_name):
  l =[]
  # exits = [i for i in range(2,10)]
  exits = [i for i in range(2,5)]

  if dataset_name == 'Shoaib':
      n_classes = 7
  if dataset_name == 'Epilepsy':
      n_classes = 4
  if dataset_name == 'EMGPhysical':
      n_classes = 4
  if dataset_name == 'SelfRegulationSCP1':
      n_classes = 2
  if dataset_name == 'WESADchest':
      n_classes = 3
  if dataset_name == 'PAMAP2':
      n_classes = 5

  entropy_max = max_entropy(n_classes)
  num_bins = 4
  entropy_min = entropy_max/num_bins

  file_path = f'Datasets\\Datasets\\{dataset_name}_dataLabels.pkl'
  with open(file_path, 'rb') as file:
     data_dict = pickle.load(file)
  data = data_dict['data']          # data
  labels_array = data_dict['labels'] # labels   

  n_window, n_channel, n_data = data.shape  # data shape
  lst = list(range(0, n_window))

  # First split: 80% (Train + Validation) and 20% (Test)
  X_temp, X_test_ind, l_temp, l_test = train_test_split(lst, labels_array, test_size=0.20, random_state=42)

  # Second split: Split the 80% (Train + Validation) into 60% (Train) and 20% (Validation)
  X_train_ind, X_val_ind, l_train, l_val = train_test_split(X_temp, l_temp, test_size=0.25, random_state=42)  # 0.25 * 80% = 20%

  # data
  X_train = data[X_train_ind,:,:]
  X_val = data[X_val_ind,:,:]
  X_test = data[X_test_ind,:,:]
  # labels
  y_train = labels_array[X_train_ind]
  y_val = labels_array[X_val_ind]
  y_test = labels_array[X_test_ind]

  classes = np.unique(y_train)
  n_classes = len(classes)

  # flatten the data
  input_data_RF_train = copy.deepcopy(X_train)
  train_flatten = input_data_RF_train.reshape(len(y_train), n_channel*n_data)

  input_data_RF_val = copy.deepcopy(X_val)
  val_flatten = input_data_RF_val.reshape(len(y_val), n_channel*n_data)

  input_data_RF_test = copy.deepcopy(X_test)
  test_flatten = input_data_RF_test.reshape(len(y_test), n_channel*n_data)

  # Perform BO
  models_dir = f"{dataset_name}_Models"
  os.makedirs(models_dir, exist_ok=True)
  list_of_parallel_processes = [(dataset_name, num_exits, entropy_min, entropy_max, train_flatten, y_train, val_flatten, y_val, n_classes, models_dir) for num_exits in exits]
  with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
      list_of_configs = list(executor.map(perform_BO, list_of_parallel_processes))

      
  l = [d for sublist in list_of_configs for d in sublist]
                
  return l


# the parallel code
def parallel_code (dataset_name):
    
    all_configs = generate_all_configurations(dataset_name)
    logging.info(f"Total configurations generated: {len(all_configs)}")

    dataset_saver = SaveToCSV(DataSet_Name = dataset_name)
    
    num_configs = len(all_configs)
    BATCH_SIZE = 8
    num_batches = int(np.ceil(num_configs/BATCH_SIZE))
    overall_start_time = time.time()

    for b in range(0, num_batches):
        batch_start = b*BATCH_SIZE
        batch_end = (b+1)*BATCH_SIZE

        if(batch_end > num_configs):
            batch_end = num_configs
       
        batch_results = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count() or 4) as executor:
            futures = {executor.submit(run_and_record_experiment, config): config for config in all_configs[batch_start:batch_end]}
            for future in concurrent.futures.as_completed(futures):
                original_config = futures[future]
                try:
                    result = future.result() # This will now receive the dictionary {'dataset_name': ..., 'rows': ...}
                    if result: # Check if result is not None (in case of unexpected errors)
                        batch_results.append(result)
                    else:
                        logging.warning(f"Received None result for config (Dataset: {original_config.get('dataset_name', 'N/A')}).")
                except Exception as e:
                    logging.error(f"Error retrieving result for config (Dataset: {original_config.get('dataset_name', 'N/A')}): {e}", exc_info=True)

          # Write in batches to reduce I/O overhead
        for result_dict in batch_results: # Renamed 'result' to 'result_dict' for clarity
                if 'rows' in result_dict and result_dict['rows']:
                    for row in result_dict['rows']:
                        dataset_saver.append_completed_config_row(row)
                elif 'error' in result_dict:
                    logging.error(f"Skipping saving for errored config (Dataset: {result_dict['dataset_name']}): {result_dict['error']}")
            
        # After all batches for a dataset are processed, write its collected data to file
        dataset_saver.write_all_combined_metrics_to_file()
        logging.info(f"Completed all configurations for Dataset: {dataset_name}")

    overall_end_time = time.time()
    logging.info(f"All experiments completed across all datasets. Total duration: {overall_end_time - overall_start_time:.2f} seconds.")
        
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='Epilepsy')
    args = parser.parse_args()
    datasetname= args.dataset_name
    start_time = time.time()
    parallel_code(datasetname)
    # parallel_code('Epilepsy')
    
    end_time = time.time()
    total_time = end_time - start_time
    logging.info(f"Total execution time: {total_time:.2f} seconds")
    time_data = "time_data_exhaustive.txt"
    with open(time_data, "a") as tf:
        tf.write(f"{args.dataset_name} takes {total_time:.2f} seconds\n")