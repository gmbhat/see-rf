import concurrent.futures
import itertools
import csv
import torch
import torch.nn.functional as F
from eval_config import run_and_record_experiment
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
from RandomForest import RandomForest
from helper_functions1 import error, accuracy, entropy, Check_exits_acc
from ax.service.ax_client import AxClient
from ax.service.utils.instantiation import ObjectiveProperties
import os
from DataLoad import LoadData





pp = pprint.PrettyPrinter(indent=2)

device = torch.device('cpu')
torch.set_num_threads(1)

random.seed(42)
np.random.seed(42)



def max_entropy(num_classes):
    probabilities = np.ones(num_classes) / num_classes
    return -np.sum(probabilities * np.log(probabilities))



# Evaluate Objective functions
def evaluate(params, n_ex, eval_acc_energy_time, train_time_cache, individual_surg_aqui_step, surg_aquiz_BO_time, validation_acc, validation_energy, train_flatten, y_train, val_flatten, y_val, n_classes, models_dir):
    # features percentages
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

    V_acc_exit = [] 
    V_perc_taken = [] 
    Total_V_accuracy = []

    RF_clf_param = (params['n_estimators'],params['max_depth'], n_ex, tuple(proportions), tuple(t_s))
    RF_param_with_th = (params['n_estimators'],params['max_depth'], n_ex, tuple(proportions), tuple(t_s), tuple(th_combination))

    F = f"{params['n_estimators']}_{params['max_depth']}_{n_ex}_" \
    f"{'_'.join(f'{x:.2f}' for x in proportions)}_" \
    f"{'_'.join(f'{x:.2f}' for x in t_s)}.pkl"

    model_file_path = os.path.join(models_dir, F)

    # If we build the RF previously, then we reuse it
    if os.path.isfile(model_file_path):
        with open(model_file_path, 'rb') as f:
            clf = pickle.load(f)
    #if RF_clf_param in RF_cache:
        #clf = RF_cache[RF_clf_param]

    # Otherwise, we build it    
    else: 
        t1 = time.time()
        clf = RandomForest(n_trees=params['n_estimators'], max_depth=params['max_depth'])
        clf.fit(train_flatten, y_train, proportions, t_s) 
        t2 = time.time()
        #RF_cache[RF_clf_param] = clf 
        # Shut down the executor
        if hasattr(clf,"executor") and clf.executor:
            clf.executor.shutdown(wait=True)
            clf.executor = None
        with open(model_file_path, 'wb') as f:
            pickle.dump(clf, f)  
        train_time_cache[RF_clf_param] = t2 - t1 

    # Turn the executor on
    if hasattr(clf, "executor") and clf.executor is None:
      clf.executor = concurrent.futures.ProcessPoolExecutor(max_workers=5)

    eval_acc_energy_time_1 = time.time()
    E, _ = Check_exits_acc(clf, val_flatten, y_val , n_classes , proportions, th_combination, n_ex, V_acc_exit, V_perc_taken, Total_V_accuracy, ece_per_exit=[], overall_ece=[])
    eval_acc_energy_time_2 = time.time()
    acc = Total_V_accuracy[0]
    energy = 0
    for i in range(len(E)):
        energy = energy + E[i]

    # Store the corresponding validation accuracy and energy
    if RF_param_with_th not in validation_acc:
        validation_acc[RF_param_with_th] = acc
        validation_energy[RF_param_with_th] = energy

    # store eval_acc_energy time
    eval_acc_energy_time[RF_param_with_th] = eval_acc_energy_time_2-eval_acc_energy_time_1
        
    # store BO_surg_aquiz_time
    individual_surg_aqui_step[RF_param_with_th] = surg_aquiz_BO_time

    return  {
    "accuracy": (acc, 0.0),
    "energy": (energy, 0.0)
        }


def perform_BO(args):
    dataset_name, num_exits, entropy_min, entropy_max, train_flatten, y_train, val_flatten, y_val, n_classes, models_dir = args
    list_configs = []
    individual_surg_aqui_step = {}
    eval_acc_energy_time = {}
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
    NUM_TRIALS = 30
    for _ in range(NUM_TRIALS):
        surg_aquiz_BO_t1 = time.time()
        parameters, trial_index = ax_client.get_next_trial()
        surg_aquiz_BO_t2 = time.time()
        evaluation = evaluate(parameters, num_exits, eval_acc_energy_time, train_time_cache, individual_surg_aqui_step, surg_aquiz_BO_t2-surg_aquiz_BO_t1, validation_acc, validation_energy, train_flatten, y_train, val_flatten, y_val, n_classes, models_dir)
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
            clf = pickle.load(f)

        #clf = RF_cache[RF_clf_param]
        BO_time = BO_t2-BO_t1

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
            'th': [th_combinations],
            'clf': clf,
            'train_time': train_time_cache[RF_clf_param],
            'validation_acc': validation_acc[RF_param_with_th],
            'validation_energy': validation_energy[RF_param_with_th],
            'eval_acc_energy_time': eval_acc_energy_time[RF_param_with_th],
            'surogate_aquiz_time': individual_surg_aqui_step[RF_param_with_th],
            'BO_time': BO_time
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

  file_path = f'Datasets/{dataset_name}_dataLabels.pkl'
  with open(file_path, 'rb') as file:
     data_dict = pickle.load(file)

  '''data = data_dict['data']          # data
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
  test_flatten = input_data_RF_test.reshape(len(y_test), n_channel*n_data)'''

  loader = LoadData()
  loader.Read(dataset_name)
  loader.SplitData()

  train_flatten = loader.GetTrainX()
  val_flatten   = loader.GetValX()
  y_train = loader.GetYtrain()
  y_val   = loader.GetYval()

  # Perform BO
  models_dir = f"{dataset_name}_Models"
  os.makedirs(models_dir, exist_ok=True)
  list_of_parallel_processes = [(dataset_name, num_exits, entropy_min, entropy_max, train_flatten, y_train, val_flatten, y_val, n_classes, models_dir) for num_exits in exits]
  with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
      list_of_configs = list(executor.map(perform_BO, list_of_parallel_processes))

      
  l = [d for sublist in list_of_configs for d in sublist]
                
  return l


# The function that will prepare the header
def prepare_header(num_exits):
    header = ['dataset', 'num_exits', 'max_depth', 'n_estimators', 'tree splits', 'data percentages']
    # Adding accuracy for Train, Validation, and Test (at each exit)
    for i in range(num_exits):
        header.append(f'T_acc_{i+1}')

    for i in range(num_exits):
        header.append(f'V_acc_{i+1}')

    for i in range(num_exits):
        header.append(f'Test_acc_{i+1}')

    # Adding exit thresholds (E_TH)
    for i in range(num_exits):
        header.append(f'E_TH{i+1}')

    # Adding accuracy at each exit
    for i in range(num_exits):
        header.append(f'T_acc_exit_{i+1}')
    for i in range(num_exits):
        header.append(f'V_acc_exit_{i+1}')
    for i in range(num_exits):
        header.append(f'Test_acc_exit_{i+1}')

    # Adding percentage taken at each exit
    for i in range(num_exits):
        header.append(f'T_perc_taken_{i+1}')
    for i in range(num_exits):
        header.append(f'V_perc_taken_{i+1}')
    for i in range(num_exits):
        header.append(f'Test_perc_taken_{i+1}')

    # Adding entire data percentages
    for i in range(num_exits):
        header.append(f'entire_data_perc_taken_{i+1}')
    
    header.append("Total_train_accuracy")
    header.append("Total_validation_accuracy")
    header.append("Total_test_accuracy")
    header.append("train_energy")
    header.append("val_energy")
    header.append("test_energy")
    header.append("data_total_energy")
    for i in range(num_exits):
     header.append(f'T_ece_exit_{i+1}')
    for i in range(num_exits):
     header.append(f'V_ece_exit_{i+1}')
    for i in range(num_exits):
     header.append(f'Test_ece_exit_{i+1}')

    header.append("T_model_overall_ece")
    header.append("V_model_overall_ece")
    header.append("Test_model_overall_ece")
    header.append("model_ece_entire_data")
    header.append("total nodes")
    header.append("train_time")
    header.append("inference_time")
    header.append("BO_eval_acc_energy_time")
    header.append("BO_surogate_aquiz_time")
    header.append("BO_time")

    header.append("train_precision")
    header.append("train_recall")
    header.append("train_f1")

    header.append("val_precision")
    header.append("val_recall")
    header.append("val_f1")

    header.append("test_precision")
    header.append("test_recall")
    header.append("test_f1")


    

    return header



# the function that will write the results to the csv file
def write_to_csv(config_results, header, file):  

    writer = csv.writer(file)
    for config_result in config_results:
        row = [config_result[key] for key in header]
        writer.writerow(row)

# the parallel code
def parallel_code (csv_file, datasest_name):

    header = prepare_header(4) 
    with open(csv_file, mode='w', newline='') as f:
      writer = csv.writer(f)
      writer.writerow(header)


      # Generate all configurations upfront
      all_configs = generate_all_configurations(datasest_name)
      logging.info(f"Total configurations generated: {len(all_configs)}")

      # Use ProcessPoolExecutor for CPU-bound tasks

      num_configs = len(all_configs)
      BATCH_SIZE = 8
      num_batches = int(np.ceil(num_configs/BATCH_SIZE))

      for b in range(0, num_batches):
          batch_start = b*BATCH_SIZE
          batch_end = (b+1)*BATCH_SIZE

          if(batch_end > num_configs):
              batch_end = num_configs
          print(time.time(), "Batch ", b, "/", num_batches)

          batch_results = []
          with concurrent.futures.ThreadPoolExecutor(max_workers = 8) as executor: 
              futures = {executor.submit(run_and_record_experiment, config): config for config in all_configs[batch_start:batch_end]} 
              for future in concurrent.futures.as_completed(futures):
                  try:
                      result = future.result()
                      batch_results.append(result)

                  except Exception as e:
                      logging.error(f"Error processing config: {e}")

          # Write in batches to reduce I/O overhead
          for br in batch_results:
            write_to_csv(br, header, f)

          f.flush()


    logging.info("All experiments completed and results written")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='Epilepsy')
    args = parser.parse_args()

    start_time = time.time()
    parallel_code(f"RF_results_{args.dataset_name}.csv", args.dataset_name)
    end_time = time.time()
    total_time = end_time - start_time
    time_data = "time_data.txt" 
    with open(time_data, "a") as tf:
        tf.write(f"{args.dataset_name} takes {total_time:.2f} seconds\n")    
    logging.info(f"Total execution time: {total_time:.2f} seconds")
