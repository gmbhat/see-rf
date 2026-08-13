import concurrent.futures
import itertools
import csv
import torch
import torch.nn.functional as F
import random
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
import copy
import pickle
from sklearn.model_selection import train_test_split
from RandomForest import RandomForest
from helper_functions import error, accuracy, entropy, Check_exits_acc
import random




pp = pprint.PrettyPrinter(indent=2)

device = torch.device('cpu')
torch.set_num_threads(1)



def max_entropy(num_classes):
    probabilities = np.ones(num_classes) / num_classes
    return -np.sum(probabilities * np.log(probabilities))



def func_threshold_combinations(num_exits, dataset_max_entropy):
    num_bins = 4
    bin_width = dataset_max_entropy / num_bins
    thresholds = [(i + 1) * bin_width for i in range(num_bins)]  # e.g., [0.5, 1.0, 1.5, 2.0]

    # Generate permutations 
    permutations = [list(p) for p in itertools.permutations(thresholds, num_exits-1)]

    # Shuffle to randomize selection
    random.shuffle(permutations)

    #return permutations[:10]
    return permutations




def generate_percentages(num_exits): # this will generate multiple tree splits or data percentages
    #wid = 0.9 / num_exits
    if num_exits == 2:  
        wid = 0.5
    elif num_exits == 3:
        wid = 0.25
    else:
        wid = 0.16
    list_of_percentages = []

    for _ in range(10): 
        per_list = []
        current = 0
        #current += 0.5 # first exit
        current += random.uniform(0.3,0.5) # first exit
        per_list.append(round(current, 2))
        for _ in range(num_exits - 2):
            current += random.uniform(0.1, wid)
            per_list.append(round(current, 2))
        per_list.append(1)
        list_of_percentages.append(tuple(per_list))  # convert to tuple for hashing

    unique_set = set(list_of_percentages)
    return [list(p) for p in unique_set]  # convert tuples back to lists


# ********************************************************************************************* Use this function
def generate_all_configurations(dataset_name):
    random.seed(42)
    exits = [2,3,4] 
    l=[]
    #my_list = [(80,20),(90,30),(80,40),(50,60),(80,50),(100,100),(90,90)]
    if dataset_name == 'Shoaib':
        n_classes = 7
       # my_list = [(d,e) for d in range(50,91,10) for e in [60,75,80]]
        my_list = [(50,60), (60,60), (70,75), (50,80)]
    if dataset_name == 'Epilepsy':
        n_classes = 4
        #my_list = [(d,e) for d in range(10,31,10) for e in range(80,95,5)] 
        my_list = [(10,85), (15,90), (20,80), (25,80)] 
    if dataset_name == 'EMGPhysical':
        n_classes = 4
        #my_list = [(d,e) for d in [25, 35, 40, 50] for e in range(10,50,10)]
        my_list = [(25,40), (35,40), (40,30), (35,30)]
    if dataset_name == 'SelfRegulationSCP1':
        n_classes = 2
        #my_list = [(d,e) for d in range(10,41,10) for e in range(10,21,5)]
        my_list = [(10,20), (20,20), (30,15), (20,15)]
    if dataset_name == 'WESADchest':
        n_classes = 3
        #my_list = [(d,e) for d in [20,40,60,90] for e in range(2,10,2)]
        my_list = [(20,8), (40,6), (60,4)]
    if dataset_name == 'PAMAP2':
        n_classes = 5
        #my_list = [(d,e) for d in [15,20,30,40] for e in range(10,40,10)]
        my_list = [(15,30), (20,20), (30,10), (30,20)]
    
    for (depth, n_est) in my_list:
        for num_exits in exits:
            entropy_max = max_entropy(n_classes)
            th_combinations_list = func_threshold_combinations(num_exits, entropy_max)
            #th_combinations = [th_combinations_list[0], th_combinations_list[1]]
            #th_combinations = random.sample(th_combinations_list,2) # extract randomly to ensure that the total set of final configs has ascending, descending, and random order
            th_combinations = th_combinations_list

            list_tree_splits_l = generate_percentages(num_exits)
            #list_tree_splits = [list_tree_splits_l[0], list_tree_splits_l[1]]
            list_tree_splits = random.sample(list_tree_splits_l,2)

            list_data_percentages_l = generate_percentages(num_exits)
            #list_data_percentages = [list_data_percentages_l[0], list_data_percentages_l[1]]
            list_data_percentages = random.sample(list_data_percentages_l,4)

            for tree_splits in list_tree_splits:
                for proportions in list_data_percentages:
                    dictionary = {'Dataset': dataset_name, 'Num_exits': num_exits, 'Tree_depth': depth, 'Tree_splits': tree_splits, 'Proportions': proportions, 'th': th_combinations, 'n_estimators': n_est}
                    l.append(dictionary)

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

    header = prepare_header(4) #******************************##########################################################################################################
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
          with concurrent.futures.ProcessPoolExecutor(max_workers = 8) as executor: 
              futures = {executor.submit(run_and_record_experiment, config): config for config in all_configs[batch_start:batch_end]} 
              for future in concurrent.futures.as_completed(futures):
                  try:
                      result = future.result()
                      batch_results.append(result)

                  except Exception as e:
                      #logging.error(f"Error processing config: {e}, {all_configs[batch_start:batch_end]}")
                      config = futures[future]  # get the exact config that failed
                      logging.error(f"Error processing config: {e}, config: {config}")

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
    time_data = "time_data_exhaustive.txt" 
    with open(time_data, "a") as tf:
        tf.write(f"{args.dataset_name} takes {total_time:.2f} seconds\n")   
    logging.info(f"Total execution time: {total_time:.2f} seconds")
