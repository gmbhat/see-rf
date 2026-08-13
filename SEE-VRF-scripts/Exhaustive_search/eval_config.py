import pandas as pd
import numpy as np
import argparse
import sys
import copy
import pickle
from sklearn.model_selection import train_test_split
from RandomForest import RandomForest
from DataLoad import LoadData
import torch
import torch.nn.functional as F
import math
from helper_functions import error, accuracy, entropy, Check_exits_acc
import random
import time



random.seed(42)
np.random.seed(42)



def run_and_record_experiment(config): # config: dictionnary with one configuration -> returns a dictionnary (that needs to be appended later on
    
    print('we are in config_eval')
    # the configuration parameters
    dataset_name = config['Dataset']
    num_exits = config['Num_exits']
    max_depth = config['Tree_depth']
    n_estimators = config['n_estimators']
    tree_splits = config['Tree_splits']
    proportions = config['Proportions']
    all_config_th = config['th'] # list of lists
    


    file_path = f'Datasets/{dataset_name}_dataLabels.pkl'
    with open(file_path, 'rb') as file:
      data_dict = pickle.load(file)

    loader = LoadData()
    loader.Read(dataset_name)
    loader.SplitData()

    train_flatten = loader.GetTrainX()
    val_flatten   = loader.GetValX()
    test_flatten  = loader.GetTestX()

    y_train = loader.GetYtrain()
    y_val   = loader.GetYval()
    y_test  = loader.GetYtest()

    X_train_len = len(y_train)
    X_val_len   = len(y_val)
    X_test_len  = len(y_test)
    total_num_data = X_train_len + X_val_len + X_test_len

    classes = np.unique(y_train)
    n_classes = len(classes)



    

    

    dataset=[]
    num_exits_col = []
    depth_col = []
    tree_splits_col = [] # will contain a list
    prop = [] # will contain a list

    T_acc = []
    V_acc = []
    Test_acc = []

    dataset.append(dataset_name)
    num_exits_col.append(num_exits)
    depth_col.append(max_depth)
    tree_splits_col.append(tree_splits)
    prop.append(proportions)




    # **************** Train the RF ****************
    train_t1 = time.time()
    clf = RandomForest(n_trees=n_estimators, max_depth=max_depth)
    clf.fit(train_flatten, y_train, proportions, tree_splits)
    train_t2 = time.time()

    # **************** Check train accuracy ****************
    # check exits accuracies for training dataset (by forcing the whole training dataset at each exit level)
    # first exit
    num_samples_train = int(train_flatten.shape[1] * proportions[0])
    train_subset = train_flatten[:,:num_samples_train]
    exit_level = 1
    start_nodes = None
    predictions, exit_nodes, prob = clf.predict(train_subset, n_classes, exit_level, start_nodes=None)
    Train_exit_acc = float(accuracy(y_train, predictions))
    T_acc.append(Train_exit_acc)

    # next exits
    for i in range(1,num_exits):
      num_samples_train = int(train_flatten.shape[1] * proportions[i])
      train_subset = train_flatten[:,:num_samples_train]
      exit_level = i+1
      predictions, exit_nodes, prob = clf.predict(train_subset, n_classes, exit_level, start_nodes=exit_nodes)

      Train_exit_acc = float(accuracy(y_train, predictions))
      T_acc.append(Train_exit_acc)

    # now T_acc is ready

    # **************** Check validation accuracy ****************
    # check exits accuracies for validation dataset (by forcing the whole validation dataset at each exit level)
    # first exit
    num_samples_val = int(val_flatten.shape[1] * proportions[0])
    val_subset = val_flatten[:,:num_samples_val]
    exit_level = 1
    start_nodes = None
    predictions, exit_nodes, prob = clf.predict(val_subset, n_classes, exit_level, start_nodes=None)

    Val_exit_acc = float(accuracy(y_val, predictions))
    V_acc.append(Val_exit_acc)

    for i in range(1,num_exits):
      num_samples_val = int(val_flatten.shape[1] * proportions[i])
      val_subset = val_flatten[:,:num_samples_val]
      exit_level = i+1
      predictions, exit_nodes, prob = clf.predict(val_subset, n_classes, exit_level, start_nodes=exit_nodes)

      Val_exit_acc = float(accuracy(y_val, predictions))
      V_acc.append(Val_exit_acc)

    # now V_acc is ready
    #print("validation accuracies: ", V_acc)

    # **************** Check test accuracy ****************
    # check exits accuracies for testing dataset (by forcing the whole testing dataset at each exit level)
    #print("check exits accuracies for testing dataset by forcing the whole testing dataset at each exit level")

    # first exit
    num_samples_test = int(test_flatten.shape[1] * proportions[0])
    test_subset = test_flatten[:,:num_samples_test]
    exit_level = 1
    start_nodes = None
    predictions, exit_nodes, prob = clf.predict(test_subset, n_classes, exit_level, start_nodes=None)

    Test_exit_acc = float(accuracy(y_test, predictions))
    Test_acc.append(Test_exit_acc)

    # next exits
    for i in range(1,num_exits):
      num_samples_test = int(test_flatten.shape[1] * proportions[i])
      test_subset = test_flatten[:,:num_samples_test]
      exit_level = i+1
      predictions, exit_nodes, prob = clf.predict(test_subset, n_classes, exit_level, start_nodes=exit_nodes)

      Test_exit_acc = float(accuracy(y_test, predictions))
      Test_acc.append(Test_exit_acc)

    # now Test_acc is ready
    #print("Test accuracies: ", Test_acc)

    while len(T_acc)<4:
      T_acc.append(-1)

    while len(V_acc)<4:
      V_acc.append(-1)

    while len(Test_acc)<4:
      Test_acc.append(-1)

    #print('Let us try to exit these levels whenever we are confident enough!')
    # evaluate train dataset with exits ; check how many data exited through every each exit and calculate the exits accuracies and the averaged accuracy
    # print('Let us do that for the training dataset')


    list_of_rows = []
    for thh in range(len(all_config_th)):

      T_acc_exit = []
      V_acc_exit = []
      Test_acc_exit = []

      T_perc_taken = []
      V_perc_taken = []
      Test_perc_taken = []

      Total_train_accuracy = []
      Total_validation_accuracy = []
      Total_test_accuracy = []

      entire_data_perc_taken = []

      th_combination = all_config_th[thh]
      # ****************************************************************
      T_ece_per_exit = []
      T_overall_ece = []

      V_ece_per_exit = []
      V_overall_ece = []

      Test_ece_per_exit = []
      Test_overall_ece = []

      #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
      train_precision = []
      train_recall = []
      train_F_score = []

      V_precision = []
      V_recall = []
      V_F_score = []

      T_precision = []
      T_recall = []
      T_F_score = []


      # ****************************************************************

      # Train exits accuracies
      E_train, _ = Check_exits_acc(clf, train_flatten, y_train , n_classes , proportions, th_combination, num_exits, T_acc_exit, T_perc_taken, Total_train_accuracy, T_ece_per_exit, T_overall_ece, train_precision, train_recall, train_F_score)

      # Validation exits accuracies
      E_val, _ = Check_exits_acc(clf, val_flatten, y_val , n_classes , proportions, th_combination, num_exits, V_acc_exit, V_perc_taken, Total_validation_accuracy, V_ece_per_exit, V_overall_ece, V_precision, V_recall, V_F_score)

      # Test exits accuracies
      E_test, test_time = Check_exits_acc(clf, test_flatten, y_test , n_classes , proportions, th_combination, num_exits, Test_acc_exit, Test_perc_taken, Total_test_accuracy, Test_ece_per_exit, Test_overall_ece, T_precision, T_recall, T_F_score)

      # calculate the total number of exited data for each exit
      len_max = num_exits

      check = False
      while (check==False):
        if len(T_perc_taken) < len_max:
          T_perc_taken.append(0)

        if len(V_perc_taken) < len_max:
          V_perc_taken.append(0)

        if len(Test_perc_taken) < len_max:
          Test_perc_taken.append(0)

        if (len(Test_perc_taken) == len(T_perc_taken)) and  (len(Test_perc_taken) == len(V_perc_taken)) and (len(V_perc_taken) == len_max):
          check = True


      for i in range(len_max):
        r = (X_train_len / total_num_data) * T_perc_taken[i] + (X_val_len / total_num_data) * V_perc_taken[i] + (X_test_len / total_num_data) * Test_perc_taken[i]
        entire_data_perc_taken.append(r)


      th_col = [round(x,2) for x in th_combination]
      while len(th_col) < 4:
        th_col.append(-1)

      while len(T_acc_exit)<4:
        T_acc_exit.append(-1)

      while len(V_acc_exit)<4:
        V_acc_exit.append(-1)

      while len(Test_acc_exit)<4:
        Test_acc_exit.append(-1)

      while len(T_perc_taken)<4:
        T_perc_taken.append(-1)

      while len(V_perc_taken)<4:
        V_perc_taken.append(-1)

      while len(Test_perc_taken)<4:
        Test_perc_taken.append(-1)

      while len(entire_data_perc_taken)<4:
        entire_data_perc_taken.append(-1)


      while len(T_ece_per_exit)<4:
        T_ece_per_exit.append(-1)

      while len(V_ece_per_exit)<4:
        V_ece_per_exit.append(-1)

      while len(Test_ece_per_exit)<4:
        Test_ece_per_exit.append(-1)

      total_nodes = clf.get_total_nodes()

      # energy
      e_train = 0
      for i in range(len(E_train)):
        e_train= e_train + E_train[i]

      e_val = 0
      for i in range(len(E_val)):
        e_val= e_val + E_val[i]

      e_test = 0
      for i in range(len(E_test)):
        e_test= e_test + E_test[i]

      data_total_energy = (X_train_len / total_num_data) * e_train + (X_val_len / total_num_data) * e_val + (X_test_len / total_num_data) * e_test
      entire_data_ece = (X_train_len / total_num_data) * T_overall_ece[0] + (X_val_len / total_num_data) * V_overall_ece[0] + (X_test_len / total_num_data) * Test_overall_ece[0]


      # dictionary = {"dataset": dataset[0], "num_exits": num_exits_col[0], "max_depth" : depth_col[0], 'n_estimators': n_estimators, "tree splits": tree_splits_col[0], "data percentages": prop[0],
      # "T_acc_1": T_acc[0], "T_acc_2": T_acc[1], "T_acc_3": T_acc[2], "T_acc_4": T_acc[3],
      #         "V_acc_1": V_acc[0], "V_acc_2": V_acc[1], "V_acc_3": V_acc[2], "V_acc_4": V_acc[3],
      #         "Test_acc_1": Test_acc[0], "Test_acc_2": Test_acc[1], "Test_acc_3": Test_acc[2], "Test_acc_4": Test_acc[3],
      #         "E_TH1": th_col[0], "E_TH2": th_col[1], "E_TH3": th_col[2], "E_TH4": th_col[3],
      #         "T_acc_exit_1": T_acc_exit[0], "T_acc_exit_2": T_acc_exit[1], "T_acc_exit_3": T_acc_exit[2], "T_acc_exit_4": T_acc_exit[3],
      # "V_acc_exit_1": V_acc_exit[0], "V_acc_exit_2": V_acc_exit[1], "V_acc_exit_3": V_acc_exit[2], "V_acc_exit_4": V_acc_exit[3],
      # "Test_acc_exit_1": Test_acc_exit[0], "Test_acc_exit_2": Test_acc_exit[1], "Test_acc_exit_3": Test_acc_exit[2], "Test_acc_exit_4": Test_acc_exit[3],
      # "T_perc_taken_1": T_perc_taken[0], "T_perc_taken_2": T_perc_taken[1], "T_perc_taken_3": T_perc_taken[2], "T_perc_taken_4": T_perc_taken[3],
      # "V_perc_taken_1": V_perc_taken[0], "V_perc_taken_2": V_perc_taken[1], "V_perc_taken_3": V_perc_taken[2], "V_perc_taken_4": V_perc_taken[3],
      # "Test_perc_taken_1": Test_perc_taken[0], "Test_perc_taken_2": Test_perc_taken[1], "Test_perc_taken_3": Test_perc_taken[2], "Test_perc_taken_4": Test_perc_taken[3],
      # "entire_data_perc_taken_1": entire_data_perc_taken[0], "entire_data_perc_taken_2": entire_data_perc_taken[1],
      # "entire_data_perc_taken_3": entire_data_perc_taken[2], "entire_data_perc_taken_4": entire_data_perc_taken[3],
      # "Total_train_accuracy": Total_train_accuracy[0], "Total_validation_accuracy": Total_validation_accuracy[0], "Total_test_accuracy": Total_test_accuracy[0], "train_energy": e_train, "val_energy": e_val, "test_energy": e_test, "data_total_energy": data_total_energy, 
      #  "T_ece_exit_1": T_ece_per_exit[0], "T_ece_exit_2": T_ece_per_exit[1], "T_ece_exit_3": T_ece_per_exit[2], "T_ece_exit_4": T_ece_per_exit[3],
      # "V_ece_exit_1": V_ece_per_exit[0], "V_ece_exit_2": V_ece_per_exit[1], "V_ece_exit_3": V_ece_per_exit[2], "V_ece_exit_4": V_ece_per_exit[3],
      # "Test_ece_exit_1": Test_ece_per_exit[0], "Test_ece_exit_2": Test_ece_per_exit[1], "Test_ece_exit_3": Test_ece_per_exit[2], "Test_ece_exit_4": Test_ece_per_exit[3],
      # "T_model_overall_ece": T_overall_ece[0], "V_model_overall_ece": V_overall_ece[0], "Test_model_overall_ece": Test_overall_ece[0], "model_ece_entire_data": entire_data_ece,"total nodes": total_nodes, "train_time": train_t2 - train_t1, "inference_time": test_time}

      dictionary = {"dataset": dataset[0], "num_exits": num_exits_col[0], "max_depth" : depth_col[0], 'n_estimators': n_estimators, "tree splits": tree_splits_col[0], "data percentages": prop[0],
      "T_acc_1": T_acc[0], "T_acc_2": T_acc[1], "T_acc_3": T_acc[2], "T_acc_4": T_acc[3],
              "V_acc_1": V_acc[0], "V_acc_2": V_acc[1], "V_acc_3": V_acc[2], "V_acc_4": V_acc[3],
              "Test_acc_1": Test_acc[0], "Test_acc_2": Test_acc[1], "Test_acc_3": Test_acc[2], "Test_acc_4": Test_acc[3],
              "E_TH1": th_col[0], "E_TH2": th_col[1], "E_TH3": th_col[2], "E_TH4": th_col[3],
              "T_acc_exit_1": T_acc_exit[0], "T_acc_exit_2": T_acc_exit[1], "T_acc_exit_3": T_acc_exit[2], "T_acc_exit_4": T_acc_exit[3],
      "V_acc_exit_1": V_acc_exit[0], "V_acc_exit_2": V_acc_exit[1], "V_acc_exit_3": V_acc_exit[2], "V_acc_exit_4": V_acc_exit[3],
      "Test_acc_exit_1": Test_acc_exit[0], "Test_acc_exit_2": Test_acc_exit[1], "Test_acc_exit_3": Test_acc_exit[2], "Test_acc_exit_4": Test_acc_exit[3],
      "T_perc_taken_1": T_perc_taken[0], "T_perc_taken_2": T_perc_taken[1], "T_perc_taken_3": T_perc_taken[2], "T_perc_taken_4": T_perc_taken[3],
      "V_perc_taken_1": V_perc_taken[0], "V_perc_taken_2": V_perc_taken[1], "V_perc_taken_3": V_perc_taken[2], "V_perc_taken_4": V_perc_taken[3],
      "Test_perc_taken_1": Test_perc_taken[0], "Test_perc_taken_2": Test_perc_taken[1], "Test_perc_taken_3": Test_perc_taken[2], "Test_perc_taken_4": Test_perc_taken[3],
      "entire_data_perc_taken_1": entire_data_perc_taken[0], "entire_data_perc_taken_2": entire_data_perc_taken[1],
      "entire_data_perc_taken_3": entire_data_perc_taken[2], "entire_data_perc_taken_4": entire_data_perc_taken[3],
      "Total_train_accuracy": Total_train_accuracy[0], "Total_validation_accuracy": Total_validation_accuracy[0], "Total_test_accuracy": Total_test_accuracy[0], "train_energy": e_train, "val_energy": e_val, "test_energy": e_test, "data_total_energy": data_total_energy, 
       "T_ece_exit_1": T_ece_per_exit[0], "T_ece_exit_2": T_ece_per_exit[1], "T_ece_exit_3": T_ece_per_exit[2], "T_ece_exit_4": T_ece_per_exit[3],
      "V_ece_exit_1": V_ece_per_exit[0], "V_ece_exit_2": V_ece_per_exit[1], "V_ece_exit_3": V_ece_per_exit[2], "V_ece_exit_4": V_ece_per_exit[3],
      "Test_ece_exit_1": Test_ece_per_exit[0], "Test_ece_exit_2": Test_ece_per_exit[1], "Test_ece_exit_3": Test_ece_per_exit[2], "Test_ece_exit_4": Test_ece_per_exit[3],
      "T_model_overall_ece": T_overall_ece[0], "V_model_overall_ece": V_overall_ece[0], "Test_model_overall_ece": Test_overall_ece[0], "model_ece_entire_data": entire_data_ece,"total nodes": total_nodes, "train_time": train_t2 - train_t1, "inference_time": test_time,
      "train_precision": train_precision[0], "train_recall": train_recall[0], "train_f1": train_F_score[0], "val_precision": V_precision[0], "val_recall": V_recall[0], "val_f1": V_F_score[0], "test_precision": T_precision[0], "test_recall": T_recall[0], "test_f1": T_F_score[0]}

      list_of_rows.append(dictionary)


    return list_of_rows