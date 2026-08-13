import pandas as pd
import numpy as np
import argparse
import sys
import copy
import pickle
from sklearn.model_selection import train_test_split
import torch
import torch.nn.functional as F
import math
import time


def error(y_true, y_pred): # will return a list that indicates whether each data was well classified or not
    err=(y_true == y_pred).astype(int).tolist()
    return np.array(err)

def accuracy(y_true, y_pred):
    if len(y_true)>0:
      accuracy = np.sum(y_true == y_pred) / len(y_true)
    else:
      accuracy = np.nan
    return accuracy



def entropy(probabilities):   # probabilities is of shape (n_windows, n_classes)
    epsilon = 1e-5  # to avoid taking the logarithm of zero
    return -np.sum(probabilities * np.log(probabilities + epsilon), axis=1)



def max_entropy(num_classes):
    probabilities = np.ones(num_classes) / num_classes
    return -np.sum(probabilities * np.log(probabilities))


def calculate_confidence_entropy_based(entropies, max_entropy):
    return np.clip(1 - (entropies / max_entropy), 0, 1).tolist()

def calculate_confidence_max_prob(probabilities):
    probabilities = np.array(probabilities)  # convert list of lists to ndarray
    return np.max(probabilities, axis=1).tolist()




def ECE_computation(confidences, corrects, num_bins=10):
    # Divide the confidence interval [0,1] into equal-width bins
    bin_bounds = np.linspace(0, 1, num_bins + 1)  # list of bin edges
    ECE = 0.0

    # Loop through each bin
    for i in range(num_bins):
        b_0 = bin_bounds[i]
        b_1 = bin_bounds[i + 1]

        # Select indices of samples whose confidence falls into the bin
        samples_indices = [j for j, c in enumerate(confidences) if b_0 <= c < b_1]

        if len(samples_indices) == 0:
            continue  # skip empty bins

        acc_mean = np.mean([corrects[j] for j in samples_indices])
        conf_mean = np.mean([confidences[j] for j in samples_indices])

        weight = len(samples_indices) / len(confidences)
        ECE += weight * abs(acc_mean - conf_mean)

    return round(ECE, 2)




def Check_exits_acc(clf, data, y, nb_clss, proportions, th_combination, num_exits, acc_exit, perc_taken, Total_accuracy, ece_per_exit, overall_ece): # add ece_per_exit(list), and overall_ece
    """ this function checks wether an exit can be taken or not based on an entropy threshold """

    all_confidences = []
    all_corrects = []
    maximal_entropy = max_entropy(nb_clss)
  

    avg1 = 0
    threshold = round(th_combination[0],2)
    num_samples = int(data.shape[1] * proportions[0])
    subset = data[:,:num_samples]
    flatten_1 = data
    y_subset_1 = y
    exit_level = 1
    start_nodes = None
    t1 = time.time()
    predictions, exit_nodes, prob_1 = clf.predict(subset, nb_clss, exit_level, start_nodes=None)

    energy = []

    # check entropy
    entropy_1 = entropy(prob_1) # a list of entropies numpy.ndarray

    # check if one of the inputs were classifed wrongly (in terms of confidence)
    indices = np.argwhere(entropy_1 > threshold).flatten()


    good_confidence_indices = np.argwhere(entropy_1 <= threshold).flatten() # indices of the data that exited
    acc = accuracy(y_subset_1[good_confidence_indices], predictions[good_confidence_indices]) # accuracy at first exit
    acc_exit.append(float(acc)) # append the accuracy for first exit
    perc_taken.append(len(good_confidence_indices)/len(subset)) # append the percentage of exited data for first exit
    avg1 += (len(good_confidence_indices)/len(subset)) * (0.0 if math.isnan(acc) else float(acc))
    energy.append(perc_taken[-1]*(proportions[0]))

    
    if len(good_confidence_indices)>0: # if we have samples that exited
      corrects = [int(predictions[cor] == y_subset_1[cor]) for cor in good_confidence_indices] # a list
      #confidences = calculate_confidence_entropy_based(entropy_1[good_confidence_indices],maximal_entropy) # a list
      confidences = calculate_confidence_max_prob(prob_1[good_confidence_indices]) # a list
      ece = ECE_computation(confidences, corrects, 10)
      ece_per_exit.append(ece)
      all_confidences += confidences
      all_corrects += corrects
      # model_ece += perc_taken[-1] * ece_per_exit[-1]
    else:
       ece_per_exit.append(-1)

    
    



    # if one of the inputs were classifed wrongly then we need to add more features to the wrongly classified samples, and start another prediction process starting from the previous exit nodes
    i=1 # proportions counter


    while (len(indices) > 0 and i+1 < num_exits):
      # process the rest of the data using the next level
      exit_level = i+1  # now we will exit using the next level
      start_nodes_2 = exit_nodes[indices] # the starting nodes
      num_samples = int(data.shape[1] * proportions[i])
      flatten_2 = flatten_1[indices]
      subset_2 = flatten_1[indices,:num_samples]
      y_subset_2 = y_subset_1[indices]
      predictions_2, exit_nodes_2, prob_2 = clf.predict(subset_2, nb_clss, exit_level, start_nodes=start_nodes_2)

      # check entropy
      entropy_2 = entropy(prob_2) # a list of entropies

      # check if one of the inputs were classifed wrongly (in terms of confidence)
      indices_2 = np.argwhere(entropy_2 > round(th_combination[i],2)).flatten()


      good_confidence_indices = np.argwhere(entropy_2 <= round(th_combination[i],2)).flatten()
      acc = accuracy(y_subset_2[good_confidence_indices], predictions_2[good_confidence_indices])
      acc_exit.append(float(acc))
      perc_taken.append(len(good_confidence_indices)/len(subset))
      avg1 += (len(good_confidence_indices)/len(subset)) * (0.0 if math.isnan(acc) else float(acc))
      energy.append(perc_taken[-1]*(proportions[i]))

      if len(good_confidence_indices)>0: # if we have samples that exited
        corrects = [int(predictions_2[cor] == y_subset_2[cor]) for cor in good_confidence_indices] # a list
        #confidences = calculate_confidence_entropy_based(entropy_2[good_confidence_indices],maximal_entropy) # a list
        confidences = calculate_confidence_max_prob(prob_2[good_confidence_indices]) # a list
        ece = ECE_computation(confidences, corrects, 10)
        ece_per_exit.append(ece)
        all_confidences += confidences
        all_corrects += corrects
        # model_ece += perc_taken[-1] * ece_per_exit[-1]
      else:
        ece_per_exit.append(-1)

      indices = indices_2
      exit_nodes = exit_nodes_2
      flatten_1 = flatten_2
      y_subset_1 = y_subset_2
      i += 1 # move to next proportion


    if len(indices) > 0 and i+1 == num_exits: # we had to go through all the levels up to the last exit because some of the data is still classified wrongly
      # process the rest of the data using the next level
      exit_level = i+1  # now we will exit using the next level
      start_nodes_2 = exit_nodes[indices] # the starting nodes
      num_samples = int(data.shape[1] * proportions[i])
      flatten_2 = flatten_1[indices]
      subset_2 = flatten_1[indices,:num_samples]
      y_subset_2 = y_subset_1[indices]

      predictions_2, exit_nodes_2, prob_2 = clf.predict(subset_2, nb_clss, exit_level, start_nodes=start_nodes_2)
      


      # check entropy
      entropy_2 = entropy(prob_2) # a list of entropies
      # check accuracy
      acc = accuracy(y_subset_2, predictions_2)
      acc_exit.append(float(acc))
      perc_taken.append(len(predictions_2)/len(subset))

      avg1 += (len(predictions_2)/len(subset)) * (0.0 if math.isnan(acc) else float(acc))
      energy.append(perc_taken[-1]*(proportions[i]))

      
      corrects = [int(predictions_2[cor] == y_subset_2[cor]) for cor in range(len(predictions_2))] # a list
      #confidences = calculate_confidence_entropy_based(entropy_2,maximal_entropy) # a list
      confidences = calculate_confidence_max_prob(prob_2) # a list
      ece = ECE_computation(confidences, corrects, 10)
      ece_per_exit.append(ece)
      all_confidences += confidences
      all_corrects += corrects
      # model_ece += perc_taken[-1] * ece_per_exit[-1]

    t2 = time.time()
    Total_accuracy.append(avg1)  
    model_ece = ECE_computation(all_confidences, all_corrects)
    overall_ece.append(model_ece)

    

    return energy, t2-t1
