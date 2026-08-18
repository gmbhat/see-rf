import csv
import pickle
import argparse
from ReadFile import LoadData
import logging
import time
import numpy as np
from sklearn.metrics import accuracy_score
from Seq_RandomForest import SequentialRandomForest
import pickle


def run_staged_srf(X_train, y_train, X_VAL, y_VAL, config_index,
                   overall_max_depth, overall_n_estimators, stages, tree_splits,dataset_name, random_state=42,  logger=None):
    
    TIME_LOG_FILE = f"Boosting_trainingTime\\training_times_Boosting-{dataset_name}.txt"
    current_config_training_metrics = {}
    n_features_total = X_train.shape[1]
    models_info = []

    if logger: logger.info(f"\n--- Starting Staged sRF Learning for Config {config_index} ---")
    
    #assing number of trees per proportion
    number_of_tress_per_forest = []
    allocated_trees = 0
    for k in range(len(tree_splits)):
        if k == len(tree_splits) - 1:
            num_estimators = overall_n_estimators - allocated_trees
        else:
            num_estimators = max(1 , int(tree_splits[k] * overall_n_estimators)- allocated_trees)
            allocated_trees += num_estimators

        number_of_tress_per_forest.append(num_estimators)

    start_time_total = time.time()
    
    for i, percentage in enumerate(stages):
        stage_num = i + 1
        if logger: logger.info(f"\n--- Stage {stage_num} ({int(percentage*100)}% Initial Features) ---")
        start_time_stage = time.time()
        n_features_to_select = int(np.ceil(n_features_total * percentage))
        n_features_to_select = min(n_features_to_select, n_features_total)
        selected_feature_indices = np.arange(n_features_to_select)
        if logger: logger.info(f"  Using first {n_features_to_select} features (Indices 0 to {n_features_to_select-1}).")
        current_X_train_stage = X_train[:, selected_feature_indices]
        current_X_VAL_stage = X_VAL[:, selected_feature_indices]
        if logger: logger.info(f"  Training data shape for stage: {current_X_train_stage.shape}")
        # create the models
        srf_stage = SequentialRandomForest(
            n_estimators= number_of_tress_per_forest[i],
            max_depth = overall_max_depth,
            random_state = random_state,
            tree_splits = tree_splits
        )
     
        # Train sRF for this stage
        if logger: logger.info(f"  Training sRF with {tree_splits[i]} estimators...")
        srf_stage.fit(current_X_train_stage, y_train)
        if logger: logger.info(f" *** number of trees: {len(srf_stage.trees_)}")

        # Evaluate
        y_pred_stage  = srf_stage.predict(current_X_VAL_stage)
        acc_stage = accuracy_score(y_VAL, y_pred_stage)
        if logger: logger.info(f"  Stage {stage_num} Test Accuracy: {acc_stage:.4f}")
        # print(f"  Stage {stage_num} Test Accuracy: {acc_stage:.4f}")

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

    return models_info, current_config_training_metrics  
    

def parse_args():
    parser = argparse.ArgumentParser(description = "Training an RF and Testing it")
    parser.add_argument("--dataset_name", type=str, default= "Epilepsy", help = "The Dataset name")
    parser.add_argument("--n_est", type=int,default=50, help = "The number of estimators")
    parser.add_argument("--max_depth", type=int, default=100, help = "The max depth")
    parser.add_argument("--num_exits", type=int,  default= 3 ,help = "The number of exits")
    parser.add_argument("--tree_splits", type=list, default=[0.3, 0.7 , 0.9] ,help = "Tree splits")
    parser.add_argument("--proportions", type=list, default=[0.25, 0.5 , 0.75] ,help = "Data proportions")
    return parser.parse_args()



if __name__ == "__main__":

    args = parse_args()
    classData = LoadData()
    classData.Read(args.dataset_name)
    classData.SplitData()

    X_test = classData.GetTestX()
    y_train=classData.GetYtrain()
    y_test = classData.GetYtest()
    nb_clss = len(np.unique(y_train))
    
    all_model, current_config_training_metrics = run_staged_srf(
        classData.GetTrainX(), classData.GetYtrain(), classData.GetValX(), classData.GetYval(),
        config_index= 1,
        overall_max_depth=args.max_depth,
        overall_n_estimators= args.n_est,
        stages=args.proportions,
        tree_splits = args.tree_splits,
        dataset_name = args.dataset_name,
        random_state=42,
        logger=logging
    )
    
    dataset_name = args.dataset_name 
    with open(f"PKL_Saved_Files\\{dataset_name}_trained_model.pkl", "wb") as f:
        pickle.dump(all_model, f)
    with open(f"PKL_Saved_Files\\{dataset_name}_trained_results.pkl", "wb") as f:
        pickle.dump(current_config_training_metrics, f)
        
    # Save the X_train and y_train and nb_classes
    np.save(f"PKL_Saved_Files\\{dataset_name}_test_data.npy", X_test)
    np.save(f"PKL_Saved_Files\\{dataset_name}_test_labels.npy", y_test) 
    np.save(f"PKL_Saved_Files\\{dataset_name}_nb_classes.npy", np.array([nb_clss]))
 
print(f"Model saved to trained_model_{dataset_name}.pkl")