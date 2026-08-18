import random
import pandas as pd
import numpy as np
import itertools


class Config:
    
    def __init__(self):
        self.depths =  [i for i in range(50,100,10)]
        self.treenum_options = [i for i in range(10,101,10)]  
        self.RESTRICTED_POINTS = []
    
    
    def generate_random_split(self, min_pct=0.25,min_distance=0.1,restricted_zone=(0.2, 0.3),max_retries=1000):
        
        self.RESTRICTED_POINTS = []
        max_m = int(1 / min_pct)
        
        for _ in range(max_retries):
            m = random.randint(2, max_m)
            total_min = min_pct * m
            S = 1 - total_min

            # Generate random points and compute splits
            random_points = [random.uniform(0, S) for _ in range(m - 1)]
            random_points.sort()
            points = [0.0] + random_points + [S]
            intervals = [points[i+1] - points[i] for i in range(len(points)-1)]
            splits = [min_pct + interval for interval in intervals]

            # Build split points
            split_points = []
            current = 0.0
            for split in splits:
                current += split
                split_points.append(round(current, 2))
            split_points[-1] = 1.0  # Ensure last point is exactly 1.0

            # Check for duplicates
            if len(split_points) != len(set(split_points)):
                continue

            # Check restricted zone rules
            valid = True
            restricted_in_current = []
            for i, point in enumerate(split_points[:-1]):
                if restricted_zone[0] <= point < restricted_zone[1]:
                    next_point = split_points[i + 1]
                    if (next_point - point) < min_distance:
                        valid = False
                        break
                    restricted_in_current.append(point)
                
                for prev_point in self.RESTRICTED_POINTS:
                    if abs(point - prev_point) < min_distance:
                        valid = False
                        break

            if valid:
                self.RESTRICTED_POINTS.extend(restricted_in_current)
                return split_points
        
        raise ValueError("Failed to generate valid splits after max retries.")

    
    def select_random_depth(self):
        self.max_depth = random.choice(self.depths)

    def select_random_treenum(self ,deviation):
        divisible_options = [num for num in self.treenum_options if num % deviation == 0]
        self.n_estimators = int((random.choice(divisible_options)/deviation))
       
        
def generate_random_split_plots( num_splits, min_pct=0.25, min_distance=0.1, restricted_zone=(0.2, 0.3), max_retries=1000):
    
    RESTRICTED_POINTS = []
    
    total_min = min_pct * num_splits
    if total_min > 1:
        raise ValueError("Total minimum percentage exceeds 1. Reduce num_splits or min_pct.")
    
    S = 1 - total_min  # Remaining space after guaranteeing min_pct for each split

    for _ in range(max_retries):
        
        # Generate random points for internal split positions
        random_points = [random.uniform(0, S) for _ in range(num_splits - 1)]
        random_points.sort()
        points = [0.0] + random_points + [S]
        
        intervals = [points[i+1] - points[i] for i in range(len(points)-1)]
        splits = [min_pct + interval for interval in intervals]

        # Build split points
        split_points = []
        current = 0.0
        for split in splits:
            current += split
            split_points.append(round(current, 2))
        split_points[-1] = 1.0  # Ensure last point is exactly 1.0

        # Check for duplicates
        if len(split_points) != len(set(split_points)):
            continue

        # Check restricted zone rules
        valid = True
        restricted_in_current = []
        for i, point in enumerate(split_points[:-1]):
            if restricted_zone[0] <= point < restricted_zone[1]:
                next_point = split_points[i + 1]
                if (next_point - point) < min_distance:
                    valid = False
                    break
                restricted_in_current.append(point)
            
            for prev_point in RESTRICTED_POINTS:
                if abs(point - prev_point) < min_distance:
                    valid = False
                    break

        if valid:
            RESTRICTED_POINTS.extend(restricted_in_current)
            return split_points

    raise ValueError("Failed to generate valid splits after max retries.")


# this func creats the same split as Sirine has
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


def generate_configurations_for_datasets(datasetnames , num_configurations=100):
    """
    Generates a list of configurations for various datasets.
    Each configuration is a dictionary including dataset_name, split_points, max_depth, and n_estimators.
    """
    all_configurations = []
    
    # Define your datasets
    # datasets = ['Shoaib', 'Epilepsy', 'EMGPhysical', 'SelfRegulationSCP1', 'WESADchest', 'PAMAP2']


    for dataset_name in datasetnames:
        print(f"Generating configurations for dataset: {dataset_name}")
        for _ in range(num_configurations):
            config_obj = Config()
            try:
                splits = config_obj.generate_random_split()
                num_exit = len(splits)
                config_obj.select_random_depth()
                config_obj.select_random_treenum(num_exit)
            except ValueError:
                continue  # Skip configurations with invalid splits
            
            config_data = {
                'dataset_name': dataset_name,
                'split_points': splits, # Store as list of floats
                'max_depth': config_obj.max_depth,
                'n_estimators': config_obj.n_estimators
            }
            all_configurations.append(config_data)
    
    print(f"Generated {len(all_configurations)} configurations across all datasets.")
    return all_configurations



def max_entropy(num_classes):
        probabilities = np.ones(num_classes) / num_classes
        return -np.sum(probabilities * np.log(probabilities))
    

def func_threshold_combinations(num_exits, dataset_max_entropy):
        num_bins = 4
        bin_width = dataset_max_entropy / num_bins
        thresholds = [(i + 1) * bin_width for i in range(num_bins)]  # e.g., [0.5, 1.0, 1.5, 2.0]

        # Generate permutations
        permutations = [list(p) for p in itertools.permutations(thresholds, num_exits -1 )] 

        # Shuffle to randomize selection
        random.shuffle(permutations)

        return permutations
def func_threshold_combinations_fixed_exit_th(num_exits, dataset_max_entropy, exit_index, th_values): 
    num_bins = 5
    bin_width = dataset_max_entropy / num_bins
    thresholds = [round((i + 1) * bin_width,2) for i in range(num_bins)]  # e.g., [0.5, 1.0, 1.5, 2.0]

    th_extracted = random.sample(thresholds,5)

    dictionary = {exit_index: th_extracted} | th_values

    list_th = []

    for i in range(5): # we will have 5 lists
        l = []
        for j in range(1,num_exits):
            if j != exit_index:
                l.append(dictionary[j][0])
            else: 
                l.append(dictionary[j][i])

        list_th.append(l)
    
    return list_th


def generate_configurations_for_datasets_plots(dataset_name):
    
    l=[]
    dataset_name = dataset_name[0]    
   
    random.seed(42)
    exits = [2,3,4] 
    if dataset_name == 'Shoaib':
        n_classes = 7
        my_list = [(50,60), (60,60), (70,75), (50,80)]
        
    if dataset_name == 'Epilepsy':
        n_classes = 4
        my_list = [(10,85), (15,90), (20,80), (25,80)] 
        
    if dataset_name == 'EMGPhysical':
        n_classes = 4
        my_list = [(25,40), (35,40), (40,30), (35,30)]
        
    if dataset_name == 'SelfRegulationSCP1':
        n_classes = 2
        my_list = [(10,20), (20,20), (30,15), (20,15)]
        
    if dataset_name == 'WESADchest':
        n_classes = 3
        my_list = [(20,8), (40,6), (60,4)]
        
    if dataset_name == 'PAMAP2':
        n_classes = 5
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
                    dictionary = {'dataset_name': dataset_name, 'Num_exits': num_exits, 'max_depth': depth, 'tree_splits': tree_splits, 'split_points': proportions, 'th': th_combinations, 'n_estimators': n_est}
                    l.append(dictionary)



    return l
  

 



