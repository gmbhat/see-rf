from DecisionTree import DecisionTree
import numpy as np
from collections import Counter
import concurrent.futures
import logging



def _predict_from_a_list_of_starting_nodes(args):
    tree, X, exit_level, start_nodes = args
    labels, nodes = tree.predict_batch(X, exit_level, start_nodes)
    return list(zip(labels, nodes))

class RandomForest:
    def __init__(self, n_trees=10, max_depth=10, min_samples_split=3, n_feature=None):
        self.n_trees = n_trees
        self.max_depth=max_depth
        self.min_samples_split=min_samples_split
        self.n_features=n_feature
        self.trees = []
        self.executor = concurrent.futures.ProcessPoolExecutor(max_workers=5)
        #self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=5)

    # parallel trees building
    #def fit(self, X, y, proportions, tree_splits):
     #   self.trees = []
      #  n_estimators = self.n_trees
       # BATCH_SIZE = n_estimators
       # num_batches = int(np.ceil(n_estimators/BATCH_SIZE))

        #for b in range(0, num_batches):
         #   batch_start = b*BATCH_SIZE
          #  batch_end = (b+1)*BATCH_SIZE

           # if(batch_end > n_estimators):
            #    batch_end = n_estimators


            #random_seeds = np.random.randint(0, 1000000, size=batch_end - batch_start)
            #args_list = [(self, X, y, proportions, tree_splits, seed) for seed in random_seeds]


            #with concurrent.futures.ProcessPoolExecutor(max_workers=5) as executor:
             #   batch_results = list(executor.map(_fit_one_tree_wrapper, args_list))

            #self.trees.extend(batch_results)



    # building trees with boosting
    def fit(self, X, y, proportions, tree_splits):
        self.trees = []

        # create initial weights for each class
        labels = list(np.unique(y))
        n_classes = len(labels)
        class_weights = {label: 1/n_classes for label in labels} # every each class has a weight

        for tree_number in range(self.n_trees):
          # build one tree
          tree = DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split, n_features=self.n_features)
          rng = np.random.default_rng()
          X_sample, y_sample, X_oob, y_oob = self._weighted_bootstrap_samples(X, y, class_weights, rng)
          tree.fit(X_sample, y_sample, proportions, tree_splits)
          self.trees.append(tree)

          # if we are in the last tree
          if tree_number == self.n_trees - 1:
            continue
          # if we are not in the last tree
          else:
            # evaluate out of bag accuracies
            if len(X_oob)==0:
              continue
            predictions, _ = tree.predict_batch(X_oob, len(proportions), start_nodes=None) # predict function from my decision tree

            errors = {}
            for label in labels:
                mask = (y_oob == label)
                n_cls = np.sum(mask)
                if n_cls > 0:
                    errors[label] = np.mean(predictions[mask] != label)
                else:
                    errors[label] = 0.5

            importances = {}
            total_importance = 0
            for i in class_weights.keys():
              importances[i] = class_weights[i] * (1 + errors[i])
              total_importance += importances[i]


            # update the next label weights
            class_weights = {i: importances[i] / total_importance for i in labels}




    def _weighted_bootstrap_samples(self, X, y, class_weights, rng=None):
        rng = rng or np.random
        n_samples = X.shape[0]
        class_labels = np.array(list(class_weights.keys()))
        weights_array = np.array(list(class_weights.values()))

        #  weights sum to 1
        total_weight = weights_array.sum()
        if not np.isclose(total_weight, 1.0):
             if total_weight > 0:
                 weights_array /= total_weight
             else:                                                  # if all weights are zero
                 weights_array = np.ones(len(class_labels)) / len(class_labels)


        class_indices_map = {cls: np.where(y == cls)[0] for cls in class_labels}

        in_bag_indices = []
        valid_classes = [cls for cls in class_labels if len(class_indices_map[cls]) > 0]
        if not valid_classes:
          return np.array([]), np.array([]), np.array([]), np.array([])

        # Adjust probabilities only for classes present
        valid_weights_array = np.array([class_weights[c] for c in valid_classes])
        valid_weights_array /= valid_weights_array.sum() # Renormalize


        #for _ in range(n_samples):
         #   target_class = rng.choice(valid_classes, p=valid_weights_array)
          #  indices_for_class = class_indices_map[target_class]
           # # Should always find a sample now if class was valid
            #chosen_index = rng.choice(indices_for_class)
            #in_bag_indices.append(chosen_index)

        chosen_classes = rng.choice(valid_classes, size=n_samples, p=valid_weights_array)
        in_bag_indices = np.array([rng.choice(class_indices_map[c]) for c in chosen_classes])


        oob_indxs = np.setdiff1d(np.arange(n_samples), in_bag_indices)
        return X[in_bag_indices], y[in_bag_indices], X[oob_indxs], y[oob_indxs]


    def _most_common_label(self, y):
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common


    def predict(self, X, n_classes, exit_level, start_nodes=None): # exit_level should be <= max_depth, start_nodes is a list of lists of starting nodes for every each input data,
                                                                    # for every each tree
        trees = self.trees
        if start_nodes is None:
          start_nodes = [[None for i in range(len(trees))] for j in range(X.shape[0])]

        args_list = [(trees[i], X, exit_level, [row[i] for row in start_nodes]) for i in range(len(trees))]

        preds = list(self.executor.map(_predict_from_a_list_of_starting_nodes, args_list))

        predictions = np.array([[e[0] for e in tree] for tree in preds], dtype=object)
        exit_nodes = np.array([[e[1] for e in tree] for tree in preds], dtype=object)

        tree_preds = np.swapaxes(predictions, 0, 1)
        tree_exit_nodes = np.swapaxes(exit_nodes, 0, 1)

        predictions_2 = np.array([self._most_common_label(pred) for pred in tree_preds])
        probabilities = self._prob(X, tree_preds, n_classes)
        return predictions_2, tree_exit_nodes, probabilities # for each data input return: one single label, a list of exit_nodes for each tree, probabilities list
                                              # predictions_2: [label_1, label_2,..., label_{num_samples}],
                                              #[tree_exit_nodes: [exit_11, exit_12,..], ..., [exit_{num_samples}1,...]]
                                              # probabilities: [[probabilites for x1], [probabilites for x2],...]




    #def _prob(self, X, tree_preds, n_classes):
     # probabilities = []
    #  for x_p in tree_preds:
     #   counter = Counter(x_p)
      #  prob = np.zeros(n_classes)
       # for label, count in counter.items():
        #  prob[label] = count/len(x_p)
      #  probabilities.append(prob)
     # return np.array(probabilities)

    def _prob(self, X, tree_preds, n_classes):
      tree_preds = np.array(tree_preds, dtype=np.int64)
      counts = np.apply_along_axis(np.bincount, 1, tree_preds, minlength=n_classes)
      return counts/tree_preds.shape[1]


    def get_total_nodes(self):
        return sum(tree.get_total_nodes() for tree in self.trees)

    def get_nodes_per_tree(self):
        return [tree.get_total_nodes() for tree in self.trees]