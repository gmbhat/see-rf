import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.random import check_random_state
import warnings 
from sklearn.tree import _tree

class SequentialRandomForest(BaseEstimator, ClassifierMixin):
    
    def __init__(self,
                 n_estimators=None,
                 max_depth=None,
                 tree_splits = None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 random_state=None,
                 initial_class_weights=None,
                 initial_sample_weights=None): # weights passed from the previous sub-forest
        
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.initial_class_weights = initial_class_weights 
        self.initial_sample_weights = initial_sample_weights
        self.tree_splits = tree_splits
        
        self._final_class_weights = {}              # <<< Store final weights     
        self._final_sample_weights = None           # per-sample weights after the last tree
        
        self.node_counts = []          

    def _weighted_bootstrap_sample(self, n_samples, y, class_weights, random_state):

        class_labels = np.array(list(class_weights.keys()))
        weights_array = np.array([class_weights[c] for c in class_labels])

        #  weights sum to 1 
        total_weight = weights_array.sum()
        if not np.isclose(total_weight, 1.0):
             if total_weight > 0:
                 weights_array /= total_weight
             else:                                                  # if all weights are zero 
                 weights_array = np.ones(len(class_labels)) / len(class_labels)


        class_indices_map = {
            cls: np.where(y == cls)[0] for cls in class_labels
        }

        in_bag_indices = []
        valid_classes = [cls for cls in class_labels if len(class_indices_map[cls]) > 0]
        if not valid_classes:                                   # Handle case where y has no samples somehow
            return np.array([], dtype=int)

        # Adjust probabilities only for classes present
        valid_weights_array = np.array([class_weights[c] for c in valid_classes])
        valid_weights_array /= valid_weights_array.sum() # Renormalize

        for _ in range(n_samples):
            target_class = random_state.choice(valid_classes, p=valid_weights_array)
            indices_for_class = class_indices_map[target_class]
            # Should always find a sample now if class was valid
            chosen_index = random_state.choice(indices_for_class)
            in_bag_indices.append(chosen_index)

        return np.array(in_bag_indices)


    def fit(self, X, y):
     
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        self.n_classes_ = len(self.classes_)
        n_samples, self.n_features_in_ = X.shape
        random_state = check_random_state(self.random_state)

        self.trees_ = []
        self.tree_oob_errors_ = []                 # Stores  OOB errors for each tree
        
        
        # --- Initial per-sample weights (Algorithm 2, line 1) ---
        if self.initial_sample_weights is not None:
            # weights passed from the final tree of the previous sub-forest
            sample_weights = np.asarray(self.initial_sample_weights, dtype=float).copy()
            if sample_weights.shape[0] != n_samples:
                raise ValueError("initial_sample_weights must have one weight per training sample")
        else:
            sample_weights = np.full(n_samples, 1.0 / n_samples)   # w(i) = 1/N
        sample_weights /= sample_weights.sum()


        # Tree Building ---
        for i in range(self.n_estimators):
        # for i in enumerate (self.tree_splits):
            # Weighted bootstrap: each sample is drawn with probability w(i) (Algorithm 2, line 6)
            in_bag_indices = random_state.choice(
                n_samples, size=n_samples, replace=True, p=sample_weights
            )
            oob_indices = np.setdiff1d(np.arange(n_samples), np.unique(in_bag_indices), assume_unique=True)

            if len(in_bag_indices) == 0:
                 warnings.warn(f"Tree {i+1}/{self.n_estimators} had empty in-bag sample. Skipping tree fitting and updates for this step.", UserWarning)
                 # Add a dummy predictor or handle as needed
                 
                 self.trees_.append(None) 
                 
                 self.tree_oob_errors_.append({cls: 0.5 for cls in self.classes_}) # Neutral error
                 continue 


            X_train = X[in_bag_indices]
            y_train = y[in_bag_indices]

            # 2. Train a Decision Tree
            
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                random_state=random_state # Pass down random state
            )

            # Handle cases where the bootstrap sample has only one class
            unique_y_train = np.unique(y_train)
            if len(unique_y_train) < 2:
                from sklearn.dummy import DummyClassifier
                # Suppress potential warning about DummyClassifier behavior if needed
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=UserWarning)
                    tree = DummyClassifier(strategy="most_frequent")
                    tree.fit(X_train, y_train) # Needs fit to set classes_ attribute
            else:
                tree.fit(X_train, y_train)

            self.trees_.append(tree)
            
            # 4. Access the underlying tree structure
            tree_structure = tree.tree_
            
            # 5. Get the total number of nodes
            num_nodes = tree_structure.node_count
            self.node_counts.append(num_nodes) 
            
            # 3. Calculate Class-Specific OOB Error
            tree_oob_errors_current = {}
            if len(oob_indices) > 0:
                X_oob = X[oob_indices]
                y_oob = y[oob_indices]
                try:
                    # Need to handle dummy classifier predict slightly differently if used
                    if hasattr(tree, 'predict_proba'): # Standard DT
                         y_oob_pred = tree.predict(X_oob)
                    else: # Dummy classifier
                         y_oob_pred = tree.predict(X_oob) # Dummy predict works

                except Exception as e:
                    warnings.warn(f"Could not predict OOB for tree {i+1}. Error: {e}. Assigning neutral error.", UserWarning)
                    y_oob_pred = None # Flag that prediction failed

                if y_oob_pred is not None:
                    for cls in self.classes_:
                        cls_mask_oob = (y_oob == cls)
                        n_cls_oob = np.sum(cls_mask_oob)
                        if n_cls_oob > 0:
                            errors_cls = np.sum((y_oob_pred[cls_mask_oob] != cls))
                            tree_oob_errors_current[cls] = errors_cls / n_cls_oob
                        else:
                            tree_oob_errors_current[cls] = 0.5 # Default neutral error
                else:
                    tree_oob_errors_current = {cls: 0.5 for cls in self.classes_} # Neutral on failure
            else:
                # No OOB samples
                 tree_oob_errors_current = {cls: 0.5 for cls in self.classes_}

            self.tree_oob_errors_.append(tree_oob_errors_current)

            # 4. Update sample weights (Algorithm 2, lines 10-16):
            #    only OOB samples that this tree misclassified are boosted by (1 + E_t(c));
            #    correctly classified samples keep their weight. Then normalize.
            if len(oob_indices) > 0 and y_oob_pred is not None:
                wrong = (y_oob_pred != y_oob)
                for idx, cls in zip(oob_indices[wrong], y_oob[wrong]):
                    sample_weights[idx] *= (1.0 + tree_oob_errors_current[cls])
                sample_weights /= sample_weights.sum()

        self._final_sample_weights = sample_weights.copy()
        # class-level summary of the final weights (kept for logging only)
        self._final_class_weights = {cls: float(sample_weights[y == cls].sum()) for cls in self.classes_}

        self.is_fitted_ = True
        return self

    
    def get_final_class_weights(self):
        """Returns the class weights after the last iteration."""
        check_is_fitted(self)
        if  self._final_class_weights is None:
             # Fallback if fitting didn't complete properly
             return {cls: 1.0 / self.n_classes_ for cls in self.classes_}
        return self._final_class_weights

    def get_final_sample_weights(self):
        """Returns the per-sample weights after the last tree (passed to the next sub-forest)."""
        check_is_fitted(self)
        return self._final_sample_weights

    def predict_proba(self, X):
        """Predict class probabilities for X using weighted voting."""
        check_is_fitted(self)
        X = check_array(X)
        n_samples = X.shape[0]

        scores = np.zeros((n_samples, self.n_classes_))

        valid_tree_count = 0
        for i, tree in enumerate(self.trees_):
            if tree is None: # Skip placeholders for failed trees
                continue

            tree_errors = self.tree_oob_errors_[i]
            try:
                predictions = tree.predict(X)
            except Exception as e:
                warnings.warn(f"Could not predict with tree {i+1}. Error: {e}. Skipping tree.", UserWarning)
                continue # Skip this tree if prediction fails

            valid_tree_count += 1
            for sample_idx in range(n_samples):
                predicted_class = predictions[sample_idx]
                # Ensure predicted class is known
                if predicted_class in self.classes_:
                     class_idx = np.where(self.classes_ == predicted_class)[0][0]
                     error = tree_errors.get(predicted_class, 0.5)
                     weight = 1.0 - error
                     scores[sample_idx, class_idx] += weight
                # else: Ignore prediction if class is somehow unknown

        # Normalize scores
        scores_sum = np.sum(scores, axis=1)
        
        # Initialize probabilities with uniform distribution
        proba = np.full(scores.shape, 1.0 / self.n_classes_)
        # Avoid division by zero - only normalize where sum is positive
        valid_rows = scores_sum > 1e-9
        if np.any(valid_rows):
            proba[valid_rows] = scores[valid_rows] / scores_sum[valid_rows][:, np.newaxis]

        # Handle case where no trees made valid predictions (e.g., all failed)
        if valid_tree_count == 0:
            warnings.warn("No valid trees found for prediction. Returning uniform probabilities.", UserWarning)
            # proba is already uniform in this case

        return proba

    def predict(self, X):
        """Predict class for X."""
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)] 
    # I ADD proba only for baseline version, remove the proba !!!!!
