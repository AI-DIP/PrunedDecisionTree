import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.base import ClassifierMixin, BaseEstimator
from scipy.stats import ttest_rel

from .utils import prune_redundant_nodes

class PrunedDecisionTree(ClassifierMixin, BaseEstimator):
    """
    """
    
    def __init__(
        self,
        *,
        criterion="gini",
        splitter="best",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=None,
        random_state=None,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        class_weight=None,
        ccp_alpha=0.0,
        monotonic_cst=None,
    ):
        self.criterion=criterion
        self.splitter=splitter
        self.max_depth=max_depth
        self.min_samples_split=min_samples_split
        self.min_samples_leaf=min_samples_leaf
        self.min_weight_fraction_leaf=min_weight_fraction_leaf
        self.max_features=max_features
        self.random_state=random_state
        self.max_leaf_nodes=max_leaf_nodes
        self.min_impurity_decrease=min_impurity_decrease
        self.class_weight=class_weight
        self.ccp_alpha=ccp_alpha
        self.monotonic_cst=monotonic_cst
        
        self._tree_estimator = None  
        self.complexity_pruning_cost = None
        self.pruning_accuracy = None
        self.delta:float = 0.05
    
    def _create_copy_estimator(self, ccp_alpha=None):
        if ccp_alpha is None: ccp_alpha=self.ccp_alpha
        temp_estimator = DecisionTreeClassifier(
                criterion=self.criterion,
                splitter=self.splitter,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                max_features=self.max_features,
                random_state=self.random_state,
                max_leaf_nodes=self.max_leaf_nodes,
                min_impurity_decrease=self.min_impurity_decrease,
                class_weight=self.class_weight,
                ccp_alpha=ccp_alpha,
                monotonic_cst=self.monotonic_cst
                )
        return temp_estimator
    
    def fit(self, 
            X, 
            y,
            sample_weight=None,
            check_input:bool=True
        ):
        #GET COMPLEXITY COST
        temp_estimator = self._create_copy_estimator()
        #temp_estimator.fit(X,y)
        self.complexity_pruning_cost = temp_estimator.cost_complexity_pruning_path(X,y)

        #CHECK ALL CCP ALPHA
        self.pruning_accuracy = {"ccp_alphas":[],"acc_mean":[], "acc_folds":[]}
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        
        for ccp_alpha in self.complexity_pruning_cost.ccp_alphas:
            est = self._create_copy_estimator(ccp_alpha=ccp_alpha)
            res = cross_validate(est, X,y,cv=cv, scoring=['accuracy'])
            
            self.pruning_accuracy["ccp_alphas"].append(ccp_alpha)
            self.pruning_accuracy["acc_folds"].append(res["test_accuracy"])
            self.pruning_accuracy["acc_mean"].append(res["test_accuracy"].mean())
        
        #SELECT BEST CCP_ALPHA
        best_ccp_alpha_index = np.argmax(self.pruning_accuracy["acc_mean"])
        best_folds_scores = self.pruning_accuracy["acc_folds"][best_ccp_alpha_index]
        
        best_selected_ccp_index = best_ccp_alpha_index
        
        for i in range(len(self.pruning_accuracy["ccp_alphas"])):
            scores = self.pruning_accuracy["acc_folds"][i]
            t_stat, p_value = ttest_rel(best_folds_scores, scores)
            # print(f"CCP:{self.pruning_accuracy['ccp_alphas'][i]} T:{t_stat} P:{p_value} ACC:{self.pruning_accuracy['acc_mean'][i]}")
            if p_value > self.delta:
                best_selected_ccp_index = i
        
        self.ccp_alpha = self.pruning_accuracy["ccp_alphas"][best_selected_ccp_index]
        # print(f"### CCP_ALPHA: {self.ccp_alpha}")
        if(self.ccp_alpha < 0):self.ccp_alpha=0
        
        self._tree_estimator = self._create_copy_estimator(ccp_alpha=self.ccp_alpha)
        self._tree_estimator.fit(X,y,sample_weight,check_input)
                
        prune_redundant_nodes(self._tree_estimator)

        return self

    def get_complexity_pruning_cost(self):
        return self.complexity_pruning_cost

    def get_cpp_alphas(self):
        return self.complexity_pruning_cost.cpp_alphas
    
    def get_impurities(self):
        return self.complexity_pruning_cost.impurities
    
    def get_pruning_accuracy(self):
        return self.pruning_accuracy

    def get_ccp_alpha(self):
        return self.ccp_alpha
    
    def predict(self, X, check_input: bool = True):
        return self._tree_estimator.predict(X,check_input=check_input)
    
    def get_feature_importances(self):
        return self._tree_estimator.feature_importances_