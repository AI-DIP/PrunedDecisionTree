from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate

from .utils import prune_redundant_nodes
class PrunedDecisionTree(DecisionTreeClassifier):
    """
    """
    auto_prune:bool = False
    complexity_pruning_cost = None
    pruning_accuracy = None
    
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
        auto_prune:bool = False
    ):
        super().__init__(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            random_state=random_state,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            class_weight=class_weight,
            ccp_alpha=ccp_alpha,
            monotonic_cst=monotonic_cst
        )
        self.auto_prune = auto_prune
    
    def _create_copy_estimator(self, ccp_alpha=None):
        if ccp_alpha is None: ccp_alpha=self.ccp_alpha
        temp_estimator = PrunedDecisionTree(
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
            check_input:bool=True,
            calc_complexity:bool=False,
            auto_prune:bool=False):
        
        if auto_prune or self.auto_prune:
                temp_estimator = self._create_copy_estimator()
                temp_estimator.fit(X,y,calc_complexity=True)
                self.complexity_pruning_cost = temp_estimator.cost_complexity_pruning_path(X,y)

                self.pruning_accuracy = {"ccp_alphas":[],"acc":[]}
                cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
                for ccp_alpha in self.complexity_pruning_cost.ccp_alphas:
                    est = self._create_copy_estimator(ccp_alpha=ccp_alpha)
                    res = cross_validate(est, X,y,cv=cv, scoring=['accuracy'])
                    self.pruning_accuracy["ccp_alphas"].append(ccp_alpha)
                    self.pruning_accuracy["acc"].append(res["test_accuracy"].mean())
                
                self.ccp_alpha = self.pruning_accuracy["ccp_alphas"][max(range(len(self.pruning_accuracy["acc"])), key=lambda i: self.pruning_accuracy["acc"][i])]
                if(self.ccp_alpha < 0):self.ccp_alpha=0
        
        
        super().fit(X,y,sample_weight,check_input)
        
        if calc_complexity and not(auto_prune): self.complexity_pruning_cost = self.cost_complexity_pruning_path(X,y)
        
        if auto_prune:prune_redundant_nodes(self)

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
