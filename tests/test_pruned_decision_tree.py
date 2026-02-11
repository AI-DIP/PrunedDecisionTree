import numpy as np
from pruned_decision_tree import PrunedDecisionTree

def test_fit_and_predict():
    X = np.array([[0, 0], [1, 1],[0.5,0.5],[0.3,0.3],[0.2,0.2],[0.1,0.1]])
    y = np.array([0, 1,1,0,0,0])

    model = PrunedDecisionTree(auto_prune=True)
    model.fit(X, y)

    preds = model.predict(X)
    assert len(preds) == 6
    
    print(model.get_complexity_pruning_cost())
    print(model.get_pruning_accuracy())

if __name__ == "__main__":
    test_fit_and_predict()