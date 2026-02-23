import numpy as np
import pandas as pd
from pruned_decision_tree import PrunedDecisionTree

def test_fit_and_predict():
    df = pd.read_csv("tests/datasets/train_region.csv",sep=";")
    X = df[["a1","a2"]]
    y = df[["Class"]]
    # X = np.array([[0, 0], [1, 1],[0.5,0.5],[0.3,0.3],[0.2,0.2],[0.1,0.1],[0.9,0.9],[0.8,0.8],[0.7,0.7]])
    # y = np.array([0, 1,1,0,0,0,1,1,1])

    model = PrunedDecisionTree()
    model.fit(X, y)

    preds = model.predict(X)
    assert len(preds) == 904
    
    print(model.get_complexity_pruning_cost())
    print(model.get_pruning_accuracy())

def test_clone():
    from sklearn.base import clone
    model = PrunedDecisionTree()
    m2 = clone(model)


if __name__ == "__main__":
    test_fit_and_predict()
    test_clone()