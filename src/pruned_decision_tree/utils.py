import numpy as np
from sklearn.tree import  _tree

def prune_redundant_nodes(clf):
    """
    Post-processes a trained sklearn DecisionTreeClassifier to merge nodes
    where both children are leaves that predict the same class.
    """
    tree = clf.tree_

    def is_leaf(inner_node):
        return tree.children_left[inner_node] == _tree.TREE_LEAF

    def prune_recursive(node):
        # Base case: already a leaf
        if is_leaf(node):
            return

        left = tree.children_left[node]
        right = tree.children_right[node]

        # Post-order: visit children first
        prune_recursive(left)
        prune_recursive(right)

        # After visiting children, check if they are now leaves
        if is_leaf(left) and is_leaf(right):
            # Get the predicted class for both leaves
            # value contains the count of samples for each class
            left_class = np.argmax(tree.value[left])
            right_class = np.argmax(tree.value[right])

            if left_class == right_class:
                # Merge: make current node a leaf
                tree.children_left[node] = _tree.TREE_LEAF
                tree.children_right[node] = _tree.TREE_LEAF
                # Optional: update the node's feature/threshold to signify it's a leaf
                tree.feature[node] = _tree.TREE_UNDEFINED
                tree.threshold[node] = _tree.TREE_UNDEFINED

    prune_recursive(0)