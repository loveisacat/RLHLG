#Created by Stephen 15 Jun,2023

import numpy as np
import graphviz

TREE_DATA = {
    'cart': {
        'feature': np.array([[0], [0], [2], [-2], [-2], [2], [-2], [1], [2], [-2], [-2], [3],
                             [-2], [-2], [2], [1], [3], [-2], [-2], [2], [-2], [-2], [-2]]),
        'children_left': np.array([1, 2, 3, -1, -1, 6, -1, 8, 9, -1, -1, 12, -1, -1, 15, 16, 17, -1, -1, 20, -1, -1, -1]),
        'children_right': np.array([14, 5, 4, -1, -1, 7, -1, 11, 10, -1, -1, 13, -1, -1, 22, 19, 18, -1, -1, 21, -1, -1, -1])
    }
}

class DecisionTree:
    def __init__(self):
        self.tree = {}

    def build_tree(self, feature, children_left, children_right, depth=0, node_id=0):
        if node_id == -1:
            return None

        node = {}
        node['feature'] = feature[node_id]
        node['depth'] = depth

        left_child = children_left[node_id]
        right_child = children_right[node_id]

        if left_child == -1 and right_child == -1:
            # 叶子节点，无需继续划分
            node['left'] = None
            node['right'] = None
        else:
            # 非叶子节点，递归构建左右子树
            node['left'] = self.build_tree(feature, children_left, children_right, depth + 1, left_child)
            node['right'] = self.build_tree(feature, children_left, children_right, depth + 1, right_child)

        return node

    def fit(self, feature, children_left, children_right):
        self.tree = self.build_tree(feature, children_left, children_right)

    def predict(self, X):
        # 使用训练好的决策树进行预测
        # 根据决策树的规则路径选择相应的输出
        pass

# 创建决策树对象
dt = DecisionTree()

# 使用已有的特征和节点信息进行构建
tree_dict = TREE_DATA['cart'] 
dt.fit(tree_dict['feature'], tree_dict['children_left'], tree_dict['children_right'])

print(len(tree_dict['feature']))
print(len(tree_dict['children_left']))
print(len(tree_dict['children_right']))
print(dt.tree)

exit(0)

# 使用测试数据进行预测
predictions = dt.predict(X_test)

