import numpy as np


####################### Task 1, calculation of entropy ########################
def get_entropy(n_cat1: int, n_cat2: int) -> float:
    if n_cat1 == 0 or n_cat2 == 0:
        return 0

    total = n_cat1 + n_cat2

    if total == 0:
        return 0

    p_cat1 = n_cat1 / total
    p_cat2 = n_cat2 / total

    entropy = -p_cat1 * np.log2(p_cat1) - p_cat2 * np.log2(p_cat2)

    return entropy


def get_information_gain(parent_entropy: float, child_entropies: list, child_weights: list) -> float:
    weighted_child_entropy = np.sum(
        np.multiply(child_weights, child_entropies))
    information_gain = parent_entropy - weighted_child_entropy
    return information_gain

###################### Task 2, optimal separation #############################
def get_best_separation(features: list, labels: list) -> (int, int):
    num_features = len(features[0])
    num_samples = len(labels)

    parent_entropy = get_entropy(np.count_nonzero(
        labels == 0), np.count_nonzero(labels == 1))

    best_feature = 0
    best_threshold = 0
    max_information_gain = -np.inf

    for feature_idx in range(num_features):
        sorted_samples = sorted(zip(features, labels),
                                key=lambda x: x[0][feature_idx])
        sorted_features, sorted_labels = zip(*sorted_samples)

        for threshold_idx in range(1, num_samples):
            left_labels = sorted_labels[:threshold_idx]
            right_labels = sorted_labels[threshold_idx:]

            left_entropy = get_entropy(np.count_nonzero(
                left_labels == 0), np.count_nonzero(left_labels == 1))
            right_entropy = get_entropy(np.count_nonzero(
                right_labels == 0), np.count_nonzero(right_labels == 1))

            child_entropies = np.array([left_entropy, right_entropy])
            child_weights = np.array(
                [len(left_labels) / num_samples, len(right_labels) / num_samples])

            info_gain = get_information_gain(
                parent_entropy, child_entropies, child_weights)

            if info_gain > max_information_gain:
                max_information_gain = info_gain
                best_feature = feature_idx
                best_threshold = sorted_features[threshold_idx][feature_idx]

    return best_feature, best_threshold


def build_tree(features, labels):
    best_feature_index, best_value = get_best_separation(features, labels)

    if best_value == 0:
        return np.argmax(np.bincount(labels))

    true_features = features[features[:, best_feature_index] <= best_value]
    true_labels = labels[features[:, best_feature_index] <= best_value]

    false_features = features[features[:, best_feature_index] > best_value]
    false_labels = labels[features[:, best_feature_index] > best_value]

    true_branch = build_tree(true_features, true_labels)
    false_branch = build_tree(false_features, false_labels)

    return best_feature_index, best_value, true_branch, false_branch


def predict(tree, instance):
    if isinstance(tree, int):  # Leaf node
        return tree

    feature_index, value, true_branch, false_branch = tree

    if instance[feature_index] <= value:
        return predict(true_branch, instance)
    else:
        return predict(false_branch, instance)

################### Task 3, implementing a decision tree ######################
def main():
    train_data = np.genfromtxt('train.csv', delimiter=',', dtype=int)
    X_train = train_data[:, :-1]
    y_train = train_data[:, -1]
    X_test = np.genfromtxt('test.csv', delimiter=',')
    dec_tree = build_tree(X_train, y_train)

    y_pred = [predict(dec_tree, sample) for sample in X_test]

    np.savetxt('results.csv', y_pred, fmt='%d', delimiter=',')

    return 0


if __name__ == "__main__":
    main()
