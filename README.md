## Decision Tree Classifier

This repository contains a simple implementation of a decision tree classifier in Python. The classifier is capable of binary classification tasks using entropy-based information gain for optimal separation. The repository includes functions for calculating entropy, determining optimal feature separation, building the decision tree, and making predictions.

### Tasks Overview:

1. **Calculation of Entropy**:
   - The `get_entropy` function computes the entropy of a given binary split based on the counts of two categories.
   - Entropy is used to measure the impurity or uncertainty in a dataset, which is crucial for decision tree learning.

2. **Optimal Feature Separation**:
   - The `get_best_separation` function identifies the best feature and threshold to split the dataset, maximizing information gain.
   - It utilizes entropy calculations to evaluate the purity of potential splits and select the most informative feature.

3. **Decision Tree Implementation**:
   - The `build_tree` function recursively constructs the decision tree based on the identified optimal separations.
   - Once built, the tree can be used for making predictions on new data using the `predict` function.

### Usage:

To run the decision tree classifier:
1. Prepare your training and testing datasets in CSV format (`train.csv` and `test.csv`).
2. Update the file paths in the `main` function to point to your dataset files.
3. Execute the script to train the decision tree on the training data and generate predictions for the test data.
4. The predicted labels will be saved to `results.csv`.

This implementation serves as a foundational example of decision tree learning for educational purposes.
