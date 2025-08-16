# Scikit-learn Compatible Estimators Powered by ReHLine

## Project Description

This project provides a general-purpose estimator framework for empirical risk minimization with linear constraints, built on top of the `ReHLine` optimization method. It is designed to solve both classification and regression problems using a consistent and extensible interface. The estimator supports a broad class of piecewise linear-quadratic (PLQ) loss functions, with the flexibility to enforce domain-specific constraints such as non-negativity or fairness. The framework integrates tightly with scikit-learn, enabling compatibility with standard tools such as `GridSearchCV`, `Pipeline`, and model evaluation utilities. It is intended for users who require a modular and interpretable modeling pipeline that supports constraint-based learning.

## Key Features

1. The core class `plqERM_Ridge` serves as a base implementation for both classifiers and regressors. It automatically maps PLQ loss specifications and constraints into internal ReHLine solver-compatible matrices (`U`, `V`, `Tau`, `S`, `T`, `A`, `b`) without manual intervention. This simplifies the design of new models while providing a stable and extensible foundation for constraint-based learning

2. Support for multiple loss functions including hinge (SVM), quantile regression (QR), smooth SVM, total variation (TV), Huber, SVR, and MAE

3. Subclasses `plqERMClassifier` and `plqERMRegressor` expose task-specific behavior and integrate seamlessly with scikit-learn's classification and regression utilities

4. Automatic conversion of user-defined `loss` and `constraint` dictionaries into ReHLine parameter matrices (U, V, Tau, S, T, A, b)

5. Compatibility with `GridSearchCV` for automated hyperparameter tuning, including nested dictionaries as search space

6. Built-in methods for decision function computation, prediction, and task-aware scoring (accuracy for classification, R² for regression)

7. Support for optional sample weighting, warm-start initialization, verbose tracing, and trace logging

   

- Contributors: Youtong Li
- Mentors: [Ben Dai](https://www.bendai.org/)
- Time Period: **6 Months**
- Languages: Python 