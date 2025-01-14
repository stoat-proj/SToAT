# Matrix Factorization Optimization with Various Loss Functions

## Introduction

Matrix factorization is a fundamental technique in machine learning and data analysis, used extensively in areas such as recommendation systems, image processing, and dimensionality reduction. The goal of matrix factorization is to decompose a large matrix into the product of two or more smaller matrices, capturing the underlying structure of the data. Different applications and datasets require different **loss functions** to optimize the factorization process. This proposal explores the optimization of matrix factorization using various loss functions, aiming to enhance flexibility and accuracy in diverse contexts.

## Math Formulation

**Parameters:**

- $n$ as the number of rows in the matrix.
- $m$ as the number of columns in the matrix.
- $r$ as the rank of the factorization.
- $\Omega$ as the index set for observed entries.
- $y_{ij}$ as the observed value at position $(i, j)$.

**Variables:**

- $\mathbf{P} \in \mathbb{R}^{n \times r}$ and $\mathbf{Q} \in \mathbb{R}^{m \times r}$ as the factor matrices.
- $\mathbf{a} \in \mathbb{R}^n$ and $\mathbf{b} \in \mathbb{R}^m$ as the bias vectors.
- $\mu \in \mathbb{R}$ as the global bias term. 

Furthermore, the prediction for $(i,j)$ entry is formulated as:

$$
\hat{y}_{ij} = \mathbf{p}_i^\intercal \mathbf{q}_j + \mathbf{a}_i + \mathbf{b}_j + \mu,
$$

**Objective Function:**

The objective function for matrix factorization can be formulated as:

$$
\min_{\mathbf{P}, \mathbf{Q}, \mathbf{a}, \mathbf{b}, \mu} \sum_{(i,j) \in \Omega} \phi(y_{ij}, \mathbf{p}_i^\intercal \mathbf{q}_j + \mathbf{a}_i + \mathbf{b}_j + \mu),
$$

where $\phi$ is a loss function that measures the difference between the original matrix $\mathbf{Y}$ and the predicted outcome. Common choices for $\phi$ include:

- Regression loss: $y \in \mathbb{R}$:
  - **Squared Loss**: $\phi(y, \widehat{y}) = (y - \widehat{y})^2$,
  - **Absolute Loss**: $\phi(y, \widehat{y}) = |y - \widehat{y}|$,

- Classification loss: $y \in \{-1, 1\}$:
  - **Hinge Loss**: $\phi(y, \widehat{y}) = \max(0, 1 - y \widehat{y})$.

## Motivation

[ReHLine](https://rehline-python.readthedocs.io/en/latest/) is a powerful solver to solve ERM with various loss functions [^rehline], and using blockwise coordinate descent, each subproblem of MF can be regarded as a ReHLine problem. Thus, the primary aim of this project is to extend the capabilities of ReHLine to solve matrix factorization problems with different loss functions. 


## Aims

1. **Integrate ReHLine with Matrix Factorization**: Adapt the ReHLine solver to handle matrix factorization problems by formulating each subproblem within the factorization process as a ReHLine problem. This involves leveraging blockwise coordinate descent to efficiently solve for factor matrices.

2. **Enhance Computational Efficiency**: Optimize the implementation to ensure that the integration of ReHLine with matrix factorization is computationally efficient, scalable, and capable of handling large datasets. This includes refining the blockwise coordinate descent approach for matrix factorization contexts.

3. **Implement and Test the Extended Solver**: Develop a robust software implementation that incorporates the extended capabilities of ReHLine for matrix factorization. Conduct extensive testing on a variety of datasets to validate the accuracy and performance of the solver.

4. **Benchmark Against Existing Methods**: Compare the performance of the ReHLine-based matrix factorization approach with existing matrix factorization methods. Analyze the results to identify strengths and areas for improvement, focusing on the solver's ability to handle diverse data characteristics and loss functions.

5. **Documentation and User Guide Development**: Create comprehensive documentation and a user guide to facilitate the use of the extended ReHLine solver for matrix factorization. Ensure that the documentation is accessible and provides clear instructions for users to apply the solver to their specific applications.

By achieving these aims, the project seeks to enhance the versatility and applicability of the ReHLine solver in the realm of matrix factorization, enabling it to tackle a broader range of data analysis and machine learning challenges.

- Contributors: Xiaochen Su
- Mentors: [Ben Dai](https://www.bendai.org/)
- Time Period: **6 Months**
- Languages: Python and C++


## Reference

[^rehline]: Dai, B., & Qiu, Y. (2024).  ReHLine: Regularized Composite ReLU-ReHU Loss Minimization with Linear  Computation and Linear Convergence. *Advances in Neural Information Processing Systems*, *36*.
