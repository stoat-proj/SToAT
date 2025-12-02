# Matrix Factorization Optimization with Various Loss Functions

## Introduction

Matrix factorization is a fundamental technique in machine learning and data analysis, used extensively in areas such as recommendation systems, image processing, and dimensionality reduction. The goal of matrix factorization is to decompose a large matrix into the product of two or more smaller matrices, capturing the underlying structure of the data. Different applications and datasets require different **loss functions** to optimize the factorization process. This proposal explores the optimization of matrix factorization using various loss functions, aiming to enhance flexibility and accuracy in diverse contexts.

## Math Formulation

**Objective Function:**

The objective function for matrix factorization can be formulated as:

$$
\min_{\substack{
    \mathbf{P} \in \mathbb{R}^{n \times k}\ 
    \pmb{\alpha} \in \mathbb{R}^n \\
    \mathbf{Q} \in \mathbb{R}^{m \times k}\ 
    \pmb{\beta} \in \mathbb{R}^m
}} 
\left[
    \sum_{(u,i)\in \Omega} C \cdot \text{PLQ}(r_{ui}, \ \mathbf{p}_u^T \mathbf{q}_i + \alpha_u + \beta_i) 
\right]  
+ 
\left[ 
    \frac{\rho}{n}\sum_{u=1}^n(\|\mathbf{p}_u\|_2^2 + \alpha_u^2) 
    + \frac{1-\rho}{m}\sum_{i=1}^m(\|\mathbf{q}_i\|_2^2 + \beta_i^2) 
\right]
$$

$$
\ \text{ s.t. } \quad \  \ 
\mathbf{A}_{\text{user}} \begin{pmatrix} \alpha_u \\ \mathbf{p}_u \end{pmatrix} + \mathbf{b}_{\text{user}} \geq \mathbf{0},\ u = 1,\dots,n
\quad \text{and} \quad
\mathbf{A}_{\text{item}} \begin{pmatrix} \beta_i \\ \mathbf{q}_i \end{pmatrix} + \mathbf{b}_{\text{item}} \geq \mathbf{0},\ i = 1,\dots,m
$$

where

- $\phi(\cdot , \cdot)$ 
  is a convex piecewise linear-quadratic loss function
  
- **A**<sub>user</sub> is a *d* × (*k*+1) matrix and **b**<sub>user</sub> is a *d*-dimensional vector representing *d* linear constraints to user-side parameters.

- **A**<sub>item</sub> is a *d* × (*k*+1) matrix and **b**<sub>item</sub> is a *d*-dimensional vector representing *d* linear constraints to item-side parameters.
  
- $\Omega$
  is a user-item collection that records all training data

- $n$ is number of rows in target matrix, $m$ is number of columns in target matrix

- $k$ is length of latent factors (rank of MF) 

- $C$ is regularization parameter, $\rho$ balances regularization strength between user and item

- $\mathbf{p}_u$ and $\alpha_u$
  are latent vector and individual bias of u-th row. Specifically, $\mathbf{p}_u$ is the u-th row of $\mathbf{P}$, and $\alpha_u$ is the u-th element of 𝝰
  
- $\mathbf{q}_i$ and $\beta_i$
  are latent vector and individual bias of i-th column. Specifically, $\mathbf{q}_i$ is the i-th row of $\mathbf{Q}$, and $\beta_i$ is the i-th element of 𝝱

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
