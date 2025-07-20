# Fast Path Solution for Ridge Composite Quantile Regression (CQR)

## Introduction

Ridge Composite Quantile Regression (CQR) is a robust statistical method that extends traditional quantile regression by jointly estimating multiple quantile levels. This approach provides a more informative view of the conditional distribution of the response variable and is particularly useful in the presence of heteroscedasticity, asymmetric errors, or outliers. This project develops a fast and reliable path solution for Ridge-regularized CQR, featuring warm-start optimization, efficient tracking across regularization parameters, and interpretable visualization of model dynamics.

## Problem Definition

The regularized composite quantile regression solves the following optimization problem:

$$
\min_{\beta \in \mathbb{R}^{d}, \{\alpha_k\}_{k=1}^K} \sum_{k=1}^K \sum_{i=1}^n \rho_{\tau_k} ( y_i - x^\intercal_i \beta - \alpha_k ) + \frac{1}{2C} \| \beta \|^2,
$$

where:

- $\alpha_k$ is the intercept associated with the $k$-th quantile.
- $\rho_{\tau}(u) = u \cdot (\tau - \mathbf{1}(u < 0))$ is the check loss function.
- $\beta$ is a shared slope across all quantiles.
- $C$ is the inverse regularization parameter controlling the trade-off between fit and complexity.

## Motivation

Regularization path analysis is an important tool in statistical modeling. It allows users to understand how model parameters evolve under varying levels of penalization. For CQR models, this is especially valuable for analyzing how predictors affect different parts of the conditional distribution. This project introduces a warm-started path solver that is both efficient and compatible with high-dimensional settings.

## Aims

1. **Algorithm Development**  
   Implement a modular and efficient solver for Ridge Composite Quantile Regression that supports path computation over a grid of regularization values.

2. **Warm Start Techniques**  
   Reuse solutions from previous regularization steps to accelerate convergence and reduce total computational cost.

3. **Shared-Slope Structure**  
   Support a modeling structure where all quantiles share a common coefficient vector $\beta$, with quantile-specific intercepts $\alpha_k$ to capture vertical shifts.

4. **Unified Visualization**  
   Automatically detect whether to plot coefficient or intercept paths, depending on the model structure. Provide interpretable plots over $\log_{10}(C)$.

5. **Robust and Fail-Safe Behavior**  
   Tolerate partial convergence failures by skipping or recording `NaN` values when fitting fails, and continue across the entire path.

6. **Testing and Validation**  
   Validate the implementation on simulated datasets with controlled noise and dimensionality. Assess robustness, stability, and runtime.

7. **Documentation and Examples**  
   Provide clean example scripts and documentation to demonstrate usage and integration into broader workflows.

- **Contributors**: [Youtong Li](https://github.com/Leona-LYT)  
- **Mentors**: [Ben Dai](https://www.bendai.org/)  
- **Time Period**: **3 Months**  
- **Languages**: Python and C++  

## References

[^rehline]: Dai, B., & Qiu, Y. (2024). ReHLine: Regularized Composite ReLU-ReHU Loss Minimization with Linear Computation and Linear Convergence. *Advances in Neural Information Processing Systems*, *36*.
