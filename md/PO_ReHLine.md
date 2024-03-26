

# PO-ReHLine: Portfolio Optimization via ReHLine

## Introduction

Portfolio optimization with transaction costs is a critical area within financial mathematics and investment management. In traditional portfolio optimization, the objective is to construct a portfolio that maximizes expected return for a given level of risk. However, incorporating transaction costs into the optimization process adds complexity, as it requires balancing the benefits of rebalancing the portfolio against the costs incurred from trading. Transaction costs include brokerage fees, taxes, and market impact costs, among others. Therefore, the goal of **portfolio optimization with linear transaction costs** is to find an optimal balance between maximizing returns and minimizing trading costs.

## Math Formulation

**Parameters:**

- $n$ as the number of assets in the portfolio.
- $\mu_i$ as the expected return of asset \( i \).
- $\sigma_{ij}$​ as the covariance between assets \( i \) and \( j \).

**Variables:**

- $w_i$​ as the weight of asset \( i \) in the portfolio, where \( i = 1, 2, ..., n \).

**Objective Function:**

The objective function for portfolio optimization with transaction costs can be formulated as[^PO1][^PO2]:

$$
\max_{\mathbf{w} \in \mathbb{R}^n} \frac{\alpha}{2} \mathbf{w}^T \mathbf{\Sigma} \mathbf{w} - \mathbf{\mu}^T \mathbf{w} + \sum_{i=1}^n \phi_i(w_i), \qquad \text{s.t.} \quad \mathbf{A} \mathbf{w} \leq \mathbf{b},
$$

Where the first two terms represents the Minkowski **quadratic mean-variance model** with a hyperparameter $\alpha \geq 0$​, and the last term represents the **piecewise transaction costs**. The **polyhedral constraints** ensure the feasibility of the portfolio, such as the portfolio weights sum to one and are non-negative.

The piecewise transaction costs are defined as a piecewise-linear function:
$$
\phi_i(w_i) := \{ p_{il} w_i + q_{il}, \text{ if } d_{il} \leq w_i \leq d_{i(l+1)}\}_{l=0}^{L_i+1},
$$
where $d_{i0} = - \infty$ and $d_{i{L_i + 1}} = + \infty$​.

## Motivation

We found that this formulation can be written in the form of a **ReHLine**[^rehline], enabling it to be solved with linear computational complexity. Here, it is necessary to *add a linear term* to the original ReHLine, but in fact, this does not affect the general iteration and algorithmic logic of ReHLine. We could consider the case like (or can be even more specific for PO):

$$
L_i(\mathbf{w})=\sum_{l=1}^L \text{ReLU}( u_{l} \mathbf{x}^T_i \mathbf{w}  + v_{l}) + \sum_{h=1}^H {\text{ReHU}}_ {\tau_{h}}( s_{h} \mathbf{x}^T_i \mathbf{w} + t_{h}) \tag{2} + \mathbf{r}^T \mathbf{w}
$$

## Aims

1. Derive the math formula of PO under the ReHLine formulation.
2. Following the steps of ReHLine, derive the iterative formula for PO-RehLine.
3. Revise the "rehline.cpp" code to complete the implementation of PO-ReHLine.
4. Testing, API, and documentation.
5. Benchmarking with other existing algorithms, summarize the strength and weakness of PO-ReHLine.

- Mentors: [Ben Dai](https://www.bendai.org/) and [Yixuan Qiu](https://statr.me/about/)
- Time Period: **4 Months**
- Languages: C++ and Python
- Position: RA@CUHK-STAT

## Reference

[^Rehline]: Dai, B., & Qiu, Y. (2024).  ReHLine: Regularized Composite ReLU-ReHU Loss Minimization with Linear  Computation and Linear Convergence. *Advances in Neural Information Processing Systems*, *36*.
[^PO1]: Potaptchik, M., Tunçel, L., &  Wolkowicz, H. (2008). Large scale portfolio optimization with piecewise  linear transaction costs. *Optimization Methods & Software*, *23*(6), 929-952.
[^PO2]: Lobo, M. S., Fazel, M., & Boyd, S. (2007). Portfolio optimization with linear and fixed transaction costs. *Annals of Operations Research*, *152*, 341-365.