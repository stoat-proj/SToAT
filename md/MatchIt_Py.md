# Python interface for MatchIt

## Introduction and Background

In the causal inference field, accurately estimating the treatment effect from observational data is a focus of study. 
In randomized experiments, the treatment effect can be estimated by taking the average difference between the treatment and control groups, 
as the two groups are considered homogeneous and comparable. 
However, in observational studies, subjects in treatment and control groups often differ due to confounding factors, and the observed differences in outcomes may not reflect the true treatment effect [^Benedetto2018]. 
Therefore, researchers aim to estimate the counterfactual outcome for each subject, i.e., the outcome that would have been observed if the subject had been in the opposite treatment group.

A causal effect of a subject $i$ can be defined as the difference
between an observed outcome $Y_{i1}$, when subject $i$ is treated, and its counterfactual $Y_{i0}$, when subject $i$ is untreated. Here we denote covariates of subject $i$ as $X_i$, and the treatment indicator as $T_i$. The treatment effect can then be estimated as 

$$
\tau = \hat{f_1}(X, T_i=1) - \hat{f_0}(X, T_i=0)
$$

where $f_1$ and $f_0$ are functions estimating the outcome $Y$. 

Matching is a classic and widely used method to estimate the treatment effect in observational data. 
It can be seen as estimating the counterfactual outcome of a subject in the treatment (control) group with the factual outcome of a similar subject in the control (treatment) group. 

## Motivation

There have been many useful matching algorithms and corresponding software packages, [MatchIt](https://kosukeimai.github.io/MatchIt/index.html) [^matchit] is an authoritative and widely used R package among them. In this project,
we want to make a Python interface for MatchIt, making it convenient for Python users to conduct causal inference related work in Python.

### Why do we need the Python version

Many Matching algorithms have excellent R version Packages but no Python version because of the dominance of R language in the Statistics field, MatchIt is an example.
However, the increasing prominence of deep learning-related methods in the domain of causal inference, coupled with the computational efficiency of Python, 
has resulted in the emergence of advanced treatment effect estimation packages in Python, such as EconML [^econml], CausalML [^causalml], etc. 
Given that matching can serve as a crucial preprocessing step for treatment effect estimation, it is imperative to have a Python version of the excellent matching algorithms that are currently available primarily in R.

### Why MatchIt

We choose MatchIt instead of other Matching packages in R because of the following advantages of it:

- Comprehensive: MatchIt implements a wide range of sophisticated matching methods, including Coarsened Exact Matching [^iacus2012], Propensity Score Matching [^rubin1976], etc.
  It also provides different Balance Checking and Treatment Effect Estimation functions, then users can finish the whole Matching procedure in one software.

- Open Source: MatchIt has an official [GitHub repository](https://kosukeimai.github.io/MatchIt/), which makes it feasible for us to make a Python Interface.
  We declare and acknowledge the valuable contributions of the original authors [^matchit].

- Authoritative: MatchIt implements the suggestions of Ho, Imai, King, and Stuart [^ho2007] 
  for improving parametric statistical models by preprocessing data with nonparametric matching methods [^matchit],
  and it was developed by the original authors of this paper. Since launched in 2011, MatchIt has been widely used in the Causal Inference field.

## Aims

- Build a practical and reliable Python Interface for MatchIt.
- Make example notebooks for it, using simulated and real datasets.
- Write concise documentation for it, making it user-friendly.

## Timeline

- Get familiar with the structures and functions of MatchIt, and design its structure in Python. [~ 0.5 month]
- Implement the Python Interface with [Pybind11] (https://github.com/pybind/pybind11), an open-source Python package that exposes C++ types in Python. [~ 1 month]
- Conduct tests and make example notebooks. [~ 0.5 month]
- Write documentation. [~ 0.5 month]



## References

[^Benedetto2018]: Benedetto, U., Head, S. J., Angelini, G. D., & Blackstone, E. H. (2018). Statistical primer: Propensity score matching and its alternatives. European Journal of Cardio-Thoracic Surgery, 53(6), 1112-1117. 

[^causalml]: Huigang Chen, Totte Harinen, Jeong-Yoon Lee, Mike Yung, and Zhenyu Zhao. Causalml: Python package for causal machine learning, 2020.

[^matchit]: Ho, D., Imai, K., King, G., & Stuart, E. A. (2011). MatchIt: Nonparametric Preprocessing for Parametric Causal Inference. Journal of Statistical Software, 42(8), 1–28. https://doi.org/10.18637/jss.v042.i08

[^ho2007]: Ho D, Imai K, King G, Stuart E (2007). “Matching as Nonparametric Preprocessing for Reducing Model Dependence in Parametric Causal Inference.” Political Analysis, 15(3), 199–236.

[^iacus2012]: Iacus, S. M., King, G., & Porro, G. (2012). Causal Inference Without Balance Checking: Coarsened Exact Matching. Political Analysis, 20(1), 1–24. 

[^econml]: Maggie Hei Greg Lewis Paul Oka Miruna Oprescu Vasilis Syrgkanis Keith Battocchi, Eleanor Dillon. EconML: A Python Package for ML-Based Heterogeneous Treatment Effects Estimation. https://github.com/py-why/EconML, 2019. Version 0.15.1.

[^rubin1976]: Rubin, D. B., & Thomas, N. (1996). Matching using estimated propensity scores: relating theory to practice. Biometrics, 249-264.







