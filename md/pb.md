# Poisson Binomial Distribution for PyTorch

## Background

The Poisson Binomial Distribution (PBD) is a generalization of the binomial distribution, representing the sum of $n$ independent but not necessarily identically distributed Bernoulli trials. Specifically, let $X_1, X_2, \ldots, X_n$ be independent Bernoulli random variables where $X_i \sim \text{Bernoulli}(p_i)$. The random variable $Y = \sum_{i=1}^n X_i$ follows a Poisson Binomial distribution with success probability vector $\mathbf{p} = (p_1, p_2, \ldots, p_n)$.

The probability mass function (PMF) of the Poisson Binomial distribution is given by:

$$
P(Y = k) = \sum_{A \in F_k} \prod_{i \in A} p_i \prod_{j \in A^c} (1-p_j),
$$

where $F_k$ is the set of all subsets of $k$ integers that can be selected from $\{1,2,\ldots,n\}$. The direct computation of this PMF has exponential complexity $O(2^n)$, making it computationally prohibitive for large $n$.

Several efficient algorithms have been developed for computing the PBD, including:
- **Discrete Fourier Transform (DFT) method**: Reduces complexity to $O(n \log n)$ using FFT
- **Recursive methods**: Dynamic programming approaches with $O(n^2)$ complexity
- **Normal approximation**: Using refined normal approximations with skewness correction
- **Refined Normal Approximation (RNA)**: Incorporating higher-order moments (mean, variance, skewness) for improved accuracy

## Motivation

The Poisson Binomial distribution has wide applications in statistics, machine learning, and data science:

- **Hypothesis testing**: Used in multiple testing and false discovery rate control
- **Statistical quality control**: Modeling acceptance sampling with varying defect rates
- **Bioinformatics**: Analyzing sequencing data with position-dependent error rates
- **Machine learning**: Uncertainty quantification in ensemble models with heterogeneous base learners
- **Network analysis**: Modeling degree distributions in random graphs with heterogeneous connection probabilities

Despite its importance, PyTorch's `torch.distributions` module currently lacks support for the Poisson Binomial distribution. Adding this distribution would:

1. **Fill a critical gap**: Provide practitioners with a well-integrated tool for modeling heterogeneous Bernoulli processes
2. **Enable GPU acceleration**: Leverage PyTorch's computational backend for efficient batch processing
3. **Support probabilistic programming**: Integrate seamlessly with PyTorch-based probabilistic frameworks (Pyro, GPyTorch)
4. **Maintain consistency**: Follow PyTorch's design patterns and API conventions for distributions

### Why PyTorch?

PyTorch has become the de facto standard for deep learning and probabilistic modeling due to:
- Native automatic differentiation support
- Efficient tensor operations with GPU acceleration
- A well-designed `distributions` module with consistent APIs
- Strong ecosystem integration (Pyro, Botorch, etc.)

Adding PBD to `torch.distributions` would make it accessible to the large PyTorch user base and enable seamless integration with existing workflows.

## Aims and Objectives

This project aims to implement a production-ready Poisson Binomial distribution for PyTorch with the following objectives:

### 1. Core Implementation (Month 1-1.5)

- **Design and implement `PoissonBinomial` class** that strictly adheres to the `torch.distributions.Distribution` interface
  - Inherit from `Distribution` base class
  - Implement required methods: `log_prob()`, `sample()`, `cdf()`, `icdf()` (inverse CDF)
  - Support batch operations and broadcasting following PyTorch conventions
  - Proper parameter validation and constraint handling (probabilities in [0,1])

- **Implement exact PMF computation algorithms**
  - DFT-based method using FFT for $O(n \log n)$ complexity
  - Recursive dynamic programming method as fallback
  - Automatic algorithm selection based on problem size and numerical stability

- **Testing and validation**
  - Unit tests for all distribution methods
  - Numerical accuracy tests against reference implementations
  - Edge case handling (all probabilities equal, extreme probabilities)
  - Gradient correctness tests for differentiable operations

### 2. Approximation Methods (Month 1.5-2.5)

- **Refined Normal Approximation (RNA)**
  - Implement moment-based approximation using population mean, variance, and skewness
  - Cornish-Fisher expansion for improved quantile estimation
  - Automatic switching between exact and approximate methods based on accuracy requirements

- **Statistical properties computation**
  - **Mean**, **Variance**, **Skewness** population computation
  - **Quantile functions**: Both exact (via CDF inversion) and empirical (via Newton's method)

- **Other approximation methods**
  - Simple normal approximation with continuity correction
  - Saddlepoint approximation for tail probabilities (if time permits)
  - Performance benchmarking and accuracy analysis for different approximations

### 3. Optimization and Documentation (Month 2.5-3)

- **Performance optimization**
  - Vectorized operations for batch processing
  - GPU kernel optimization for critical operations
  - Memory-efficient implementations for large $n$
  - Caching strategies for repeated computations

- **Comprehensive documentation**
  - API documentation with detailed docstrings
  - Mathematical background and algorithm descriptions
  - Usage examples and tutorials
  - Performance guidelines and best practices

- **Integration and benchmarking**
  - Benchmark against existing implementations (scipy, numpy)
  - Create comprehensive test suite
  - Prepare contribution for potential PyTorch upstream integration

## Timeline

### Month 1: Core Implementation
- Week 1-2: Design `PoissonBinomial` class structure and implement basic distribution interface
- Week 3-4: Implement DFT-based PMF computation and sampling methods
- Complete: Basic functional implementation with unit tests

### Month 2: Approximations and Statistical Properties
- Week 5-6: Implement refined normal approximation with moment calculations (mean, variance, skewness)
- Week 7-8: Implement quantile functions and CDF methods, both exact and approximate
- Complete: Full-featured distribution with multiple computation backends

### Month 3: Optimization and Documentation
- Week 9-10: Performance optimization, GPU acceleration, and batch operation tuning
- Week 11: Comprehensive documentation, examples, and tutorials
- Week 12: Final testing, benchmarking, and integration preparation
- Complete: Production-ready implementation with complete documentation

## Key Features

1. **Strict PyTorch compliance**: Full adherence to `torch.distributions` design patterns and API conventions

2. **Multiple computation backends**: Automatic selection between exact (DFT, recursive) and approximate (RNA, normal) methods

3. **Complete statistical analysis**: Efficient computation of mean, variance, skewness, and quantiles

4. **GPU acceleration**: Optimized tensor operations for efficient batch processing on CUDA devices

5. **Numerical stability**: Robust handling of extreme probabilities and large sample sizes

6. **Comprehensive testing**: Extensive test coverage including numerical accuracy, gradient correctness, and edge cases

7. **Production-ready documentation**: Clear API docs, mathematical background, usage examples, and performance guidelines

## Expected Deliverables

1. **Source code**: Fully functional `PoissonBinomial` distribution class with all required methods
2. **Test suite**: Comprehensive unit tests and integration tests
3. **Documentation**: API reference, mathematical background, and usage tutorials
4. **Benchmarks**: Performance comparison with existing implementations
5. **Example notebooks**: Jupyter notebooks demonstrating various use cases
6. **Technical report**: Summary of implementation choices, algorithm selection criteria, and accuracy analysis

## Success Criteria

- All methods of `torch.distributions.Distribution` properly implemented
- Numerical accuracy within acceptable tolerance (relative error < 1e-6 for exact methods)
- Performance competitive with or better than existing implementations
- Successful integration with PyTorch's autograd system
- Complete documentation and test coverage (>90%)
- Passes PyTorch's code quality and style guidelines

---

- **Contributors**: [To be assigned]
- **Mentors**: [Ben Dai](https://www.bendai.org/)
- **Time Period**: 3 Months
- **Languages**: Python
- **Framework**: PyTorch

## References

[^hong2013]: Hong, Y. (2013). On computing the distribution function for the Poisson binomial distribution. *Computational Statistics & Data Analysis*.

[^chen1997]: Chen, S. X., & Liu, J. S. (1997). Statistical applications of the Poisson-binomial and conditional Bernoulli distributions.

[^fernandez2010]: Fernández, M., & Williams, S. (2010). Closed-form expression for the Poisson-binomial probability density function. *IEEE Transactions on Aerospace and Electronic Systems*.

[^volkova2023]: Volkova, K. (2005). A refinement of the normal approximation for the Poisson binomial distribution. *International Journal of Mathematics and Mathematical Sciences*.