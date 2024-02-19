# Empirical Evaluation of Dice/IoU in Segmentation

## Background

The primary aim of segmentation is to label each foreground feature/pixel of an input with a corresponding class. Specifically, for a feature vector or an image, the ground truth segmentation is $\mathbf{Y} \in \{0,1\}^d$, where $Y_j = 1$ indicates that the $j$-th feature/pixel is segmented, and $I(\mathbf{Y}) = \{ j: \mathbf{Y}_j = 1; \text{ for } j = 1, \cdots, d \}$ is the index set of the ground-truth segmentation. Correspondingly, the predicted segmentation is denoted as $\pmb{\delta} = ( \delta_1, \cdots, \delta_d )^\intercal$, where $\delta_j$ represents the predicted segmentation for the $j$-th feature, and $I(\pmb{\delta}) = \{j: \delta_j = 1; \text{ for } j = 1, \cdots, d \}$ is the index set of the segmented features.

To access the performance for a segmentation function $\pmb{\delta}$, the Dice and IoU metrics are introduced and widely used in the literature[^milletari2016v], both of which measure the overlap between the ground truth and the predicted segmentation:

$$
\text{Dice}_\gamma(\pmb{\delta}) = \mathbb{E} ( \frac{2 | I(\mathbf{Y}) \cap I(\pmb{\delta}) | + \gamma }{ | I(\mathbf{Y}) | + | I(\pmb{\delta}) | + \gamma } ) = \mathbb{E} ( \frac{2 \mathbf{Y}^\intercal \pmb{\delta} + \gamma }{ \| \mathbf{Y} \|_1 + \| \pmb{\delta} \|_1 + \gamma } ),
$$

$$
\text{IoU}_\gamma(\pmb{\delta}) = \mathbb{E} ( \frac{| I(\mathbf{Y}) \cap I({\pmb{\delta}}) | + \gamma }{ | I(\mathbf{Y}) \cup I({\pmb{\delta}}) | + \gamma } ) = \mathbb{E} ( \frac{ \mathbf{Y}^\intercal \pmb{\delta} + \gamma }{ \| \mathbf{Y} \|_1 + \| \pmb{\delta} \|_1 - \mathbf{Y}^\intercal \pmb{\delta} + \gamma } ),
$$

where $|\cdot|$ is the cardinality of a set, and $\gamma \geq 0$ is a smoothing parameter.

## Motivation

**Question.** Given a dataset with $n$ samples, and their ground-truth segmentation $(\mathbf{y}_1, \cdots, \mathbf{y}_n)$ and the predicted segmentation $(\pmb{\delta}_1, \cdots, \pmb{\delta}_n)$, how can we empirically evaluate IoU and Dice metrics?

The correct empirical version of Dice and IoU metrics should be:

```math
\widehat{\text{Dice}}_{\gamma} (\pmb{\delta}) = \frac{1}{n} \sum_{i=1}^{n} \frac{2 \mathbf{y}^\intercal_i \pmb{\delta}_i + \gamma }{ \| \mathbf{y}_i \|_1 + \| \pmb{\delta} \|_1 + \gamma } = \frac{1}{n} \sum_{i=1}^{n} \frac{2 \text{TP}_i  + \gamma }{ 2 \text{TP}_i + \text{FP}_i + \text{FN}_i + \gamma },
```

```math
\widehat{\text{IoU}}_\gamma(\pmb{\delta}) = \frac{1}{n} \sum_{i=1}^{n} \Big( \frac{ {\mathbf{y}}_i^\intercal \pmb{\delta}_i + \gamma }{ \| \mathbf{y}_i \|_1 + \| \pmb{\delta}_i \|_1 - \mathbf{y}_i^\intercal \pmb{\delta}_i + \gamma } \Big) = \frac{1}{n} \sum_{i=1}^{n} \frac{\text{TP}_i + \gamma}{ \text{TP}_i + \text{FP}_i + \text{FN}_i + \gamma },
```

where $\text{TP}_i$, $\text{FP}_i$ and $\text{FN}_i$ are *true-positive*, *false-positive* and *false-negative* defined **at the instance level**. In general, the empirical Dice and IoU metrics are not equal to the evaluation criteria used in some literature:

```math
\widehat{\text{Dice}}_\gamma(\mathbf{\delta}) \neq \overline{\text{Dice}}_\gamma(\mathbf{\delta}) := \frac{ \frac{1}{n} {\sum}_{i=1}^{n} 2 \mathbf{y}^\intercal_i \pmb{\delta}_i   + \gamma }{ \frac{1}{n} ( {\sum}_{i=1}^{n} \| \mathbf{y}_i \|_1 + {\sum}_{i=1}^{n} \| \mathbf{\delta}_i \|_1) + \gamma } \overset{\mathbb{P}}{\longrightarrow} \frac{ \mathbb{E}\big( 2 \mathbf{Y}^\intercal \pmb{\delta} \big) + \gamma }{ \mathbb{E}\big( \| \mathbf{Y} \|_1 ) + \mathbb{E}\big( \| \pmb{\delta} \|_1 ) + \gamma },
```

```math
\widehat{\text{IoU}}_\gamma(\mathbf{\delta}) \neq \overline{\text{IoU}}_\gamma(\pmb{\delta}) := \frac{ \frac{1}{n} {\sum}_{i=1}^{n} \mathbf{y}^\intercal_i \pmb{\delta}_i  + \gamma }{ \frac{1}{n} ( {\sum}_{i=1}^{n} \| \mathbf{y}_i \|_1 + {\sum}_{i=1}^{n} \| \pmb{\delta}_i \|_1 - {\sum}_{i=1}^n \mathbf{y}_i^\intercal \pmb{\delta}_i) + \gamma } \overset{\mathbb{P}}{\longrightarrow} \frac{ \mathbb{E}( 2 \mathbf{Y}^\intercal \pmb{\delta} ) + \gamma }{ \mathbb{E}( \| \mathbf{Y} \|_1 ) + \mathbb{E}( \| \pmb{\delta} \|_1 ) - \mathbb{E}( \mathbf{Y}^\intercal \pmb{\delta} ) + \gamma }.
```

Here $\overset{\mathbb{P}}{\to}$ denotes convergence in probability following from the law of large numbers and Slutsky's theorem. $\overline{\text{Dice}}$ and $\overline{\text{IoU}}$ are called **linear-fractional approximation** of Dice/IoU in some literature. Clearly, both empirical and population evaluations of $\widehat{\text{Dice}}$ and $\widehat{\text{IoU}}$ do not match with the empirical and the population of $\overline{\text{Dice}}$ and $\overline{\text{IoU}}$. Although the empirical evaluation of $\overline{\text{Dice}}$ and $\overline{\text{IoU}}$ are widely used, it inherently discounts the effects of instances with small segmented features/pixels, leading to bias in the empirical evaluation. The issues of $\overline{\text{Dice}}$ and $\overline{\text{IoU}}$ are also indicated in some recent literature, including [^cordts2016cityscapes]. 

See more details in Appendix A of [^dai2023rankseg].

## Aim

A large number of implementation of segmentation evaluation rely heavily on $\overline{\text{Dice}}$ and $\overline{\text{IoU}}$. From a statistical perspective, it may introduce certain biases into the final segmentation evaluation. Thus, the aim of our project is to provide practical, unbiased empirical evaluations of Dice and IoU metrics.

- Mentors: [Ben Dai](https://www.bendai.org/)
- Time Period: **3 Months**
- Languages: Python
- Position: RA@CUHK-STAT | Student Helper (CUHK undergrad student only)

## Skills Required

- Proficiency in programming using Python and LaTex;
- Patience in drafting the documentation for the library.

## Related Work

N.A.

## Reference

[^milletari2016v]: Milletari, F., Navab, N., &  Ahmadi, S. A. (2016, October). V-net: Fully convolutional neural  networks for volumetric medical image segmentation. In *2016 fourth international conference on 3D vision (3DV)* (pp. 565-571). IEEE.
[^cordts2016cityscapes]: Cordts, M., Omran, M., Ramos, S.,  Rehfeld, T., Enzweiler, M., Benenson, R., ... & Schiele, B. (2016).  The cityscapes dataset for semantic urban scene understanding. In *Proceedings of the IEEE conference on computer vision and pattern recognition* (pp. 3213-3223).
[^dai2023rankseg]: Dai, B., & Li, C. (2023). RankSEG: A Consistent Ranking-based Framework for Segmentation. *Journal of Machine Learning Research*, *24*(224), 1-50.