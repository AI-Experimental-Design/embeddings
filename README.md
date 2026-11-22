# Embeddings and Self-Supervised Learning

## The question we are asking

Embeddings allow models to represent words, DNA sequences, and other data as vectors of numbers that capture relationships between them. But how do models learn these representations, and how do we know whether the relationships they capture are meaningful?

In these four modules, we explore how models learn embeddings from data without externally supplied labels. We begin with word2vec, then examine how unsupervised methods identify structure in data. Next, we train a model to predict missing DNA bases and learn contextual embeddings. Finally, we use the pretrained model to score genetic variants and examine why a model that learns predictable sequence patterns does not necessarily learn biological function.

## Modules

### [Word2vec](./word2vec.md)

- Learning embeddings by predicting surrounding words
- Training a word2vec model on synthetic sentences
- Visualizing embeddings with PCA
- Measuring relationships using cosine similarity
- Limitations of fixed embeddings

### [Unsupervised learning](./unsupervised-learning.md)

- K-means clustering
- Principal component analysis (PCA)
- Uniform Manifold Approximation and Projection (UMAP)
- Identifying structure without labels
- Interpreting clusters and patterns in high-dimensional data

### [Masked prediction and contextual embeddings](./masked-prediction.md)

- Learning from sequences using masked prediction
- Fixed versus contextual embeddings
- Convolutional neural networks for DNA sequences
- Training and evaluating a masked-base prediction model
- Examining learned contextual representations

### [Zero-shot variant scoring](./zero-shot.md)

- Scoring variants using a pretrained model
- Log-likelihood ratios and sequence predictability
- Comparing predicted scores with functional effects
- Evaluating performance using AUC
- Why unexpected variants are not necessarily damaging

## Reading

### Animal communication

- Sainburg, Thielk and Gentner (2020). [Finding, visualizing, and quantifying latent structure across diverse animal vocal repertoires](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1008228). *PLoS Computational Biology* 16(10):e1008228.
- Hagiwara (2023). [AVES: Animal Vocalization Encoder based on Self-Supervision](https://arxiv.org/abs/2210.14493). *ICASSP 2023*.

### Genetics

- Rives et al. (2021). [Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences](https://www.pnas.org/doi/10.1073/pnas.2016239118). *PNAS* 118
