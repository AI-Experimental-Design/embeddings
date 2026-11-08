# Embeddings

*Week 7 of [AI-Integrated Experimental Design](https://github.com/AI-Experimental-Design).*

Last week's networks learned their features from labeled data. Labels are
expensive, and unlabeled sequence is cheap. This week asks what a model can
learn from sequence alone, before it sees any labels. We train a small DNA
model by hiding bases and asking it to fill them in, look at the
representations it learns, and use it to score variants with no labels at all.
The same model shows the main limit of the approach. It learns what is common,
and what is common is not always what is functional, so a learned
representation still has to be tested against the biology.

- [Embeddings and self-supervised pretraining](./embeddings.md)
    - one-hot encoding versus learned embeddings
    - masked prediction
    - softmax and cross-entropy for four classes
    - contextual embeddings and PCA
- [Zero-shot variant scoring](./zero-shot.md)
    - log-likelihood ratios
    - scoring every variant without labels
    - unexpected is not the same as damaging
    - choosing what a benchmark counts

## Reading

### Animal communication

- Sainburg, Thielk and Gentner 2020. Finding, visualizing, and quantifying latent structure across diverse animal vocal repertoires. *PLoS Computational Biology* 16(10):e1008228. Learned embedding spaces across 29 species.
- Hagiwara 2023. AVES: Animal Vocalization Encoder based on Self-Supervision. *ICASSP 2023*. Self-supervised pretraining for bioacoustics.

### Genetics

- Rives et al. 2021. Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences. *PNAS* 118:e2016239118. Structure appearing in a learned embedding.
- Meier et al. 2021. Language models enable zero-shot prediction of the effects of mutations on protein function. *NeurIPS 2021*. ESM-1v, variant effects with no labels.
- Benegas, Batra and Song 2023. DNA language models are powerful predictors of genome-wide variant effects. *PNAS* 120:e2311219120. GPN, self-supervised pretraining on plant genomes.
