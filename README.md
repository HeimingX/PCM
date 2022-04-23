# Progressive Class Semantic Matching for Semi-supervised Text Classification            

This project contains the implementation of PCM for semi-supervised text classification, as presented in our paper
    
    Progressive Class Semantic Matching for Semi-supervised Text Classification,
    Hai-Ming Xu, Lingqiao Liu and Ehsan Abbasnejad,
    *To be appeared in NAACL 2022*
    
The full paper is available at: [Open Review Link](https://openreview.net/forum?id=FitLLp-Jwa)

## News
* [2022-04-23] Repo is created. Code will come soon.

## Abstract

Semi-supervised learning is a promising way to reduce the annotation cost for text-classification. Combining with pre-trained language models (PLMs), e.g., BERT, recent semi-supervised learning methods achieved impressive performance. In this work, we further investigate the marriage between semi-supervised learning and a pre-trained language model. Unlike existing approaches that utilize PLMs only for model parameter initialization, we explore the inherent topic matching capability inside PLMs for building a more powerful semi-supervised learning approach. Specifically, we propose a joint semi-supervised learning process that can progressively build a standard $K$-way classifier and a matching network for the input text and the Class Semantic Representation (CSR). The CSR will be initialized from the given labeled sentences and progressively updated through the training process. By means of extensive experiments, we show that our method can not only bring remarkable improvement to baselines, but also overall be more stable, and achieves state-of-the-art performance in semi-supervised text classification.
