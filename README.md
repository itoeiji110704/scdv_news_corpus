![](https://img.shields.io/static/v1?label=python&message=3.8.16&color=blue)
![](https://img.shields.io/static/v1?label=last%20updated&message=december%202022&color=lightgray)

# scdv_news_corpus

This repo is an experiement SCDV (Sparse Composite Document Vectors).  
SCDV is a method to calculate document vector proposed by Microsoft Research & IIT Kanpur.  

It implemented SCDV and experimented it to news corpus from scikit-learn dataset.  
See `src/document_vectors.py`

Compared with SCDV vs other well known methods e.g. BoW, TF-IDF, averaged Word2Vec and Doc2Vec.  
Comparison methods are  
(i) Plots of 2 dim t-SNE of these document vectors  
(ii) Accuracies of a document classification using these document vectors  
See `notebooks/scdv_vs_other_methods_of_document_vector.ipynb`

# Reference

- Paper of SCDV: https://arxiv.org/abs/1612.06778
