# BERT_REVIEW

# BERT:Pre-training of Deep Bidirectional Transformers for Language Understanding Review

- Conference : NAACL
- Link :

[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)

- Year : 2018
- 저자 : Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova
- Github :

[https://github.com/google-research/bert](https://github.com/google-research/bert)

- 동영상 :

[https://vimeo.com/365139010](https://vimeo.com/365139010)

# BERT란?

Bidirectional Encoder Representations from Transformers. 

Unlike recent language representation models.

BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers.

As a result, the pre-trained BERT model can be fine tuned with just one additional output layer to create state-of-the art models for a wide range of tasks, such as question answering and language inference, without substanial task specific architecure modifications.

# 핵심 요약

- trasformer의 encoder network를 기반으로, self-attention을 이용하여 bidirectional하게 언어 특성을 학습합니다.

- MLM(Masked language model)과 NSP(next sentence prediction)등의 pre-training방법을 제시하였습니다.

- pre-training방법으로 feature representation을 학습한 뒤 fine-tuning만으로 down-stream task를 수행합니다.