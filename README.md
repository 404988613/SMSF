# Linguistic Steganalysis Model

This repo contains the pytorch implementation of different linguistic steganalysis models. 
## Dataset preprocess
```
config and run stega_data_preprocess.py
```

## Training the model
Set the parameters in config.py
```
python train.py
```

## Include

### Non-Transformer-based

- [TS-CSW: text steganalysis and hidden capacity estimation based on convolutional sliding windows (TS-CSW)](https://link.springer.com/article/10.1007/s11042-020-08716-w)
- [TS-RNN: Text Steganalysis Based on Recurrent Neural Networks (TS-BiRNN)](https://ieeexplore.ieee.org/abstract/document/8727932)
- [Linguistic Steganalysis via Densely Connected LSTM with Feature Pyramid (BiLISTM-Dense)](https://dl.acm.org/doi/abs/10.1145/3369412.3395067)
- [A Hybrid R-BILSTM-C Neural Network Based Text Steganalysis(R-BiLSTM-C)](https://ieeexplore.ieee.org/abstract/document/8903243)

------

### Transformer-based

- [SeSy: Linguistic Steganalysis Framework Integrating Semantic and Syntactic Features (Sesy)](https://ieeexplore.ieee.org/abstract/document/9591452)
- [High-Performance Linguistic Steganalysis, Capacity Estimation and Steganographic Positioning (BERT-LSTM-ATT)](https://link.springer.com/chapter/10.1007%2F978-3-030-69449-4_7)
- [SCL-Stega: Exploring Advanced Objective in Linguistic Steganalysis using Contrastive Learning](https://dl.acm.org/doi/abs/10.1145/3577163.3595111)
- TDA-FSMS
