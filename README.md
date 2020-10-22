# text-semantic-matching
The repository contains several Pytorch model implementations for text semantic matching, and the implemented models are as followed,
- ESIM [Enhanced LSTM for Natural Language Inference](https://arxiv.org/pdf/1609.06038.pdf)
- CAFE [Compare, Compress and Propagate: Enhancing Neural Architectures with Alignment Factorization for Natural Language Inference](https://arxiv.org/abs/1801.00102)
- RE2 [Simple and Effective Text Matching with Richer Alignment Features](https://www.aclweb.org/anthology/P19-1465/)

# Data Preparation
```
cd setup
bash setup_snli.sh
```

# Run Model
```
python train.py --model="esim" # run ESIM model
or
python train.py --model="cafe" # run CAFE model
or
python train.py --model="re2" # run RE2 model
```

# Performance
We report the performance of ESIM, CAFE, RE2 in the SNLI dataset
|  Models |  Accuracy |  
|---|---|
| ESIM  |  88.325 |   
| CAFE  |  87.320 |  
| RE2  |   88.458 |  
