# CG-MuAlign
A reference implementation for ["Collective Multi-type Entity Alignment Between Knowledge
Graphs"](https://gentlezhu.github.io/files/CollectiveLinkage.pdf), published in WWW 2020.

If you find our paper useful, please consider cite the following paper
```
@inproceedings{10.1145/3366423.3380289,
author = {Zhu, Qi and Wei, Hao and Sisman, Bunyamin and Zheng, Da and Faloutsos, Christos and Dong, Xin Luna and Han, Jiawei},
title = {Collective Multi-Type Entity Alignment Between Knowledge Graphs},
year = {2020},
url = {https://doi.org/10.1145/3366423.3380289},
doi = {10.1145/3366423.3380289},
booktitle = {Proceedings of The Web Conference 2020}
}
```
## Data
Unfortunately, the original data used is not public available. But this reference implementation could be easily adopt to structured data: knowledge graph, knowledge base and __etc.__ See examples below for details. 

We are collecting more public available knowledge graphs, stay tuned!

## Requirements
```
pip install -r requirements.txt
```

## Run the code

### Prepare the pre-trained fastText embedding
Most of the attributes in a knowledge graph is text. 
Obtain your binarized pre-trained word embeddings $PATH at [fastText](https://fasttext.cc/docs/en/english-vectors.html). I'm using [enwiki9.bin](https://fasttext.cc/docs/en/unsupervised-tutorial.html)

python main.py --gpu=0 --pretrain-path=$PATH

