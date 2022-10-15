# cofar
**Official implementation of the COFAR: Commonsense and Factual Reasoning in Image Search (AACL-IJCNLP 2022 Paper)**

[project page](https://vl2g.github.io/projects/cofar/) | [paper](https://vl2g.github.io/)

## Requirements
* Use **python >= 3.8.5**. Conda recommended : [https://docs.anaconda.com/anaconda/install/linux/](https://docs.anaconda.com/anaconda/install/linux/)

* Use **pytorch 1.9.0 CUDA 11.1**

**To setup environment**
```
conda env create -n kmmt --file kmmt.yml
conda activate kmmt
```

# Data

Coming soon!


# Training
MS-COCO - pretraining checkpoint [path coming soon](https://drive.google.com/file/d/1Yep6zc652isEk-e4_IcoUYPQr1bzeSet/view?usp=sharing)

Respective config files are in ```config/``` folder and are automatically loaded.

## MLM finetuning

```
python main.py --do_train --mode mlm
```

## ITM finetuning

```
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node 2 --nnodes 1 --node_rank 0 main.py --do_train --mode itm
```

## Evaluation

coming soon!

