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
Images: [link](https://drive.google.com/file/d/1pzQdDhCCLWn-L5VMxBb2s4rY7M7mQkdf/view?usp=sharing) 

Image format: [Readme](https://github.com/vl2g/vl2g.github.io/blob/master/projects/cofar/docs/dataset_README.md)


Training and testing data can be downloaded from the "Dataset Downloads" section in this [page](https://vl2g.github.io/projects/cofar/)

Also, both oracle and wikified knowledge bases for all categories can be downloaded from the same link above.

## Feature extraction
Image Feature extraction: [Script](https://gist.github.com/revantteotia/7a992edff725a08819fa21d87d8d2598)

# Training
MS-COCO - pretraining checkpoint can be downloaded from [here](https://drive.google.com/file/d/1Yep6zc652isEk-e4_IcoUYPQr1bzeSet/view?usp=sharing).

Place the downloaded ```kmmt_pretrain_checkpoint.pt``` in ```working_checkpoints``` folder.

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

# License
Detectron2 is released under the [MIT license](https://github.com/vl2g/cofar/blob/main/LICENSE).
