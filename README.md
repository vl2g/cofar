# COFAR

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/cofar-commonsense-and-factual-reasoning-in/image-retrieval-on-cofar)](https://paperswithcode.com/sota/image-retrieval-on-cofar?p=cofar-commonsense-and-factual-reasoning-in)

**Official implementation of the COFAR: Commonsense and Factual Reasoning in Image Search (AACL-IJCNLP 2022 Paper)**

[paper](https://vl2g.github.io/projects/cofar/docs/COFAR-AACL2022.pdf) | [project page](https://vl2g.github.io/projects/cofar/)

## Requirements
* Use **python >= 3.8.5**. Conda recommended : [https://docs.anaconda.com/anaconda/install/linux/](https://docs.anaconda.com/anaconda/install/linux/)

* Use **pytorch 1.9.0; CUDA 11.1**

**To setup environment**
```
conda env create -n kmmt --file kmmt.yml
conda activate kmmt
```

# Data
Images: [link](https://drive.google.com/file/d/1pzQdDhCCLWn-L5VMxBb2s4rY7M7mQkdf/view?usp=sharing).

Image format: [Readme](https://github.com/vl2g/vl2g.github.io/blob/master/projects/cofar/docs/dataset_README.md).


Training and testing data can be downloaded from the "Dataset Downloads" section in this [page](https://vl2g.github.io/projects/cofar/).

Also, both oracle and wikified knowledge bases for all categories can be downloaded from the same link above.

## Feature extraction
Image Feature extraction: [Script](https://gist.github.com/revantteotia/7a992edff725a08819fa21d87d8d2598).

Create folder ```train_obj_frcn_features/``` inside ```data/cofar_{category}/``` folder for corresponding categories and copy image features to this folder.

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

Download our cofar finetuned checkpoint from [here](https://drive.google.com/file/d/1sWoqATnTyz0-SuAMT4BAmNyhvg5feuF5/view?usp=sharing).

Copy the downloaded ```cofar_itm_final_checkpoint.pt``` to ```working_checkpoints/``` folder.

```
python cofar_eval.py --category brand
```

Other settings can be changed from ```config/test_config.yaml```

# License
This code and data are released under the [MIT license](https://github.com/vl2g/cofar/blob/main/LICENSE).

# Cite
If you find this data/code/paper useful for your research, please consider citing.

```
@inproceedings{cofar2022,
  author    = "Gatti, Prajwal and 
              Penamakuri, Abhirama Subramanyam and
              Teotia, Revant and
              Mishra, Anand and
              Sengupta, Shubhashis and
              Ramnani, Roshni",
  title     = "COFAR: Commonsense and Factual Reasoning in Image Search",
  booktitle = "AACL-IJCNLP",
  year      = "2022",
}
```
