# HiTIN: Hierarchy-aware Tree Isomorphism Network for Hierarchical Text Classification

Official implementation for ACL 2023 accepted paper "HiTIN: Hierarchy-aware Tree Isomorphism Network for Hierarchical Text Classification" . [[arXiv](https://arxiv.org/abs/2305.15182)][[pdf](https://arxiv.org/pdf/2305.15182.pdf)][[bilibili](https://www.bilibili.com/video/BV1vL411i7uY/?share_source=copy_web&vd_source=a9cc6ff9a8cf3c92bf2375da5b56a007)]

## Requirements

**It's hard to reproduce the results without the same devices and environment.** Although our code is highly compatible with mulitiple python environments, we strongly recommend that you create a new environment according to our settings.

- Python == 3.7.13
- numpy == 1.21.5
- PyTorch == 1.11.0
- scikit-learn == 1.0.2
- transformers == 4.19.2
- numba == 0.56.2
- glove.6B.300d

Note: to set up this environment use poetry that already lies in the repo.

## Data preparation

Please make sure to organize the data in the following format:

```
{
    "label": ["Computer", "MachineLearning", "DeepLearning", "Neuro", "ComputationalNeuro"],
    "token": ["I", "love", "deep", "learning"]
}

```

### Steps to organize the data:

0. Download and extract `all.csv` file from `all.7z` into `./cpv` directory. Download and put `vocab.txt` into the same directory.
   https://drive.google.com/drive/folders/14BW6nZF0Hao2EEpVSm41i7vKK6kMG-Mj?usp=drive_link

1. To download xls file of cpv codes and save it as csv, run:
   ```shell
   python data_preprocessing/cpv_download.py
   ```
2. To create a hierarchy out of csv file, run:
   ```shell
   python data_preprocessing/cpv_hierarchy.py
   ```
3. To create taxonomy based on hierarchy, run:
   ```shell
   python data_preprocessing/cpv_taxonomy.py
   ```
4. To prepare data for training, run:
   ```shell
   python data_preprocessing/cpv_prepare_data.py
   ```
5. To count the prior probabilities between parent and child labels, run:
   ```shell
   PYTHONPATH="$(pwd):$PYTHONPATH" python helper/hierarchy_tree_statistic.py config/cpv.json
   ```
6. To start training, run:
   ```shell
   python train.py -cfg config/cpv.json
   ```

## Train

Note: We have configured params that are specified in the paper's implementation details, so there is no need to change anything without acute necessity.

From the paper's authors:
The default parameters are not the best performing-hyper-parameters used to reproduce our results in the paper. Hyper-parameters need to be specified through the commandline arguments. Please refer to our paper for the details of how we set the hyper-parameters.

To learn hyperparameters to be specified, please see:

```
python train.py [-h] -cfg CONFIG_FILE [-lr LEARNING_RATE]
                [-l2 L2RATE] [-p] [-k TREE_DEPTH] [-lm NUM_MLP_LAYERS]
                [-tp {root,sum,avg,max}]
                [--log_dir LOG_DIR] [--ckpt_dir CKPT_DIR]
                [--begin_time BEGIN_TIME]

optional arguments:
  -h, --help            show this help message and exit
  -cfg CONFIG_FILE, --config_file CONFIG_FILE

  -lr LEARNING_RATE, --learning_rate LEARNING_RATE
                        learning rate (default: 0.001)
  -l2 L2RATE, --l2rate L2RATE
                        L2 penalty lambda (default: 0.01)
  -lm NUM_MLP_LAYERS, --num_mlp_layers NUM_MLP_LAYERS
                        Number of layers for MLP EXCLUDING the input one
                        (default: 2). 1 means linear model.
  -tp {root,sum,avg,max}, --tree_pooling_type {root,sum,avg,max}
                        Pool strategy for the whole tree in Eq.11. Could be
                        chosen from {root, sum, avg, max}.

  -p, --load_pretrained
  -k TREE_DEPTH, --tree_depth TREE_DEPTH
                        The depth of coding tree to be constructed by CIRCA
                        (default: 2)
  --log_dir LOG_DIR     Path to save log files (default: log).
  --ckpt_dir CKPT_DIR   Path to save checkpoints (default: ckpt).
  --begin_time BEGIN_TIME
                        The beginning time of a run, which prefixes the name
                        of log files.
```

## Evaluation Metrics

The experimental results are measured with `Micro-F1` and `Macro-F1`.

`Micro-F1` is the harmonic mean of the overall precision and recall of all the test instances, while
`Macro-F1` is the average F1-score of each category.

Thus, Micro-F1 reflects the performance on more frequent labels, while Macro-F1 treats labels equally.

## Citation

```
@inproceedings{zhu-etal-2023-hitin,
    title = "{H}i{TIN}: Hierarchy-aware Tree Isomorphism Network for Hierarchical Text Classification",
    author = "Zhu, He  and
      Zhang, Chong  and
      Huang, Junjie  and
      Wu, Junran  and
      Xu, Ke",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.acl-long.432",
    pages = "7809--7821",
}
```
