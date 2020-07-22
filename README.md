## DSA: More Efficient Budgeted Pruning via Differentiable Sparsity Allocation (ECCV 2020)
[[paper]](https://arxiv.org/abs/2004.02164) [[checkpoints]](https://drive.google.com/drive/u/1/folders/1xp-qQlOFDsNOZbhedCnPYIkJmavds9Nv)


Run test:

```
bash test.sh
```

* `models/`: all models. The codes of some basic models under this directory are copy and modified from [this repo](https://github.com/kuangliu/pytorch-cifar/commit/340751189c307d91e243df26d6d5779b7a29f781)
* `models/op.py`: MaskedConvBNReLU with the differentiable pruning process
* `graph_utils.py`: topological grouping
* `ckpt/`: download checkpoints from [this url](https://drive.google.com/drive/u/1/folders/1xp-qQlOFDsNOZbhedCnPYIkJmavds9Nv), check the path in `test.sh`.


To also test the checkpoints on imagenet, specify the imagenet dataset path via the `IMAGENET_PATH` env. There should be two subdirs under the path: `train`, `val`.

```
IMAGENET_PATH=<imgnet_path> bash test.sh
```

You can cite the paper as
```
@article{ning2020dsa,
  title={DSA: More Efficient Budgeted Pruning via Differentiable Sparsity Allocation},
  author={Ning, Xuefei and Zhao, Tianchen and Li, Wenshuo and Lei, Peng and Wang, Yu and Yang, Huazhong},
  journal={arXiv preprint arXiv:2004.02164},
  year={2020}
}
```
