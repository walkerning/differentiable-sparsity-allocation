## DSA: More Efficient Budgeted Pruning via Differentiable Sparsity Allocation

```
@article{ning2020dsa,
  title={DSA: More Efficient Budgeted Pruning via Differentiable Sparsity Allocation},
  author={Ning, Xuefei and Zhao, Tianchen and Li, Wenshuo and Lei, Peng and Wang, Yu and Yang, Huazhong},
  journal={arXiv preprint arXiv:2004.02164},
  year={2020}
}
```

Run test:

```
bash test.sh
```

* `models/`: all models
* `models/op.py`: MaskedConvBNReLU with the differentiable pruning process
* `graph_utils.py`: topological grouping
* `ckpt/`: download checkpoints from [this url](https://drive.google.com/drive/u/1/folders/1xp-qQlOFDsNOZbhedCnPYIkJmavds9Nv), check the path in `test.sh`.


~Since the checkpoints of resnet-18, resnet-50, vgg-16 models on CIFAR-10 and ImageNet are large, we cannot include them in the supplemetary material.~

To also test the checkpoints on imagenet, specify the imagenet dataset path via the `IMAGENET_PATH` env. There should be two subdirs under the path: `train`, `val`.

```
IMAGENET_PATH=<imgnet_path> bash test.sh
```
