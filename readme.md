# Pytorch implementation of Deep Compression
Pruning + finetuning part is done.

## To start with..

clone the repo to your local.

```
git clone https://github.com/kentaroy47/Deep-Compression.Pytorch
```

## To do pruning..
This will prune the default resnet18 with half parameters.

The pretrained resnet18 should be in checkpoint/ckpt.t7. Accuracy is about 90%.

It will do 20 epochs of retraining as well.

```
python prune.py
```

Even with 50% pruned weights, the accuracy is about the same (90%).

You can try arbitary pruning rates with bellow.

This will prune 75% of conv parameters.


```
python prune.py --prune 0.75
```

## To train your own model
I provided a CIFAR 10 trained resnet18 to get started with.

If you want to train your own network, do:

```

# choose what model you want to use by commentouts in the script.
python train_cifar10.py

```

### TODO:

HuffmanCoding?
