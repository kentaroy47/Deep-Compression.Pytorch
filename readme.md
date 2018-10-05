# Pytorch implementation of Deep Compression
Dataset: CIFAR10
Models: VGG16, Resnet18~

## To start with..

clone the repo to your local.

```
git clone https://github.com/kentaroy47/Deep-Compression.Pytorch
```

The CIFAR10 trained models are in checkpoint.

|VGG|Res18|
|---|--- |
|85.8%|89.8%|

Resnet is still underfitting.

## To do pruning..
This will prune the default resnet18 with 50% parameters of conv2d.

It will do 20 epochs of retraining as well.

The pretrained resnet18 should be in checkpoint/res18.t7. The val accuracy is about 90%.

```
python prune.py
```

*Pruning results for resnet18.*

|no pruning|10% pruned|25% pruned|50% pruned|75% pruned|
|---|---|---|---|---|
|89.8%|90.3%|91.2%|91.7%|91.9%|



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

# for vgg. can add mobilenet etc..
python train_cifar10.py --net vgg
python train_cifar10.py --net res50

```

## Training script
YOu can train multiple pruned resnet by script.

Do

```
sh prune.sh
```

### TODO:
Quantization
HuffmanCoding
