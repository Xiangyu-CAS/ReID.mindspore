# 7小时学会mindspore(CV版)

## 1. MINST + lenet 手写识别
- 血的教训: fc必须初始化,否则loss不下降
````shell script
python ./1_MINST_lenet.py
````

## 2. cifar10 + resnet50 图像分类
````shell script
# 下载数据集
wget https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz
tar -zvxf cifar-10-binary.tar.gz

python 2_CIFAR10_resnet.py --dataset_path ./cifar-10-batches-bin
````
