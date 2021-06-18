# deeplab_carla_segmentation
# 概述

本demo是对论文[End-to-End Model-Free Reinforcement Learning for Urban Driving using Implicit Affordances (CVPR2020)](https://arxiv.org/abs/1911.10868)中基于语义分割的Encoder的第三方复现。仿真图像数据生成自Carla仿真平台；语义分割模型为DeepLab V3+，backbone由ResNet18更换为MobileNet以减少参数，适应强化学习训练。



# 数据

数据使用自动驾驶平台Carla内置的[Semantic segmentation camera](https://carla.readthedocs.io/en/0.9.11/ref_sensors/#semantic-segmentation-camera)生成，共22438张大小为512x256的仿真场景图及其对应的分割标注。分割标注按照Carla平台默认类别进行设置。

注：实际生成数据中不包括第0、3类数据，并且第12，17类数据量极少。



将数据按照7：2：1划分为训练、验证、测试集。

生成的数据示例：



# 环境依赖

```
torch==1.1.0          
torchvision==0.3.0
tqdm==4.32.2
tensorboard==1.14.0
Pillow==6.2.0
opencv-python==4.1.0.25
```

# 功能和描述

## DeepLab V3+模型

DeepLab是由来自Google的团队最早于2014年提出的语义分割算法，目前已经推出了三个大版本（V1、V2、V3）和一个小版本V3+，一脉相承，目前最新的V3+版本于2018年提出，在很多benchmark上表现优秀，目前仍然是很多方法的对比Baseline。

为了减少模型参数，我们使用MobileNet作为Backbone。



## 代码结构

```
deeplab_carla_segmentation/
│
├── train.py - 训练入口脚本
├── inference.py - 调用训练好的模型进行推理，生成分割图
├── trainer.py - 训练主模块
├── for_rl.py - 强化学习调用接口
├── config_Carla.json - 配置文件
├── generate_video.py - 由图像帧序列生成视频
├── base/ - 基类
│   ├── base_data_loader.py
│   ├── base_model.py
│   ├── base_dataset.py
│   └── base_trainer.py
│
├── dataloader/ - Carla数据集加载所需模块
│
├── models/ - 神经网络模型
│   ├── mobilenet_2.py - MobileNet V2
│   ├── mobilenet.py - MobileNet
│   ├── network_base_mobile.py - 基于MobileNet的DeepLab V3+
│   └── network_base_res18.py - 基于ResNet18的DeepLab V3+
│
├── saved/ - 保存训练日志和模型
│
├── test_result/ - 保存测试得到的分割图
│   
└── utils/ - 一些功能模块
    ├── losses.py - 损失函数
    ├── palette.py - 用于为生成的分割图上色，专门匹配Carla数据集
    └── lr_scheduler - 学习率调整
```

## 执行前的准备

为了能够执行训练和验证，首先需要设定好配置文件`config_Carla.json`。

参考配置如下：

```json
{
    "name": "320_2_4_mobile2_carla",
    "n_gpu": 2,  // 使用GPU的数目，需要与执行训练时设定的gpu数目匹配
    "use_synch_bn": false,

    "arch": {
        "type": "SegModel_mobile",  // 我们使用的模型名称
        "args": {
            "backbone": "mobile",
            "mode": "seg_train",
            "freeze_bn": false,
            "freeze_backbone": false
        }
    },

    "train_loader": {
        "type": "Carla",
        "args":{
						// 需要提前在对应的文件夹中准备好数据集
            "data_dir": "/space1/zhaoqing/dataset/Carla_seg",
            "batch_size": 32,
            "base_size": 256,
            "crop_size": 256,
            "augment": false,
            "shuffle": true,
            "scale": true,
            "flip": true,
            "rotate": true,
            "blur": false,
            "split": "train",
            "num_workers": 8
        }
    },

    "val_loader": {
        "type": "Carla",
        "args":{
						// 验证数据集同理
            "data_dir": "/space1/zhaoqing/dataset/Carla_seg",
            "batch_size": 32,
            "crop_size": 256,
            "val": false,
            "split": "val",
            "num_workers": 4
        }
    },

    "optimizer": {
        "type": "SGD",
        "differential_lr": true,
        "args":{
            "lr": 0.1,
            "weight_decay": 1e-4,
            "momentum": 0.9
        }
    },

    "loss": "CrossEntropyLoss2d",
    "ignore_index": 255,
    "lr_scheduler": {
        "type": "Poly",
        "args": {}
    },

    "trainer": {
        "epochs": 500,
        "save_dir": "saved/",
        "save_period": 10,
  
        "monitor": "max Mean_IoU",
        "early_stop": 10,
        
        "tensorboard": true,
        "log_dir": "saved/runs",
        "log_per_iter": 20,

        "val": true,
        "val_per_epochs": 5
    }
}
```

## Carla数据集结构

```json
Carla_seg/
│
├── train.txt - 训练图像的名称
├── test.txt - 测试图像的名称
├── val.txt - 验证图像的名称
├── img/ - 存放仿真图像
│   ├── 00000001.bmp
│   ├── 00000002.bmp
│   ├── ...
│   └── xxxxxxxx.bmp
│
└── gt/ - 存放img中每张图像分别对应的分割ground truth
    ├── 00000001.bmp
    ├── 00000002.bmp
    ├── ...
    └── xxxxxxxx.bmp
```

## Training

训练模型调用`train.py`文件：

```bash
python train.py --config config_Carla.json --device 0, 1
```

## Inference

利用训练好的模型进行推理，调用`inference.py`文件：

```bash
python inference.py --config config_Carla.json --model ./saved/best_model.pth --images images_folder
```

注：其中images_folder为需要进行测试的图像的文件夹地址

训练好的模型：https://drive.google.com/file/d/1rc6P19kqMHtSu70oJXNrfq33lwbMETxV/view?usp=sharing

## 强化学习系统调用接口

`for_rl.py`是用于强化学习调用的接口，能够生成图像经过encoder之后得到的特征向量

```bash
python for_rl.py --image_path images_folder/00000001.bmp
```

image_path为要进行编码的图像的地址

# 结果和分割效果

## 模型性能



## 分割结果



# 参考和致谢

1. End-to-End Model-Free Reinforcement Learning for Urban Driving using Implicit Affordances
2. 自动驾驶仿真平台Carla：https://carla.org/
3. Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation
4. 主要参考代码：pytorch-segmentation：https://github.com/yassouali/pytorch-segmentation
5. DeepLab V3+ MobileNet Backbone代码：https://github.com/VainF/DeepLabV3Plus-Pytorch

