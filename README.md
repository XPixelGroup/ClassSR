# ClassSR
(CVPR2021) ClassSR: A General Framework to Accelerate Super-Resolution Networks by Data Characteristic.

[Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Kong_ClassSR_A_General_Framework_to_Accelerate_Super-Resolution_Networks_by_Data_CVPR_2021_paper.pdf)

Authors: Xiangtao Kong, [Hengyuan Zhao](https://github.com/zhaohengyuan1), [Yu Qiao](https://scholar.google.com/citations?user=gFtI-8QAAAAJ&hl=zh-CN), [Chao Dong](https://scholar.google.com.hk/citations?user=OSDCB0UAAAAJ&hl=zh-CN)


![Demo Image](https://raw.githubusercontent.com/Xiangtaokong/ClassSR/main/demo_images/show.png)

## Abstract

We aim at accelerating super-resolution (SR) networks on large images (2K-8K). The large images are usually decomposed into small sub-images in practical usages. Based on this processing, we found that different image regions have different restoration difficulties and can be processed by networks with different capacities. Intuitively, smooth areas are easier to super-solve than complex textures. To utilize this property, we can adopt appropriate SR networks to process different sub-images after the decomposition. On this basis, we propose a new solution pipeline -- ClassSR that combines classification and SR in a unified framework. In particular, it first uses a Class-Module to classify the sub-images into different classes according to restoration difficulties, then applies an SR-Module to perform SR for different classes. The Class-Module is a conventional classification network, while the SR-Module is a network container that consists of the to-be-accelerated SR network and its simplified versions. We further introduce a new classification method with two losses -- Class-Loss and Average-Loss to produce the classification results. After joint training, a majority of sub-images will pass through smaller networks, thus the computational cost can be significantly reduced. Experiments show that our ClassSR can help most existing methods (e.g., FSRCNN, CARN, SRResNet, RCAN) save up to 50% FLOPs on DIV8K datasets. This general framework can also be applied in other low-level vision tasks.

## Dependencies

- Python >= 3.6 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux))
- [PyTorch >= 1.5.0](https://pytorch.org/)
- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)
- Python packages: `pip install numpy opencv-python lmdb`
- [option] Python packages: [`pip install tensorboardX`](https://github.com/lanpa/tensorboardX), for visualizing curves.

# Codes 
- Our codes version based on [BasicSR](https://github.com/xinntao/BasicSR). 

## How to test a single branch
1. Clone this github repo. 
```
git clone https://github.com/Xiangtaokong/ClassSR.git
cd ClassSR
```
2. Download the testing datasets ([DIV2K_valid](https://data.vision.ee.ethz.ch/cvl/DIV2K/)). 

3. Download the [divide_val.log](https://drive.google.com/file/d/1zMDD9Z_-fM2R2qm2QLoq7N2LMG6V92JT/view?usp=sharing) and move it to `.codes/data_scripts/`.

4. Generate simple, medium, hard (class1, class2, class3) validation data. 
```
cd codes/data_scripts
python extract_subimages_test.py
python divide_subimages_test.py
```
5. Download [pretrained models](https://drive.google.com/drive/folders/1jzAFazbaGxHb-xL4vmxc-hHbR1J-uek_?usp=sharing) and move them to  `./experiments/pretrained_models/` folder. 

6. Run testing for a single branch.
```
cd codes
python test.py -opt options/test/test_FSRCNN.yml
python test.py -opt options/test/test_CARN.yml
python test.py -opt options/test/test_SRResNet.yml
python test.py -opt options/test/test_RCAN.yml
```

7. The output results will be sorted in `./results`. 

## How to test ClassSR
1. Clone this github repo. 
```
git clone https://github.com/Xiangtaokong/ClassSR.git
cd ClassSR
```

2. Download the testing datasets (Test2K, 4K, 8K) [Google Drive](https://drive.google.com/drive/folders/18b3QKaDJdrd9y0KwtrWU2Vp9nHxvfTZH?usp=sharing) or [Baidu Drive](https://pan.baidu.com/s/1OARDfd2x3ynQs7m1tu_RnA) (Password: 7dw1) .

3. You can also download the source data [DIV8K](https://competitions.codalab.org/competitions/22217#participate). Test8K contains the images (index 1401-1500) from DIV8K. Test2K/4K contain the images (index 1201-1300/1301-1400) from DIV8K which are downsampled to 2K and 4K resolution. (In this way, you need register for the competition (Ntire 2020 was held on 2020, but we can register now), then you can download DIV8K dataset.)

4. Download [pretrained models](https://drive.google.com/drive/folders/1jzAFazbaGxHb-xL4vmxc-hHbR1J-uek_?usp=sharing) and move them to  `./experiments/pretrained_models/` folder. 

5. Run testing for ClassSR.
```
cd codes
python test_ClassSR.py -opt options/test/test_ClassSR_FSRCNN.yml
python test_ClassSR.py -opt options/test/test_ClassSR_CARN.yml
python test_ClassSR.py -opt options/test/test_ClassSR_SRResNet.yml
python test_ClassSR.py -opt options/test/test_ClassSR_RCAN.yml
```
6. The output results will be sorted in `./results`. 


## How to train a single branch
1. Clone this github repo. 
```
git clone https://github.com/Xiangtaokong/ClassSR.git
cd ClassSR
```
2. Download the training datasets([DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/)) and validation dataset(Set5).

3. Download the [divide_train.log](https://drive.google.com/file/d/1WhyYYZHfpoNEjslojuqZLR46Nlr15zqQ/view?usp=sharing) and move it to `.codes/data_scripts/`.

4. Generate simple, medium, hard (class1, class2, class3) training data. 
```
cd codes/data_scripts
python data_augmentation.py
python generate_mod_LR_bic.py
python extract_subimages_train.py
python divide_subimages_train.py
```

5. Run training for a single branch (default branch1, the simplest branch).
```
cd codes
python train.py -opt options/train/train_FSRCNN.yml
python train.py -opt options/train/train_CARN.yml
python train.py -opt options/train/train_SRResNet.yml
python train.py -opt options/train/train_RCAN.yml
```
6. The experiments will be sorted in `./experiments`. 

## How to train ClassSR

1. Clone this github repo. 
```
git clone https://github.com/Xiangtaokong/ClassSR.git
cd ClassSR
```
2. Download the training datasets ([DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/)) and validation dataset([DIV2K_valid](https://data.vision.ee.ethz.ch/cvl/DIV2K/), index 801-810). 


3. Generate training data (the all data(1.59M) in paper).
```
cd codes/data_scripts
python data_augmentation.py
python generate_mod_LR_bic.py
python extract_subimages_train.py
```
4. Download [pretrained models](https://drive.google.com/drive/folders/1jzAFazbaGxHb-xL4vmxc-hHbR1J-uek_?usp=sharing)(pretrained branches) and move them to  `./experiments/pretrained_models/` folder. 

5. Run training for ClassSR.
```
cd codes
python train_ClassSR.py -opt options/train/train_ClassSR_FSRCNN.yml
python train_ClassSR.py -opt options/train/train_ClassSR_CARN.yml
python train_ClassSR.py -opt options/train/train_ClassSR_SRResNet.yml
python train_ClassSR.py -opt options/train/train_ClassSR_RCAN.yml
```
6. The experiments will be sorted in `./experiments`. 

## How to generate demo images

Generate demo images like this one:

![Demo Image](https://raw.githubusercontent.com/Xiangtaokong/ClassSR/main/demo_images/show.png)

Change the 'add_mask: False' to True in test_ClassSR_xxx.yml and run testing for ClassSR.

## Citation
```
@InProceedings{Kong_2021_CVPR,
    author    = {Kong, Xiangtao and Zhao, Hengyuan and Qiao, Yu and Dong, Chao},
    title     = {ClassSR: A General Framework to Accelerate Super-Resolution Networks by Data Characteristic},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2021},
    pages     = {12016-12025}
}
```

## Contact
Email: xt.kong@siat.ac.cn
