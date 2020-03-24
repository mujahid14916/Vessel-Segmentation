# Vessel-Segmentation
Retinal Vessel Segmentation

## Reference architecture
- [SegCaps](https://github.com/lalonderodney/SegCaps)
- [BCDU](https://github.com/rezazad68/BCDU-Net/tree/master/Retina%20Blood%20Vessel%20Segmentation)

***

## Setup
- Install Python 3.6.8 or higher
- Run **python -r requirements.txt**

## Train
- Run **python data_split.py** to generate training and testing split
- Run **python pre_process.py** to generate input training data
- Run **python patch_generator.py** to generate validation patches
- Run **python train_segcaps.py** to start training process (change parameters acccordingly)
