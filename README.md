# Image Captioning with Attention : How to Increase Speed and Quality of X-Ray Diagnostics

Here we have three main architectures implemented. 

In the `Show-Attend-Tell` folder is the implementation of https://arxiv.org/pdf/1502.03044v1.pdf paper and it's versions combined with GPT-2 models.

In the `On-the-Automatic-Generation-of-Medical-Imaging-Reports` is the implementation of https://arxiv.org/pdf/1711.08195.pdf paper. 

In the `Transformer-Based-Generation` folder we have Transformer-based implementation using the [fairseq github](https://github.com/krasserm/fairseq-image-captioning). 

All the models are implemented using the CheXNet weights. The weights that are used are at **DGX** at `/raid/data/cxr14-2/DenseNet121_aug4_pretrain_WeightBelow1_1_0.829766922537.pkl`. If you dont use the **DGX** server you should change the weights path in the following files:
* Transformer-Based-Generation/preprocess/preprocess_images_cxr.py
* Show-Attend-Tell/models.py
* On-the-Automatic-Generation-of-Medical-Imaging-Reports/models.py
