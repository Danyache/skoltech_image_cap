# Requirements
## Specification of dependencies
- [PyTorch](https://pytorch.org/) (An open source deep learning platform) 

All the requirements are in file `req.txt`.

Also you can use the environment in the DGX server in `/home/dchesakov/anaconda3`

# Data
The dataset is provided in `skoltech_image_cap/NLMCXR_data/images`

To create data you should use `Создание датасета для обучения Show-Attend-Tell.ipynb`. There are two different parts to create dataset for the refular training and cross-GPT-2 training. 

# Training

First of all you should create folder `skoltech_image_cap/NLMCXR_data/embeds` and download file `glove.6B.300d.txt` there from https://www.kaggle.com/thanakomsn/glove6b300dtxt

Also create folder `skoltech_image_cap/NLMCXR_data/output3` for pretrained GPT-2 models. They are also on DGX server in `/home/dchesakov/transformers/output3`.

To train the model **from scratch** just use the command:

`python train.py`

To continue the training from a **checkpoint** just use the `checkpoint` param in the code.

If you use embeddings -- don't forget to change the size of embeddings in `train.py`

If you use GPT-2 cross training -- choose `True` for the param `GPT_also`. Also dont forget `version` and `n_min` args, which you used for the dataset creation.

# Evaluation 

To get the metric results use the files 
`bleu_predict.ipynb` (for vanilla SAT implementation and SAT+GPT) or `bleu_predict_with_gpt.ipynb` for cross-GPT model prediction. 

# Hyperparameters

All the hyperparams should be choosen in the `train.py` file. 
The best training was with `epochs` = 20 and params that are stated in the file for now. But a wide range of hyperparams (`dropout`, `attention_dim`, `decoder_dim`, learning rates) were used.

