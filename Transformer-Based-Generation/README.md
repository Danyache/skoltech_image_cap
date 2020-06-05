# Image Captioning Transformer

This projects extends [pytorch/fairseq](https://github.com/pytorch/fairseq) and based on [fairseq-image-captioning](https://github.com/krasserm/fairseq-image-captioning)

# Requirements
All the dependencies and requirements can be found here https://github.com/krasserm/fairseq-image-captioning in the **Setup** part. You should use `environment.yml` for the environment creation.

# DataSet

You should change the `mysplit_test_images.txt`, `mysplit_train_images.txt`, `mysplit_valid_images.txt` files using the paths to the **train**, **test** and **validation** images. Images are the same. 

Then you have to use commands

`./preprocess_captions_cxr.sh ms-coco`

and

`./preprocess_images_cxr.sh ms-coco`

for the captions preprocessing and images preprocessing accordingly.

# Training

For the model training you should use the command as follows:

`
python -m fairseq_cli.train --save-dir .checkpoints_cxr_final_s --user-dir task --task captioning --arch simplistic-captioning-arch --decoder-layers 6 --features grid --optimizer adam --adam-betas "(0.9,0.999)" --lr 0.0003 --lr-scheduler inverse_sqrt --min-lr 1e-09 --warmup-init-lr 1e-8 --warmup-updates 8000 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --weight-decay 0.0001 --dropout 0.3 --max-epoch 90 --max-tokens 4096 --max-source-positions 64 --encoder-embed-dim 512 --num-workers 2 --features-dir output_cxr --captions-dir output_cxr --features-dim 1024
`

To understand the params more properly you can look at [fairseq-image-captioning](https://github.com/krasserm/fairseq-image-captioning) or at documentation [here](https://fairseq.readthedocs.io/en/latest/command_line_tools.html#fairseq-train)

# Evaluation

For the model evaluation you should firstly predict with the command:

`
python generate.py   --user-dir task   --features grid   --tokenizer moses   --bpe subword_nmt   --bpe-codes output_cxr/codes.txt   --beam 5   --split test   --path .checkpoints_cxr_final_s/checkpoint_best.pt   --input output_cxr/test-ids.txt   --output output_cxr/test-predictions-2805.json
`

Then you should use `TransformerBasedEvaluation.ipynb` file for the models evaluation. 

