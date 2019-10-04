# BERT/ULMFiT Filipino
This repository contains code, data and pretrained models for the paper **Evaluating Language Model Finetuning Techniques for Low-resource Languages**.

If you use any of our models or found or work useful, please cite appropriately:
```
@article{evaluating2019cruz,
  title={{Evaluating Language Model Finetuning Techniques for Low-resource Languages}},
  author={Cruz, Jan Christian Blaise and Cheng, Charibeth},
  journal={arXiv preprint arXiv:1907.00409},
  year={2019}
}
```

## Dataset
The WikiText-TL-39 dataset is available in five files:
* [**```full.txt```**](https://storage.googleapis.com/blaisecruz/datasets/wikitext-tl-39/full.txt) -- Full dataset with no preprocessing.
* [**```train.txt```**](https://storage.googleapis.com/blaisecruz/datasets/wikitext-tl-39/train.txt) -- Preprocessed training split.
* [**```valid.txt```**](https://storage.googleapis.com/blaisecruz/datasets/wikitext-tl-39/valid.txt) -- Preprocessed validation split.
* [**```test.txt```**](https://storage.googleapis.com/blaisecruz/datasets/wikitext-tl-39/test.txt) -- Preprocessed test split with unknown tokens masked.
* [**```raw_test.txt```**](https://storage.googleapis.com/blaisecruz/datasets/wikitext-tl-39/raw_test.txt) -- Raw test split with unknown tokens not masked.

Like the original [WikiText Dataset](https://blog.einstein.ai/the-wikitext-long-term-dependency-language-modeling-dataset/), no further preprocessing is necessary except splitting by space. Documents are separated by heading titles. For corpus details and details on how the datasets are further preprocessed for BERT & ULMFiT, please refer to our paper.

## BERT Models
We provide pretrained BERT models trained using our provided corpora. The models were trained on Google's Tensor Processing Unit (TPU) v3-8 using Google's pretraining [scripts](https://github.com/google-research/bert). Please see our paper for model and training details. We provide our BERT-Large models in a smaller maximum sequence length (128 MSL) so they can fit in GPUs using a smaller batch size without hurting the performance.

**Base Models**
* [**```BERT-TL-Base-Cased```**](https://storage.googleapis.com/blaisecruz/bert-tagalog/models-512/bert-tagalog-base-cased.zip) -- 12-layer, 768-hidden, 12-heads, 110M parameters, 512 MSL
* [**```BERT-TL-Base-Uncased```**](https://storage.googleapis.com/blaisecruz/bert-tagalog/models-512/bert-tagalog-base-uncased.zip) -- 12-layer, 768-hidden, 12-heads, 110M parameters, 512 MSL

**Large Models**
* [**```BERT-TL-Large-Cased```**](https://storage.googleapis.com/blaisecruz/bert-tagalog/models/bert-tagalog-large-cased.zip) -- 24-layer, 1024-hidden, 16-heads, 340M parameters, 128 MSL

**Whole Word Masking Models**
* [**```BERT-TL-Base-Cased```**](https://storage.googleapis.com/blaisecruz/bert-tagalog/models-512/bert-tagalog-base-cased-WWM.zip) -- 12-layer, 768-hidden, 12-heads, 110M parameters, 512 MSL, Whole Word Masking Pretraining
* [**```BERT-TL-Base-Uncased```**](https://storage.googleapis.com/blaisecruz/bert-tagalog/models-512/bert-tagalog-base-uncased-WWM.zip) -- 12-layer, 768-hidden, 12-heads, 110M parameters, 512 MSL, Whole Word Masking Pretraining

The results on the paper are done in PyTorch using Huggingface's [BERT implementation](https://github.com/huggingface/transformers), however, our checkpoints are also compatible with the Tensorflow code in Google's [finetuning repository](https://github.com/google-research/bert). Please consult either repository for details on how to use the BERT models.

For usage, please ensure that you have a GPU with at least 16GB VRAM to fit sizeable batch sizes that will not hurt finetuning performance. Please check the ```config.json``` file in the BERT models for details on how to setup the models for use. 

We have included a finetuning script for text classification in this repository. This example setup finetunes the model to a sentiment classification task using 4 GPUs.

```sh
python bert_classify.py \
    --data=data/corpus \
    --filename=train \
    --model=bert/bert-tagalog-base-cased \
    --config=bert_config.json \
    --train_size=0.7 \
    --epochs=3 \
    --msl=512 \
    --bs=32 \
    --lr=5e-5 \
    --warmup=0.1 \
    --max_norm=1.0 \
    --cuda \
    --multi \
    --gpus=4 \
    --seed=42\
    --output=models/finetuned.pt
```

## ULMFiT Models
We provide a pretrained AWD-LSTM model using our provided corpora.

* [**```AWD-LSTM```**](https://storage.googleapis.com/blaisecruz/ulmfit-tagalog/models/pretrained-wikitext-tl-39.zip)

The model and the results on the paper were done in PyTorch using in-house tools we use at our lab. As such, we cannot release pretraining/finetuning code. However, extra work has been done to ensure that our pretrained checkpoints are also compatible with the [FastAI](https://github.com/fastai/fastai) library, which provides the original implementation for ULMFiT. Please check details in the library's repository for more information on usage.

## Changes and Future Releases
We will update this repository as changes are done or improvements are made. Do note that BERT-Large models will not be trained for pragmatic reasons. Let us know if you need anything beyond the models and data provided here.

## Contributing
If you find any errors, have suggestions, or would like to help with anything, feel free to drop by our issues tab and let us know!

*This repository is managed by the De La Salle University Machine Learning Group*
