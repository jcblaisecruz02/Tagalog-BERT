# BERT/ULMFiT Filipino
This repository contains data and pretrained models for the paper **Evaluating Language Model Finetuning Techniques for Low-resource Languages**.

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
* ```full.txt``` [[Link]](https://storage.googleapis.com/blaisecruz/datasets/wikitext-tl-39/full.txt) -- Full dataset with no preprocessing.
* ```train.txt``` [[Link]](https://storage.googleapis.com/blaisecruz/datasets/wikitext-tl-39/train.txt) -- Preprocessed training split.
* ```valid.txt``` [[Link]](https://storage.googleapis.com/blaisecruz/datasets/wikitext-tl-39/valid.txt) -- Preprocessed validation split.
* ```test.txt``` [[Link]](https://storage.googleapis.com/blaisecruz/datasets/wikitext-tl-39/test.txt) -- Preprocessed test split with unknown tokens masked.
* ```raw_test.txt``` [[Link]](https://storage.googleapis.com/blaisecruz/datasets/wikitext-tl-39/raw_test.txt) -- Raw test split with unknown tokens not masked.

## BERT Models
We provide pretrained BERT-Base models using our provided corpora. The models were trained on Google's Tensor Processing Unit (TPU) v2-8 using Google's [scripts](https://github.com/google-research/bert). Please see our paper for model and training details.

* BERT-TL-Base-Cased [[Link]](https://storage.googleapis.com/blaisecruz/bert-tagalog/models/bert-tagalog-base-cased.zip)
* BERT-TL-Base-Uncased [[Link]](https://storage.googleapis.com/blaisecruz/bert-tagalog/models/bert-tagalog-base-uncased.zip)

The results on the paper are done in PyTorch using Huggingface's [BERT implementation](https://github.com/huggingface/pytorch-pretrained-BERT), however, our checkpoints are also compatible with the Tensorflow code in Google's [finetuning repository](https://github.com/google-research/bert). Please consult either repository for details on how to use the BERT models.

For usage, please ensure that you have a GPU with at least 12GB VRAM to fit sizeable batch sizes that will not hurt finetuning performance. Please check the ```config.json``` file in the BERT models for details on how to setup the models for use.

## ULMFiT Models
We provide a pretrained AWD-LSTM model using our provided corpora.

* AWD-LSTM [[Link]](https://storage.googleapis.com/blaisecruz/ulmfit-tagalog/models/pretrained-wikitext-tl-39.zip)

The model and the results on the paper were done in PyTorch using in-house tools we use at our lab. As such, we cannot release pretraining/finetuning code. However, extra work has been done to ensure that our pretrained checkpoints are also compatible with the [FastAI](https://github.com/fastai/fastai) library, which provides the original implementation for ULMFiT. Please check details in the library's repository for more details.

## Contributing
If you find any errors, or would like to help with anything, feel free to drop by our issues tab and let us know!

*This repository is managed by the De La Salle University Machine Learning Group*
