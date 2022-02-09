##TRACE: A Fast Transformer-based General-Purpose Lossless Compressor

###Introduction

This repository contains the source code and dataset link mentioned in WWW 2022 accepted paper "TRACE:A Fast Transformer-based General-Purpose Lossless Compressor". TRACE is a deep-learning based lossless compressor which compresses byte streams. The model estimates probability of incoming bytes and arithmetic coder utilize this probability to encode.  We want to focus at model sequencial representation ability so **our method do not have pretraining stage**, which mean the program start compression when model knows nothing, and adaptively adjust model parameters during compression. If you want higher compression ratio, do several epochs pretraining would help a lot.  

###Requirements
Nvidia-driver 455.38
CUDA 11.1
cudnn 7605
pytorch==1.7.0
numpy==1.18.5

###Usage

```
git clone https://github.com/mynotwo/A-Fast-Transformer-based-General-Purpose-LosslessCompressor.git
cd ./A-Fast-Transformer-based-General-Purpose-LosslessCompressor
mkdir data
cd data
```

Then download dataset from https://drive.google.com/file/d/18qvfbeeOwD1Fejq9XtgAJwYoXjSV8UaC/view?usp=sharing and put it into  `\data`.
Then do
`tar zcxf compression_data.tar.gz`

To compress data, e.g. file named 'dickens'
```
python compressor.py --input_dir ./data/dickens --batch_size 512 --gpu_id 0 --prefix dickens --hidden_dim 256 --ffn_dim 4096 --seq_len 8 --learning_rate 1e-3 --vocab_dim 4
```
and the compressed file would be book_64_256_4096_bs512_random_seq32.compress.combined
