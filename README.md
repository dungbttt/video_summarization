# Video Summarization

## Clone

```bash
!git clone https://github.com/cristiano2003/Video-Summarization-Project.git
%cd Video-Summarization-Project
```

## Setup, Build Package and download Checkpoint and Dataset 

```bash
mkdir -p datasets
wget "https://www.dropbox.com/s/tdknvkpz1jp6iuz/dsnet_datasets.zip"
unzip dsnet_datasets.zip -d datasets

pip install protobuf==3.20.1
```

## Train

```bash

python -m src.train anchor-free --splits ./splits/tvsum.yml --dataset tvsum/summe

```
## Demo

```bash

python app.py

```




                       