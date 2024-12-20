# CSCI-680-02-Project-

# Project: Comparative Analysis of Transformer Models for Code Summarization

1. Fine Tuned Model CodeT5- https://drive.google.com/drive/folders/1FMgb_zInBw4NTAnGiTMOjazh9Mewa2wQ?usp=sharing
2. Fine-tuned Mdel Codebert - https://drive.google.com/drive/folders/10wl6dJwrBPS2U5VZfGpDZ4TLf91tBrnw?usp=sharing
3. fine-tuned Model GPT2- https://drive.google.com/drive/folders/1s-j21dh7FZTx00eeTUT1sdP1qv_vqReB?usp=sharing

4. Hugging face dataset format for GPT2 model- https://drive.google.com/drive/folders/1WKAqpWgvBr-ETR_fFYdGbE0VFd4y1MTy?usp=sharing
## Introduction
This repository contains the source code and datasets for a Code Summarization and Analysis Project aimed at improving the understanding and documentation of software code. The project leverages advanced machine learning models such as CodeBERT, CodeT5, and GPT-2 to automatically generate summaries and documentation for given snippets of code. The goal is to assist developers in managing and maintaining large codebases by providing concise, human-readable summaries and accurate documentation generated through state-of-the-art natural language processing techniques.

For testing purpose there are three test dataset 
1. CodeT5_test.json
2. Codebert_test.json
3. And for the GPT2 model before running the training code we need to run the GP2_dataset.py file it will generate  a Hugging  face dataset format dictionary I have added that dataset link above.  

## Table of Contents
- [Project Title](#project-title)

- [Introduction](#introduction)
- [Table of Contents](#table-of-contents)
- [Getting Started](#getting-started)
  - [Dependencies](#dependencies)
  - [Installation](#installation)
- [Usage](#usage)
  - [Fine-tuning](#fine-tuning)
  - [Evaluation](#Evaluation)
- [Project](#Project)

### To Extract the dataset
To start  execute the following commands in sequence:
```bash
# Extract method codes from source files
python Method_extractor.py
```

# Convert CSV data to JSON format
```bash
python json_conversion.py
```

### To preprocess the data
```bash
python Dataset_preprocessing.py
preprocessed_dataset.json dataset will be generated
```
### fine-tune
```bash
# For fine-tune CodeT5 model
python CodeT5_train.py

# For fine-tuning Codebert model
python Codebert_train.py

# For fine-tune GP2 model
python GP2_Train.py

```


### Evaluation
# For eval CodeT5 model

```bash
python CodeT5_test.py

# For eval Codebert model
python Codebert_test.py

# For eval GP2 model
python GP2_Test1.py
```


### Dependencies
This project requires Python 3.8 or later. Dependencies include:

transformers
torch
datasets
numpy
pandas

### Installation
To set up your environment, follow these steps:
Clone the repository:
```bash
git clone https://github.com/afia2023/CSCI-680-02-Project-.git
cd CSCI-680-02-Project-
```
Create a virtual environment (optional but recommended):

python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

Install the required packages:
pip install -r requirements.txt

```

### Assignment

The project pdf report has been added as -Afia_Farjana_Project.pdf









