# CSCI-680-02-Project-

# Project: Comparative Analysis of Transformer Models for Code Summarization

1. Fine Tuned Model CodeT5- https://drive.google.com/drive/folders/1FMgb_zInBw4NTAnGiTMOjazh9Mewa2wQ?usp=sharing
2. Fine-tuned Mdel Codebert - https://drive.google.com/drive/folders/10wl6dJwrBPS2U5VZfGpDZ4TLf91tBrnw?usp=sharing
3. fine-tuned Model GPT2- https://drive.google.com/drive/folders/1s-j21dh7FZTx00eeTUT1sdP1qv_vqReB?usp=sharing
## Introduction
This repository contains the source code and datasets for a Code Summarization and Analysis Project aimed at improving the understanding and documentation of software code. The project leverages advanced machine learning models such as CodeBERT, CodeT5, and GPT-2 to automatically generate summaries and documentation for given snippets of code. The goal is to assist developers in managing and maintaining large codebases by providing concise, human-readable summaries and accurate documentation generated through state-of-the-art natural language processing techniques.

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

# Convert CSV data to JSON format
python json_conversion.py

# Preprocess the dataset
python Dataset_preprocessing.py

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
```

# For eval Codebert model

```bash
python Codebert_test.py
```
# For eval GP2 model

```bash
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
git clone https://github.com/afia2023/Afia--CSCI680-Assignment2
cd Afia--CSCI680-Assignment2

Create a virtual environment (optional but recommended):

python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

Install the required packages:
pip install -r requirements.txt

```

### Assignment

The project pdf report has been added as -Afia_Farjana_Project.pdf









