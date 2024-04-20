# Development of hate and misogyny detection models for the project "NoMoreHate"

# Background

Hundreds of millions of users actively participate in political discussions on social media worldwide. While these discussions offer opportunities to exchange different opinions and perspectives, they also often contain hate speech, toxicity or statements of political radicalization. In the project “Understanding, Detecting, and Mitigating Online Misogyny Against Politically Active Women”,  researchers of TUM and LMU work together on developing efficient techniques to combat misogyny and hate speech on social media platforms and mitigate their negative effects on victims.

# Related Work

* [A systematic review of Hate Speech automatic detection using Natural Language Processing](https://arxiv.org/abs/2106.00742)
* [Overview of the Evalita 2018 Task on Automatic Misogyny Identification](https://ceur-ws.org/Vol-2263/paper009.pdf)
* [Identifying Different Layers of Online Misogyny](https://arxiv.org/abs/2212.00480)
* [A Survey on Automatic Detection of Hate Speech in Text](https://dl.acm.org/doi/10.1145/3232676)
* [The Datafication of Hate: Expectations and Challenges in Automated Hate Speech Monitoring](https://www.frontiersin.org/articles/10.3389/fdata.2020.00003/full)
* [Hate speech detection: Challenges and solutions](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0221152)
* [A Survey on Detecting and Preventing Hateful Comments on Social Media Using Deep Learning](https://link.springer.com/chapter/10.1007/978-981-19-3575-6_30)


# Project Overview

* We trained mainly 6 models: Seperate models for hate and misogyny detection in English, German and Portuguese languages.
* Transformer-based pretrained deep learning models are finetuned on relevant datasets and evaluations are reported.
* Experiments like data augmentation is applied to improve performances on low source languages.

# Setup

* Preprocessing steps are applied on datasets to make the data ready for finetuning phase.
* In 'DataPreparation.ipnyb', you can see example of initial modification of datasets coming from different sources. The purpose is to standardize the dataset format as having two columns {text, label}.
* In 'HateMisogynyDetection.ipnyb', sections are exploratory findings, further preprocessing, tokenization, training, evaluation and model saving.
* For each language and task, you can run the generic notebook by setting the dataset name (or its path), and the name of tokenizer and pretrained model.

### Preprocessing
Depending on the desired objective, different preprocessing steps are implemented: case lowering, punctuation removal, url removal, emoji removal, stopwords removal, frequent words removal. 
### Training
We used different versions of **BERT** model as Pretrained Language Model to finetune with each downstream objectives. And we set the following hyperparameters to make the models optimal:
* Number of epochs: 3 (can be modified depending on the validation loss)
* learning rate: 1e-5
* optimizer: Adam
* batch size: 32
### Evaluation
We evaluated each model's performance on functional test sets. 
* Using **model.eval()** function, it's important to keep weights not updated (by **torch.no_grad()** function).
* Using **classification_report** function of **sklearn** library

# Results and Challenges
* Dataset exploration
  <p align="center">
    <img width="400" height="200" src="https://raw.githubusercontent.com/hasanselimyagci/nomorehate/main/hateEnglish.png?token=GHSAT0AAAAAACNXEZ3RXCB7QCTORT3PCSR4ZREHRJA">
  </p>

* Training-Validation loss curve
  <p align="center">
    <img width="400" height="200" src="https://raw.githubusercontent.com/hasanselimyagci/nomorehate/main/hateEnTrainValid.png?token=GHSAT0AAAAAACNXEZ3RBUA7X4P45K6WI3H4ZREHS6A">
  </p>

* Evaluation report
  <p align="center">
    <img width="400" height="200" src="https://github.com/hasanselimyagci/nomorehate/blob/main/hateEnEval.png">
  </p>

* Misclassified texts
  <p align="center">
    <img width="800" height="200" src="https://raw.githubusercontent.com/hasanselimyagci/nomorehate/main/misclassifiedHateEn.png?token=GHSAT0AAAAAACNXEZ3Q5DIDDOIWFPY3I7XQZREHUOQ">
  </p>


# Future Work
* Bias Mitigation
* Language Adapters
* Human Feedback Reinforcement Learning
