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

# Data

You can find available list of datasets found relevant for the task [here](ListOfRelatedDatasets.csv).

# Methodology

* Preprocessing steps are applied on datasets to make the data ready for finetuning phase.
* In 'DataPreparation.ipnyb', you can see example of initial modification of datasets coming from different sources. The purpose is to standardize the dataset format as having two columns {text, label}.
* In 'HateMisogynyDetection.ipnyb', sections are exploratory findings, further preprocessing, tokenization, training, evaluation and model saving.
* In 'Inference.ipnyb', example inferencing code is given.
* For each language and task, you can run the generic notebook by setting the dataset name (or its path), and the name of tokenizer and pretrained model.

### Preprocessing
Depending on the desired objective, different preprocessing steps are implemented: case lowering, punctuation removal, url removal, emoji removal, stopwords removal, frequent words removal. 
### Training
We used different versions of **BERT** model as Pretrained Language Model to finetune with each downstream objectives. We wanted to use lighter models rather than LLMs in order to achieve more green and sustainable AI development.
The following hyperparameters are set to make the models optimal:
* Number of epochs: 3 *(can be increased if the observed validation loss continues to decrease)*
* learning rate: 1e-5
* optimizer: Adam
* batch size: 32
* pretrained language models: google-bert/bert-base-uncased, google-bert/bert-base-multilingual-uncased, google-bert/bert-base-german-cased, neuralmind/bert-base-portuguese-uncased
### Evaluation
We evaluated each model's performance on functional test sets. 
* Using **model.eval()** function, it's important to keep weights not updated (by **torch.no_grad()** function).
* Using **classification_report** function of **sklearn** library

# Results
* Dataset exploration
  <p align="center">
    <img width="400" height="200" src="https://github.com/hasanselimyagci/nomorehate/blob/main/hateEnglish.png">
  </p>

* Training-Validation loss curve
  <p align="center">
    <img width="400" height="200" src="https://github.com/hasanselimyagci/nomorehate/blob/main/hateEnTrainValid.png">
  </p>

* Evaluation report
  <p align="center">
    <img width="400" height="200" src="https://github.com/hasanselimyagci/nomorehate/blob/main/hateEnEval.png">
  </p>

* Misclassified texts
  <p align="center">
    <img width="800" height="200" src="https://github.com/hasanselimyagci/nomorehate/blob/main/misclassifiedHateEn.png">
  </p>

# API integration
To integrate the models, we can call *torch.load(path)* function and corresponding tokenizers, if models are saved to server we can call with the file path or if the models are pushed to HuggingFace hub we can call: `model = AutoModelForSequenceClassification.from_pretrained('modelname')`

And for the inference, we can choose the class with higher probability or if we want to make our model less/more sensitive, we can specify a treshold value (i.e. if class 1 is higher than 0.7) and return the predicted class accordingly.
```python
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', do_lower_case=False)
model = torch.load('/home/css-user/hatespeech/models/enMisogynyModel.pt', map_location=torch.device('cpu'))

input = tokenizer(body['message'], return_tensors='pt')
logits = model(**input).logits
predicted_class_id = logits.argmax().item()
result['hate'] = True if predicted_class_id else False
# or we can return the probability and choose a treshold higher/lower than 0.5 for the predicted class
# from torch.nn.functional import softmax
# probability = (softmax(logits)).data[0][1].item()
# result['hate'] = True if (probability > 0.7) else False
```
# Demo 
You can demo the inference on Hugging Face [link](https://huggingface.co/selimyagci),
and also in [Hugging Face Spaces](https://huggingface.co/spaces/selimyagci/nomorehate).

# Challenges
* Finding available data aligned with well defined hate and misogyny is hard (offensive, sexism, swear word and dialect bias). Annotation bias is also important factor.
* Using LLMs to data augmentation is not optimal because of the LLM's bias and hallucination problem
* Text classification on low source languages are not robust enough for unknown input.
* Transfer learning setting where we finetune a pretrained multilanguage model on mixed data (English + low source language) and validate on low source data has shown relatively better performance.

# Future Work
* **Bias Mitigation**  The pretrained models are tend to have biases like swear word and dialect bias. Debiasing methods for Language Models can be applied as a part of future work, such as knowledge enhancement. 
* **Interpretability**  Making models' decision interpretable is also important both for developers and for users, to achieve that two possible method is implementing LIME model and multitask classification with detailed subcategory of hate or misogyny.
* **Language Adapters**  Considering there are less quality data on low source lanugages and also parameter efficient fine tuning, it is vital to implement language adapters to train on top of the trained high source language model to perform better at other languages.
* **Integrating Human Feedback Reinforcement Learning** One further improvement is integrating feedback of users with RL approach in a continous learning fashion.
