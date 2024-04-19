# Development of hate and misogyny detection models for the project "NoMoreHate"

# Background

Hundreds of millions of users actively participate in political discussions on social media worldwide. While these discussions offer opportunities to exchange different opinions and perspectives, they also often contain hate speech, toxicity or statements of political radicalization. In the project “Understanding, Detecting, and Mitigating Online Misogyny Against Politically Active Women”,  researchers of TUM and LMU work together on developing efficient techniques to combat misogyny and hate speech on social media platforms and mitigate their negative effects on victims.

To improve the social media experiences of online hate victims and help them control the amount of hate they receive on a daily basis, the plugin “NoMoreHate” should be developed. This plugin should allow users to contextualize and block hateful messages and user profiles within several social media platforms. 

# Related Work

* [A systematic review of Hate Speech automatic detection using Natural Language Processing](https://arxiv.org/abs/2106.00742)
* [Overview of the Evalita 2018 Task on Automatic Misogyny Identification](https://ceur-ws.org/Vol-2263/paper009.pdf)
* [Identifying Different Layers of Online Misogyny](https://arxiv.org/abs/2212.00480)
* [A Survey on Automatic Detection of Hate Speech in Text](https://dl.acm.org/doi/10.1145/3232676)
* [The Datafication of Hate: Expectations and Challenges in Automated Hate Speech Monitoring](https://www.frontiersin.org/articles/10.3389/fdata.2020.00003/full)
* [Hate speech detection: Challenges and solutions](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0221152)
* [A Survey on Detecting and Preventing Hateful Comments on Social Media Using Deep Learning](https://link.springer.com/chapter/10.1007/978-981-19-3575-6_30)


# Project Goal
  
## General Requirements
* Efficiency: The models should be fast and efficient to allow for real-time detection of hateful messages. Different machine learning algorithms should be developed and tested to select the most appropriate techniques for efficient hate speech detection.
* Extendable: Models should be developed in a way, that they are adaptable to feedback on the quality of the estimation, reported by either researchers or users.
* Explainable: Different approaches should be developed to make the model interpretable and explainable. This should help researchers and users to understand why a certain message was classified as hateful by the model.
