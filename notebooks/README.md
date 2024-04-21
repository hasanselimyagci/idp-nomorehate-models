 # python notebooks
 
 * You can run *HateMisogynyClassification* notebook to finetune a pretrained language model on hate or misogyny detection task. Inside the notebook some blocks are commented out, such as some preprocessing steps (which could be useful depending on the source data). Sections of the notebook includes: loading the data, analyzing and preprocessing, training-validation-test splitting, tokenization, finetuning, evaluation, finding misclassified samples, saving the model.
 * Since in that notebook the setting is for BERT-like transformer models, you can modify the code like below to use a autoregressive model like GPT instead. For tokenizer:
   ´´´python
   CHECKPOINT="EleutherAI/gpt-neo-125m"
   from transformers import AutoTokenizer
   tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
   tokenizer.padding_side = "left"
   tokenizer.pad_token = tokenizer.eos_token
   tokenized_datasets = ds.map(lambda batch: tokenizer(batch['text'], truncation=True), batched=True)
   ´´´
   
 * In *DataPreparation* notebook, you can find several functions that are useful for preprocessing and exploratory analysis, if you asses the quality of the dataset without proceeding with training step.
   
 * In *InferenceAndTest* notebook, you can find the implementations of *inference(text)* and *testOn('dataset.csv')* functions. Input a text or path string of your test dataset to get the results. 
