 # python notebooks
 * All python notebooks in this folder can be run locally or on Google Colab. To run locally you can run the command ``` jupyter notebook ``` in your commandline (in the same file path you saved the notebook) or you can upload it to Colab and run. Notebooks are making easier to run individual cells and get interim results as you run the code.
 * You can run *HateMisogynyClassification* notebook to finetune a pretrained language model on hate or misogyny detection task. Inside the notebook some blocks are commented out, such as some preprocessing steps (which could be useful depending on the source data). Sections of the notebook includes: loading the data, analyzing and preprocessing, training-validation-test splitting, tokenization, finetuning, evaluation, finding misclassified samples, saving the model.
 * Since in that notebook the setting is for BERT-like transformer models, you can modify the code like below to use a autoregressive model like GPT instead. For tokenizer:
   ```python
   CHECKPOINT="EleutherAI/gpt-neo-125m"
   from transformers import AutoTokenizer
   tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
   tokenizer.padding_side = "left"
   tokenizer.pad_token = tokenizer.eos_token
   ```
* You can also modify the training step by using transformers library's Trainer object:
  ```python
  training_args = TrainingArguments(
    output_dir="eng_misogyny_model_mixdata",
    learning_rate=1e-5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=2,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch"
  )
  trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["valid"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
  )

  trainer.train()
  ```
  With modifiable metrics:
  
  ```python
  def compute_metrics(eval_pred):
   predictions, labels = eval_pred
   predictions = np.argmax(predictions, axis=1)
   acc = accuracy.compute(predictions=predictions, references=labels)
   prec= precision.compute(predictions=predictions, references=labels)
   rec = recall.compute(predictions=predictions, references=labels)
   f = f1_metric.compute(predictions=predictions, references=labels)
   return {'accuracy': acc, 'p': prec, 'r': rec,'f1': f}
  ```
   
 * In *DataPreparation* notebook, you can find several functions that are useful for preprocessing and exploratory analysis, if you asses the quality of the dataset without proceeding with training step.
   
 * In *InferenceAndTest* notebook, you can find the implementations of *inference(text)* and *testOn('dataset.csv')* functions. Input a text or path string of your test dataset to get the results. 
