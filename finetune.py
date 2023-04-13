import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel, TrainingArguments, Trainer
import evaluate
import numpy as np

dataset = load_dataset("timbrooks/instructpix2pix-clip-filtered", streaming=True)

tokenizer = AutoTokenizer.from_pretrained("albert-base-v2")
model = AutoModel.from_pretrained("albert-base-v2")


def tokenize_func(sampple):
    return tokenizer(sampple['edit_prompt'], padding='max_length', truncation=True)


tokenized_dataset = dataset.map(tokenize_func, batched=True)
small_train_dataset = tokenized_dataset["train"].shuffle(buffer_size=10000, seed=42)
small_eval_dataset = tokenized_dataset["test"].shuffle(buffer_size=10000, seed=42)
metric = evaluate.load('accuracy')


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


training_args = TrainingArguments(output_dir='albert_trainer', evaluation_strategy='epoch')

trainer = Trainer(model=model, args=training_args, train_dataset=small_train_dataset, eval_dataset=small_eval_dataset,
                  compute_metrics=compute_metrics)

trainer.train()
