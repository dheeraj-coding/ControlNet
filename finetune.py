import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel, TrainingArguments, Trainer
import evaluate
import numpy as np

pixdataset = load_dataset("timbrooks/instructpix2pix-clip-filtered", split='train', streaming=True)

train_dataset = pixdataset.skip(300_000).shuffle(buffer_size=10000, seed=42)
test_dataset = pixdataset.take(13000).shuffle(buffer_size=1000, seed=42)

tokenizer = AutoTokenizer.from_pretrained("albert-base-v2")
model = AutoModel.from_pretrained("albert-base-v2")


def tokenize_func(sampple):
    return tokenizer(sampple['edit_prompt'], padding='max_length', truncation=True)


train_dataset = train_dataset.map(tokenize_func, batched=True)
test_dataset = test_dataset.map(tokenize_func, batched=True)

metric = evaluate.load('accuracy')


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


training_args = TrainingArguments(output_dir='albert_trainer', evaluation_strategy='epoch')

trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=test_dataset,
                  compute_metrics=compute_metrics)

trainer.train()
