import io
from PIL import Image

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn

from huggingface_hub import login 

from datasets import load_dataset, DatasetDict

from transformers import AutoImageProcessor, ViTForImageClassification
from transformers import Trainer, TrainingArguments, TrainerCallback
from copy import deepcopy
import os
import evaluate

Hugging_face_token=""
login({Hugging_face_token})

# Enable GPU acceleration if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

train_ds = load_dataset("imagefolder", data_dir="D:/DataSetV7/aug_train_70")['train']
val_ds = load_dataset("imagefolder", data_dir="D:/DataSetV7/aug_val_30")['train']
test_ds = load_dataset("imagefolder", data_dir="D:/DataSetV7/test")['train']

our_dataset = DatasetDict({'train': train_ds, 'val': val_ds, 'test': test_ds})

labels = our_dataset['train'].unique('label')
print(f"Number of labels: {len(labels)}")
label2id = {c: idx for idx, c in enumerate(labels)}
id2label = {idx: c for idx, c in enumerate(labels)}

processor = AutoImageProcessor.from_pretrained('google/vit-base-patch16-224')

def transforms(batch):
    # Process images in batches for efficiency
    inputs = processor(batch['image'], return_tensors='pt')
    inputs['labels'] = [label2id[y] for y in batch['label']]
    return inputs

# Apply transforms with caching to avoid repeated processing
processed_dataset = our_dataset.with_transform(transforms)

def collate_fn(batch):
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': torch.tensor([x['labels'] for x in batch])
    }

# Define metrics
accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")
precision = evaluate.load("precision")
recall = evaluate.load("recall")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    # Compute all metrics at once to avoid redundant calculations
    metrics = {
        **accuracy.compute(predictions=predictions, references=labels),
        **f1.compute(predictions=predictions, references=labels, average='weighted'),
        **precision.compute(predictions=predictions, references=labels, average='weighted'),
        **recall.compute(predictions=predictions, references=labels, average='weighted')
    }
    return metrics

# Load the model with GPU support
model = ViTForImageClassification.from_pretrained(
    'google/vit-base-patch16-224',
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True,
   
)

# Freeze base model layers for faster fine-tuning
for name, p in model.named_parameters():
    if not name.startswith('classifier'):
        p.requires_grad = False

num_params = sum([p.numel() for p in model.parameters()])
trainable_params = sum([p.numel() for p in model.parameters() if p.requires_grad])

print(f"{num_params = :,} | {trainable_params = :,}")

# Optimized training arguments
training_args = TrainingArguments(
    output_dir="./vit-base-oxford-iiit-pets",
    per_device_train_batch_size=64,  # Increased batch size, adjust based on GPU memory
    per_device_eval_batch_size=128,  # Larger eval batch size for faster evaluation
    gradient_accumulation_steps=2,   # Effective batch size = 64 * 2 = 128
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    logging_dir="D:/ResearchPaperCode/results/csv",
    num_train_epochs=5,
    learning_rate=5e-4,  # Slightly higher learning rate
    warmup_ratio=0.1,    # Learning rate warmup
    save_total_limit=2,
    remove_unused_columns=False,
    push_to_hub=True,
    report_to='tensorboard',
    load_best_model_at_end=True,
    fp16=True,           # Mixed precision training for speed

)

class CustomCallback(TrainerCallback):
    def __init__(self, trainer) -> None:
        super().__init__()
        self._trainer = trainer
        
    
    def on_epoch_end(self, args, state, control, **kwargs):
        if control.should_evaluate:
            control_copy = deepcopy(control)
            self._trainer.evaluate(eval_dataset=self._trainer.train_dataset, metric_key_prefix="train")
            return control_copy

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
    train_dataset=processed_dataset["train"],
    eval_dataset=processed_dataset["val"],
    tokenizer=processor
)

trainer.add_callback(CustomCallback(trainer))

# Optional: add early stopping callback
from transformers import EarlyStoppingCallback
early_stopping = EarlyStoppingCallback(early_stopping_patience=2)
trainer.add_callback(early_stopping)

train = trainer.train()

trainer.save_model("D:/ResearchPaperCode/models")
results = trainer.evaluate(processed_dataset['test'])
print(results)

# Save confusion matrix for analysis
predictions = trainer.predict(processed_dataset['test'])
y_preds = np.argmax(predictions.predictions, axis=1)
y_true = predictions.label_ids

conf_matrix = np.zeros((len(labels), len(labels)))
for t, p in zip(y_true, y_preds):
    conf_matrix[t, p] += 1

plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='g', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig("D:/ResearchPaperCode/results/confusion_matrix_V7_70.png")