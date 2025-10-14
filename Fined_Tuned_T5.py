# Install required packages

import torch
from datasets import load_dataset
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig
)
import numpy as np
import evaluate

# Load Multi30k dataset
dataset = load_dataset("bentrevett/multi30k")

# Initialize tokenizer and model
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)

# Load model without quantization (A100 has enough memory for T5-small)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Move model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

print(f"Model loaded successfully")
print(f"Model device: {model.device}")

# Preprocessing function
def preprocess_function(examples):
    # T5 expects "translate English to German: " prefix
    inputs = ["translate English to German: " + text for text in examples["en"]]
    targets = examples["de"]

    # Tokenize inputs and targets
    model_inputs = tokenizer(
        inputs,
        max_length=128,
        truncation=True,
        padding="max_length"
    )

    # Tokenize targets
    labels = tokenizer(
        targets,
        max_length=128,
        truncation=True,
        padding="max_length"
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Preprocess datasets
tokenized_train = dataset["train"].map(
    preprocess_function,
    batched=True,
    remove_columns=dataset["train"].column_names
)

tokenized_val = dataset["validation"].map(
    preprocess_function,
    batched=True,
    remove_columns=dataset["validation"].column_names
)

tokenized_test = dataset["test"].map(
    preprocess_function,
    batched=True,
    remove_columns=dataset["test"].column_names
)

# Data collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model
)

# Load BLEU metric
bleu_metric = evaluate.load("sacrebleu")

# Compute metrics function
def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    # Decode predictions
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    # Replace -100 in labels (used for padding)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Compute BLEU score
    result = bleu_metric.compute(
        predictions=decoded_preds,
        references=[[label] for label in decoded_labels]
    )

    return {"bleu": result["score"]}

# Training arguments with optimization techniques
training_args = Seq2SeqTrainingArguments(
    output_dir="./t5-multi30k",
    eval_strategy="epoch",  # Changed from evaluation_strategy
    learning_rate=3e-4,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=5,
    weight_decay=0.01,
    save_total_limit=2,
    predict_with_generate=True,

    # Mixed Precision Training (FP16)
    fp16=True,  # Enable mixed precision for A100
    bf16=False,  # Set to True if you want BF16 instead of FP16

    # Gradient Clipping
    max_grad_norm=1.0,  # Clip gradients to prevent exploding gradients

    # Additional optimizations
    gradient_accumulation_steps=1,  # Increase if you need larger effective batch size
    gradient_checkpointing=False,  # Enable if memory constrained (trades compute for memory)
    optim="adamw_torch",  # Optimizer choice

    # Learning rate scheduling
    warmup_steps=500,  # Gradual warmup helps with stability
    lr_scheduler_type="linear",  # Linear decay after warmup

    logging_steps=100,
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="bleu",
    greater_is_better=True,
    report_to="none",  # Change to "tensorboard" if you want logging

    # Performance optimization
    dataloader_num_workers=2,  # Parallel data loading
    dataloader_pin_memory=True,  # Speed up data transfer to GPU
)

print("\nTraining Configuration:")
print(f"- Mixed Precision (FP16): Enabled")
print(f"- Gradient Clipping: {training_args.max_grad_norm}")
print(f"- Batch Size: {training_args.per_device_train_batch_size}")
print(f"- Learning Rate: {training_args.learning_rate}")
print(f"- Warmup Steps: {training_args.warmup_steps}")

# Initialize trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# Train the model
print("Starting training...")
trainer.train()

# Evaluate on test set
print("\nEvaluating on test set...")
test_results = trainer.evaluate(tokenized_test)
print(f"Test BLEU Score: {test_results['eval_bleu']:.2f}")

# Save the final model
trainer.save_model("./t5-multi30k-final")
print("\nModel saved to './t5-multi30k-final'")

# Example translation function
def translate(text, max_length=128):
    input_text = f"translate English to German: {text}"
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)

    outputs = model.generate(
        input_ids,
        max_length=max_length,
        num_beams=4,
        early_stopping=True
    )

    translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translation

# Test translation
print("\nExample translations:")
test_sentences = [
    "A group of people stand in front of an igloo.",
    "A child is playing in the park.",
    "The cat is sitting on the table."
]

for sentence in test_sentences:
    translation = translate(sentence)
    print(f"EN: {sentence}")
    print(f"DE: {translation}\n")