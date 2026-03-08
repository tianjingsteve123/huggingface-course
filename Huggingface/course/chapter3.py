"""Hugging Face NLP Course -- Chapter 3: Fine-Tuning a Pretrained Model

Standalone executable script demonstrating dataset loading, tokenization, and Trainer API.
Source: https://huggingface.co/learn/nlp-course/chapter3
"""

# ============================================================
# 第3章: 微调预训练模型 / Chapter 3: Fine-tuning a Pretrained Model
# ============================================================
#
# 本章概要 / Chapter Summary:
#   本章介绍如何使用Trainer API对预训练模型(如BERT)进行微调。
#   你将学习完整的微调流程: 加载GLUE/MRPC数据集、对句子对进行分词、
#   配置TrainingArguments(学习率、epochs、batch size)、构建Trainer
#   并运行训练循环, 最后使用evaluate库计算准确率和F1等评估指标。
#   这是将通用预训练模型适配到特定下游任务的核心技能。
#
#   This chapter teaches how to fine-tune pretrained models (e.g. BERT)
#   using the Trainer API. You will learn the full fine-tuning workflow:
#   loading the GLUE/MRPC dataset, tokenizing sentence pairs, configuring
#   TrainingArguments (learning rate, epochs, batch size), building a
#   Trainer, running the training loop, and computing evaluation metrics
#   (accuracy, F1) with the evaluate library. This is the core skill for
#   adapting general pretrained models to specific downstream tasks.
#
# 核心概念 / Key Concepts:
#   1. Trainer API -- 封装训练循环的高级接口
#      / High-level API wrapping the training loop
#   2. TrainingArguments -- 控制学习率、epochs、batch size等超参数
#      / Controls hyperparameters: lr, epochs, batch size
#   3. DataCollator -- 动态填充, 高效组批
#      / Dynamic padding for efficient batching
#
# 模型架构 / Model Architecture:
#
#   +---------------------------+
#   | Pretrained BERT           |  预训练BERT模型
#   | (bert-base-uncased)       |  110M parameters
#   +---------------------------+
#       |
#       v
#   +---------------------------+
#   | + ClassificationHead      |  添加分类头 (2 labels)
#   | (Linear 768 -> 2)        |
#   +---------------------------+
#       |
#       v
#   +---------------------------+
#   | Train on MRPC dataset     |  在MRPC数据集上训练
#   | Trainer(                  |  learning_rate=5e-5
#   |   epochs=1,               |  batch_size=8
#   |   eval_strategy="steps")  |
#   +---------------------------+
#       |
#       v
#   +---------------------------+
#   | Fine-tuned Model          |  微调后的模型 -> 保存部署
#   | save_pretrained()         |
#   +---------------------------+
#
# 代码示例说明 / Code Examples in This File:
#   - 数据集加载与探索 (Dataset loading & exploration)    ~line 55-65
#   - 分词器设置 (Tokenizer setup)                        ~line 68-83
#   - 句子对分词 (Pair tokenization)                      ~line 86-106
#   - 数据集处理 (Dataset .map processing)                ~line 109-116
#   - 数据整理器 (DataCollatorWithPadding)                ~line 119-122
#   - Trainer配置与训练 (Trainer setup & training)        ~line 125-183
#   - 模型保存与加载 (Model save & pipeline usage)        ~line 200-217
#
# 下游应用 / Downstream Applications:
#   - 句子对分类 / Sentence Pair Classification (paraphrase detection)
#   - 领域情感微调 / Domain Sentiment Fine-tuning (product reviews)
#   - 自然语言推理 / Natural Language Inference (textual entailment)
#
# 课程链接 / Course Link:
#   https://huggingface.co/learn/nlp-course/chapter3
# ============================================================

# Import necessary libraries
import warnings
warnings.filterwarnings("ignore")
import os
from typing import Any

# Set device
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print("=" * 60)
print("🚀 Chapter 3: Fine-tuning a pretrained model")
print("=" * 60)

QUICK_RUN = os.getenv("HF_QUICK_RUN", "1") == "1"
print(f"⚙️ HF_QUICK_RUN={int(QUICK_RUN)} (set HF_QUICK_RUN=0 for full chapter run)")

def run_quick_demo() -> None:
    """Run a compact fine-tuning preview without network-heavy downloads."""
    print("\n⚡ Quick Run: local mini fine-tuning workflow preview")
    from transformers import DistilBertConfig, DistilBertForSequenceClassification

    config = DistilBertConfig(
        vocab_size=30522,
        n_layers=2,
        dim=128,
        hidden_dim=256,
        n_heads=4,
        num_labels=2
    )
    model = DistilBertForSequenceClassification(config).to(device)

    encoded = {
        "input_ids": torch.randint(0, config.vocab_size, (2, 14), device=device),
        "attention_mask": torch.ones((2, 14), dtype=torch.long, device=device),
    }
    print("Batch size:", encoded["input_ids"].shape[0])
    print("Sequence length:", encoded["input_ids"].shape[1])

    with torch.no_grad():
        outputs = model(**encoded)
    print("Logits shape:", outputs.logits.shape)
    print("Predicted labels:", torch.argmax(outputs.logits, dim=-1).cpu().tolist())

    print("\n✅ Chapter 3 quick run completed successfully!")


if QUICK_RUN:
    run_quick_demo()
    raise SystemExit(0)

# === Dataset Loading ===
print("\n📊 1. Loading Dataset")
from datasets import load_dataset
raw_datasets = load_dataset("glue", "mrpc")
print("Dataset loaded:", raw_datasets)
print("Dataset features:", raw_datasets["train"].features)

# === Dataset Exploration ===
print("\n🔍 2. Dataset Exploration")
raw_train_dataset = raw_datasets["train"]
print("First training example:", raw_train_dataset[0])
print("Dataset length:", len(raw_train_dataset))

# === Tokenizer Setup ===
print("\n🔤 3. Tokenizer Setup")
from transformers import AutoTokenizer
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
print("Tokenizer loaded:", type(tokenizer))

# === Tokenization ===
print("\n✂️ 4. Tokenization")
# Fix: Convert to list to avoid type error
sentence1_list = raw_datasets["train"]["sentence1"][:5]  # Take first 5 examples
sentence2_list = raw_datasets["train"]["sentence2"][:5]

tokenized_sentences_1 = tokenizer(sentence1_list)
tokenized_sentences_2 = tokenizer(sentence2_list)
print("Tokenized sentences 1:", tokenized_sentences_1)
print("Tokenized sentences 2:", tokenized_sentences_2)

# === Pair Tokenization ===
print("\n🔗 5. Pair Tokenization")
inputs = tokenizer("This is the first sentence.", "This is the second one.")
print("Pair tokenization result:", inputs)
print("Token IDs:", tokenizer.convert_ids_to_tokens(inputs["input_ids"]))

# === Batch Tokenization ===
print("\n📦 6. Batch Tokenization")
# Use smaller batch to avoid memory issues
small_train_dataset = raw_datasets["train"].select(range(100))
# Convert to lists for tokenization
sentence1_list = [item for item in small_train_dataset["sentence1"]]
sentence2_list = [item for item in small_train_dataset["sentence2"]]
tokenized_dataset = tokenizer(
    sentence1_list,
    sentence2_list,
    padding=True,
    truncation=True,
    return_tensors="pt"
)
print("Batch tokenization completed")
print("Input IDs shape:", tokenized_dataset["input_ids"].shape)

# === Dataset Processing ===
print("\n🔄 7. Dataset Processing")
def tokenize_function(examples: dict[str, list[str]]) -> dict[str, Any]:
    """Tokenize sentence pairs from the MRPC dataset batch."""
    return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, padding=True)

# Process smaller dataset
small_dataset = raw_datasets["train"].select(range(1000))
tokenized_datasets = small_dataset.map(tokenize_function, batched=True)
print("Dataset processing completed")

# === Data Collation ===
print("\n📋 8. Data Collation")
from transformers import DataCollatorWithPadding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
print("Data collator created:", type(data_collator))

# === Model Loading ===
print("\n🤖 9. Model Loading")
from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
print("Model loaded:", type(model))

# === Training Arguments ===
print("\n⚙️ 10. Training Arguments")
from transformers import TrainingArguments
training_args = TrainingArguments(
    output_dir="test-trainer",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=1,
    eval_strategy="steps",
    eval_steps=100,
    save_steps=100,
    logging_steps=10,
    remove_unused_columns=False,
)
print("Training arguments created")

# === Metrics ===
print("\n📈 11. Metrics")
import numpy as np
from evaluate import load

def compute_metrics(eval_pred: tuple[Any, Any]) -> dict[str, float]:
    """Compute MRPC evaluation metrics from Trainer predictions."""
    metric = load("glue", "mrpc")
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)

print("Metrics function created")

# === Trainer Setup ===
print("\n🏃 12. Trainer Setup")
from transformers import Trainer

# Use smaller datasets for demo
train_dataset = tokenized_datasets.select(range(100))
eval_dataset = tokenized_datasets.select(range(100, 150))

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)
print("Trainer created")

# === Training (Optional) ===
print("\n🚀 13. Training (Demo)")
print("⚠️ Full training skipped for demo purposes")
print("To run full training, uncomment the line below:")
print("# trainer.train()")

# === Evaluation ===
print("\n📊 14. Evaluation")
print("⚠️ Evaluation skipped - requires trained model")
print("To run evaluation, uncomment the line below:")
print("# trainer.evaluate()")

# === Prediction ===
print("\n🔮 15. Prediction")
print("⚠️ Prediction skipped - requires trained model")
print("To run prediction, uncomment the code below:")
print("""
# predictions = trainer.predict(eval_dataset)
# print("Predictions:", predictions)
""")

# === Model Saving ===
print("\n💾 16. Model Saving")
model.save_pretrained("my-bert-finetuned")
tokenizer.save_pretrained("my-bert-finetuned")
print("Model and tokenizer saved locally")

# === Loading Fine-tuned Model ===
print("\n📥 17. Loading Fine-tuned Model")
from transformers import AutoModelForSequenceClassification, AutoTokenizer
model = AutoModelForSequenceClassification.from_pretrained("my-bert-finetuned")
tokenizer = AutoTokenizer.from_pretrained("my-bert-finetuned")
print("Fine-tuned model loaded")

# === Pipeline Usage ===
print("\n🔧 18. Pipeline Usage")
from transformers import pipeline
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)
result = classifier("This is a great example of fine-tuning!")
print("Pipeline result:", result)

print("\n" + "=" * 60)
print("✅ Chapter 3 completed successfully!")
print("=" * 60)
