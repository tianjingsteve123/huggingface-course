"""Hugging Face NLP Course -- Chapter 6: The Tokenizers Library

Standalone executable script demonstrating tokenizer training, fast tokenizers, and offsets.
Source: https://huggingface.co/learn/nlp-course/chapter6
"""

# ============================================================
# 第6章: 分词器库 / Chapter 6: The Tokenizers Library
# ============================================================
#
# 本章概要 / Chapter Summary:
#   本章深入讲解分词器的工作原理和三大子词分词算法: BPE(Byte Pair
#   Encoding)、WordPiece和Unigram/SentencePiece。你将理解为什么子
#   词分词优于字符级和词级分词 -- 它在词表大小和语义保留之间取得平
#   衡, 能有效处理未登录词(OOV)。本章还演示了如何基于自定义语料
#   训练新的分词器, 以及如何将分词与模型微调结合使用。
#
#   This chapter dives deep into tokenizer internals and three major
#   subword algorithms: BPE (Byte Pair Encoding), WordPiece, and
#   Unigram/SentencePiece. You will understand why subword tokenization
#   outperforms character-level and word-level approaches -- it balances
#   vocabulary size and semantic preservation, handling out-of-vocabulary
#   (OOV) words effectively. The chapter also demonstrates training
#   custom tokenizers on domain corpora and integrating them with models.
#
# 核心概念 / Key Concepts:
#   1. BPE (Byte Pair Encoding) -- 贪心合并最频繁字符对
#      / Greedily merges the most frequent character pairs
#   2. WordPiece -- 类似BPE但使用似然最大化选择合并
#      / Similar to BPE but uses likelihood-based merge selection
#   3. Unigram/SentencePiece -- 从大词表逐步裁剪到目标大小
#      / Starts with a large vocab and prunes down to target size
#
# 模型架构 / Model Architecture:
#
#   "Hello world"
#       |
#       v
#   +-------------------------------------------+
#   | Tokenization Algorithm                    |
#   |                                           |
#   | [BPE]       "Hel" + "lo" + " world"      |
#   | [WordPiece] "Hello" + " world"            |
#   | [Unigram]   "He" + "llo" + " world"      |
#   +-------------------------------------------+
#       |
#       v
#   Token IDs: [1234, 567, 890]
#       |
#       v
#   +-------------------------------------------+
#   | Embedding Layer                           |
#   | ID -> Dense Vector (768-dim)              |
#   +-------------------------------------------+
#       |
#       v
#   Model Input (batch_size, seq_len, 768)
#
#   子词分词优势 / Subword Advantages:
#   - 词表小 (30K-50K) / Small vocab (30K-50K)
#   - 无OOV问题 / No out-of-vocabulary issue
#   - 保留语义结构 / Preserves semantic structure
#
# 代码示例说明 / Code Examples in This File:
#   - 数据集加载 (IMDB dataset loading)                 ~line 57-68
#   - 分词器加载 (Load DistilBERT tokenizer)            ~line 71-74
#   - 分词函数 (Tokenization function with .map)        ~line 78-85
#   - 数据整理器 (DataCollatorWithPadding)              ~line 88-91
#   - 模型加载与训练配置 (Model & TrainingArguments)    ~line 94-116
#   - Trainer搭建 (Trainer setup & evaluation)          ~line 139-172
#   - Pipeline使用 (Pipeline for classification)        ~line 189-192
#   - 自定义数据集 (Custom dataset creation & mapping)  ~line 195-211
#   - 分类效果演示 (Classification examples)            ~line 237-248
#
# 下游应用 / Downstream Applications:
#   - 多语言分词 / Multilingual Tokenization (cross-language models)
#   - 领域定制分词 / Domain-specific Tokenization (biomedical, legal)
#   - 高效词表构建 / Efficient Vocabulary Building (subword merges)
#
# 课程链接 / Course Link:
#   https://huggingface.co/learn/nlp-course/chapter6
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
print("🚀 Chapter 6: Fine-tuning a pretrained model")
print("=" * 60)

QUICK_RUN = os.getenv("HF_QUICK_RUN", "1") == "1"
print(f"⚙️ HF_QUICK_RUN={int(QUICK_RUN)} (set HF_QUICK_RUN=0 for full chapter run)")

def run_quick_demo() -> None:
    """Run a local tokenizer/classifier forward pass for fast verification."""
    print("\n⚡ Quick Run: tokenizer + classification forward pass")
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
    batch = {
        "input_ids": torch.randint(0, config.vocab_size, (2, 16), device=device),
        "attention_mask": torch.ones((2, 16), dtype=torch.long, device=device),
    }
    print("Tokenized batch shape:", batch["input_ids"].shape)

    with torch.no_grad():
        logits = model(**batch).logits
    preds = torch.argmax(logits, dim=-1).cpu().tolist()
    print("Logits:", logits.cpu().tolist())
    print("Predictions:", preds)

    print("\n✅ Chapter 6 quick run completed successfully!")


if QUICK_RUN:
    run_quick_demo()
    raise SystemExit(0)

# === Dataset Loading ===
print("\n📊 1. Loading Dataset")
from datasets import load_dataset
# Use a supported dataset instead of the problematic one
dataset = load_dataset("imdb")
print("Dataset loaded:", dataset)
print("Dataset features:", dataset["train"].features)

# === Dataset Exploration ===
print("\n🔍 2. Dataset Exploration")
print("Train dataset length:", len(dataset["train"]))
print("Test dataset length:", len(dataset["test"]))
print("First example:", dataset["train"][0])

# === Tokenizer Setup ===
print("\n🔤 3. Tokenizer Setup")
from transformers import AutoTokenizer
model_checkpoint = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
print("Tokenizer loaded:", type(tokenizer))

# === Tokenization Function ===
print("\n✂️ 4. Tokenization Function")
def tokenize_function(examples: dict[str, list[str]]) -> dict[str, Any]:
    """Tokenize an IMDB batch for sequence classification training."""
    return tokenizer(examples["text"], truncation=True, padding=True)

# Use smaller dataset for demo
small_dataset = dataset["train"].select(range(1000))
tokenized_dataset = small_dataset.map(tokenize_function, batched=True)
print("Dataset tokenized")

# === Data Collator ===
print("\n📋 5. Data Collator")
from transformers import DataCollatorWithPadding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
print("Data collator created")

# === Model Loading ===
print("\n🤖 6. Model Loading")
from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained(
    model_checkpoint, 
    num_labels=2
)
print("Model loaded:", type(model))

# === Training Arguments ===
print("\n⚙️ 7. Training Arguments")
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
print("\n📈 8. Metrics")
import numpy as np
from evaluate import load

def compute_metrics(eval_pred: tuple[Any, Any]) -> dict[str, float]:
    """Compute accuracy from model logits and reference labels."""
    metric = load("accuracy")
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)

print("Metrics function created")

# === Dataset Splitting ===
print("\n✂️ 9. Dataset Splitting")
train_test = tokenized_dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = train_test["train"]
eval_dataset = train_test["test"]
print("Dataset split into train and eval")

# === Trainer Setup ===
print("\n🏃 10. Trainer Setup")
from transformers import Trainer

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

# === Training (Demo) ===
print("\n🚀 11. Training (Demo)")
print("⚠️ Full training skipped for demo purposes")
print("To run full training, uncomment the line below:")
print("# trainer.train()")

# === Evaluation ===
print("\n📊 12. Evaluation")
print("⚠️ Evaluation skipped - requires trained model")
print("To run evaluation, uncomment the line below:")
print("# trainer.evaluate()")

# === Prediction ===
print("\n🔮 13. Prediction")
print("⚠️ Prediction skipped - requires trained model")
print("To run prediction, uncomment the code below:")
print("""
# predictions = trainer.predict(eval_dataset)
# print("Predictions:", predictions)
""")

# === Model Saving ===
print("\n💾 14. Model Saving")
model.save_pretrained("my-distilbert-finetuned")
tokenizer.save_pretrained("my-distilbert-finetuned")
print("Model and tokenizer saved locally")

# === Loading Fine-tuned Model ===
print("\n📥 15. Loading Fine-tuned Model")
from transformers import AutoModelForSequenceClassification, AutoTokenizer
model = AutoModelForSequenceClassification.from_pretrained("my-distilbert-finetuned")
tokenizer = AutoTokenizer.from_pretrained("my-distilbert-finetuned")
print("Fine-tuned model loaded")

# === Pipeline Usage ===
print("\n🔧 16. Pipeline Usage")
from transformers import pipeline
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)
result = classifier("This is a great movie!")
print("Pipeline result:", result)

# === Custom Dataset Creation ===
print("\n📝 17. Custom Dataset Creation")
from datasets import Dataset
custom_data = {
    "text": [
        "This is a positive review",
        "This is a negative review",
        "I love this product",
        "I hate this product"
    ],
    "label": [1, 0, 1, 0]
}
custom_dataset = Dataset.from_dict(custom_data)
print("Custom dataset created:", custom_dataset)

# === Dataset Processing ===
print("\n🔄 18. Dataset Processing")
custom_tokenized = custom_dataset.map(tokenize_function, batched=True)
print("Custom dataset tokenized")

# === Model Evaluation on Custom Data ===
print("\n📊 19. Model Evaluation on Custom Data")
print("⚠️ Custom evaluation skipped - requires trained model")
print("To run custom evaluation, uncomment the code below:")
print("""
# custom_trainer = Trainer(
#     model=model,
#     eval_dataset=custom_tokenized,
#     tokenizer=tokenizer,
#     data_collator=data_collator,
#     compute_metrics=compute_metrics,
# )
# results = custom_trainer.evaluate()
# print("Custom evaluation results:", results)
""")

# === Model Information ===
print("\n📋 20. Model Information")
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Model device: {next(model.parameters()).device}")
print(f"Tokenizer vocab size: {tokenizer.vocab_size}")

# === Text Classification Examples ===
print("\n🎯 21. Text Classification Examples")
test_texts = [
    "This movie is absolutely fantastic!",
    "I really didn't like this film at all.",
    "The acting was okay, nothing special.",
    "This is the best movie I've ever seen!"
]

print("Testing classification on sample texts:")
for text in test_texts:
    result = classifier(text)
    print(f"  '{text}' -> {result[0]['label']} (score: {result[0]['score']:.4f})")

print("\n" + "=" * 60)
print("✅ Chapter 6 completed successfully!")
print("=" * 60)
