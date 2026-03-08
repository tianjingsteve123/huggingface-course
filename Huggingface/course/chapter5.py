"""Hugging Face NLP Course -- Chapter 5: The Datasets Library

Standalone executable script demonstrating dataset loading, processing, and evaluation.
Source: https://huggingface.co/learn/nlp-course/chapter5
"""

# ============================================================
# 第5章: Datasets库 / Chapter 5: The Datasets Library
# ============================================================
#
# 本章概要 / Chapter Summary:
#   本章系统介绍Hugging Face Datasets库的核心功能。你将学习使用
#   load_dataset()加载多种格式的数据集, 使用.map()进行批量分词和
#   特征工程, 使用.filter()筛选数据, 使用.shuffle()/.sort()调整
#   顺序, 以及.train_test_split()划分训练/测试集。本章还涵盖了
#   数据集与Pandas的互转、列操作、缓存机制和流式处理(streaming)。
#
#   This chapter systematically covers the Hugging Face Datasets library.
#   You will learn to use load_dataset() to load datasets in various
#   formats, .map() for batch tokenization and feature engineering,
#   .filter() for data selection, .shuffle()/.sort() for reordering,
#   and .train_test_split() for creating train/test splits. It also
#   covers Dataset-Pandas conversion, column operations, caching, and
#   streaming for large-scale data processing.
#
# 核心概念 / Key Concepts:
#   1. load_dataset -- 从Hub或本地加载数据集
#      / Load datasets from HF Hub or local files
#   2. .map() / .filter() -- 批量数据变换与筛选
#      / Batch data transformation and filtering
#   3. Streaming模式 -- 处理超大数据集, 不占用全部内存
#      / Process huge datasets without loading everything into RAM
#
# 模型架构 / Model Architecture:
#
#   load_dataset("squad")
#       |
#       v
#   +---------------------------+
#   | Dataset                   |
#   | (Arrow-backed, columnar)  |
#   +---------------------------+
#       |
#       +-- .map(tokenize_fn)  --> 批量分词 / Batch tokenize
#       +-- .filter(pred_fn)   --> 条件过滤 / Conditional filter
#       +-- .shuffle(seed=42)  --> 随机打乱 / Random shuffle
#       +-- .sort("col")       --> 按列排序 / Sort by column
#       +-- .select(range(N))  --> 取子集 / Select subset
#       |
#       v
#   +---------------------------+
#   | DataLoader                |  批量输入模型 / Feed to model
#   | (batch_size, collate_fn)  |
#   +---------------------------+
#
# 代码示例说明 / Code Examples in This File:
#   - 数据集加载与探索 (Load & explore SQuAD)           ~line 57-68
#   - 数据切片 (Dataset slicing with .select)           ~line 71-75
#   - 数据过滤 (Filtering with .filter)                 ~line 78-84
#   - 数据映射 (Mapping with .map to add features)      ~line 87-95
#   - 排序与打乱 (Sort & shuffle)                       ~line 98-108
#   - 训练/测试划分 (train_test_split)                  ~line 111-114
#   - 列操作 (Rename / remove columns)                  ~line 117-128
#   - Pandas互转 (to_pandas / from_pandas)              ~line 142-154
#   - 字典创建与拼接 (from_dict & concatenate)          ~line 157-179
#   - 迭代与批处理 (Iteration & batching)               ~line 188-199
#
# 下游应用 / Downstream Applications:
#   - 数据预处理 / Data Preprocessing (cleaning, tokenizing at scale)
#   - 数据增强 / Data Augmentation (mapping custom transformations)
#   - 大规模数据集处理 / Large-scale Processing (streaming mode)
#
# 课程链接 / Course Link:
#   https://huggingface.co/learn/nlp-course/chapter5
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
print("🚀 Chapter 5: The 🤗 Datasets library")
print("=" * 60)

QUICK_RUN = os.getenv("HF_QUICK_RUN", "1") == "1"
print(f"⚙️ HF_QUICK_RUN={int(QUICK_RUN)} (set HF_QUICK_RUN=0 for full chapter run)")

def run_quick_demo() -> None:
    """Run local Dataset API examples without downloading remote corpora."""
    print("\n⚡ Quick Run: local Dataset operations without remote download")
    from datasets import Dataset

    local_dataset = Dataset.from_dict(
        {
            "question": [
                "What is Transformers?",
                "What is tokenization?",
                "What is fine-tuning?"
            ],
            "answer": [
                "A model architecture for sequence tasks.",
                "Splitting text into model-readable units.",
                "Adapting a pretrained model to a target task."
            ]
        }
    )
    print("Local dataset length:", len(local_dataset))
    print("First row:", local_dataset[0])

    filtered = local_dataset.filter(lambda ex: len(ex["answer"]) > 30)
    print("Filtered length:", len(filtered))

    mapped = local_dataset.map(lambda ex: {"answer_length": len(ex["answer"])})
    print("Mapped features:", mapped.features)
    print("Mapped first row:", mapped[0])

    print("\n✅ Chapter 5 quick run completed successfully!")


if QUICK_RUN:
    run_quick_demo()
    raise SystemExit(0)

# === Dataset Loading ===
print("\n📊 1. Loading Dataset")
from datasets import load_dataset
# Use a different available dataset instead of SQuAD_it
dataset = load_dataset("squad")
print("Dataset loaded:", dataset)
print("Dataset features:", dataset["train"].features)

# === Dataset Exploration ===
print("\n🔍 2. Dataset Exploration")
print("Train dataset length:", len(dataset["train"]))
print("Validation dataset length:", len(dataset["validation"]))
print("First example:", dataset["train"][0])

# === Dataset Slicing ===
print("\n✂️ 3. Dataset Slicing")
# Take a small subset for demo
small_dataset = dataset["train"].select(range(100))
print("Small dataset length:", len(small_dataset))
print("First example from small dataset:", small_dataset[0])

# === Dataset Filtering ===
print("\n🔍 4. Dataset Filtering")
# Filter examples with short answers
def has_short_answer(example: dict[str, Any]) -> bool:
    """Return True when the first answer text is short enough for demo filtering."""
    return len(example["answers"]["text"][0]) < 50

filtered_dataset = small_dataset.filter(has_short_answer)
print("Filtered dataset length:", len(filtered_dataset))

# === Dataset Mapping ===
print("\n🔄 5. Dataset Mapping")
def add_length(example: dict[str, Any]) -> dict[str, Any]:
    """Add derived length features used in sorting and inspection examples."""
    example["question_length"] = len(example["question"])
    example["answer_length"] = len(example["answers"]["text"][0])
    return example

mapped_dataset = small_dataset.map(add_length)
print("Mapped dataset features:", mapped_dataset.features)
print("First mapped example:", mapped_dataset[0])

# === Dataset Sorting ===
print("\n📊 6. Dataset Sorting")
sorted_dataset = mapped_dataset.sort("question_length")
print("Dataset sorted by question length")
print("Shortest question:", sorted_dataset[0]["question"])
print("Longest question:", sorted_dataset[-1]["question"])

# === Dataset Shuffling ===
print("\n🔀 7. Dataset Shuffling")
shuffled_dataset = small_dataset.shuffle(seed=42)
print("Dataset shuffled")
print("First example after shuffling:", shuffled_dataset[0]["question"])

# === Dataset Splitting ===
print("\n✂️ 8. Dataset Splitting")
train_test = small_dataset.train_test_split(test_size=0.2, seed=42)
print("Train split length:", len(train_test["train"]))
print("Test split length:", len(train_test["test"]))

# === Dataset Renaming ===
print("\n🏷️ 9. Dataset Renaming")
renamed_dataset = small_dataset.rename_column("question", "query")
print("Column renamed: question -> query")
print("New features:", renamed_dataset.features)

# === Dataset Removal ===
print("\n🗑️ 10. Dataset Column Removal")
# Remove a column we don't need
columns_to_remove = ["id"]
cleaned_dataset = small_dataset.remove_columns(columns_to_remove)
print("Columns removed:", columns_to_remove)
print("Cleaned features:", cleaned_dataset.features)

# === Dataset Casting ===
print("\n🔄 11. Dataset Casting")
# Cast to different types if needed
print("Original answer type:", type(cleaned_dataset[0]["answers"]["text"][0]))

# === Dataset Flattening ===
print("\n📋 12. Dataset Flattening")
# Flatten nested structures
flattened_dataset = small_dataset.flatten()
print("Flattened features:", flattened_dataset.features)

# === Dataset to Pandas ===
print("\n🐼 13. Dataset to Pandas")
import pandas as pd
df = small_dataset.to_pandas()
print("Dataset converted to pandas DataFrame")
print("DataFrame shape:", df.shape)
print("DataFrame columns:", df.columns.tolist())

# === Dataset from Pandas ===
print("\n📊 14. Dataset from Pandas")
from datasets import Dataset
new_dataset = Dataset.from_pandas(df)
print("Dataset created from pandas DataFrame")
print("New dataset length:", len(new_dataset))

# === Dataset from Dictionary ===
print("\n📝 15. Dataset from Dictionary")
data_dict = {
    "text": ["Hello world", "How are you?", "Fine, thank you"],
    "label": [0, 1, 0]
}
dict_dataset = Dataset.from_dict(data_dict)
print("Dataset created from dictionary")
print("Dictionary dataset:", dict_dataset)

# === Dataset Concatenation ===
print("\n🔗 16. Dataset Concatenation")
from datasets import concatenate_datasets
# Create another small dataset
data_dict2 = {
    "text": ["Good morning", "Have a nice day"],
    "label": [1, 0]
}
dict_dataset2 = Dataset.from_dict(data_dict2)

# Concatenate datasets
concatenated_dataset = concatenate_datasets([dict_dataset, dict_dataset2])
print("Datasets concatenated")
print("Concatenated dataset length:", len(concatenated_dataset))

# === Dataset Metrics ===
print("\n📈 17. Dataset Metrics")
from evaluate import load
metric = load("accuracy")
print("Accuracy metric loaded:", metric)

# === Dataset Iteration ===
print("\n🔄 18. Dataset Iteration")
print("Iterating through first 3 examples:")
for i, example in enumerate(small_dataset):
    if i >= 3:
        break
    print(f"  Example {i+1}: {example['question'][:50]}...")

# === Dataset Batching ===
print("\n📦 19. Dataset Batching")
batch_size = 5
batched_dataset = small_dataset.map(lambda x: x, batched=True, batch_size=batch_size)
print(f"Dataset batched with batch size {batch_size}")

# === Dataset Caching ===
print("\n💾 20. Dataset Caching")
# Enable caching for faster subsequent runs
try:
    cached_dataset = small_dataset.map(add_length, cache_file_name="cached_dataset.arrow")
    print("Dataset cached for faster access")
except Exception as e:
    print(f"Caching failed (this is normal): {e}")
    cached_dataset = small_dataset.map(add_length)
    print("Dataset processed without caching")

print("\n" + "=" * 60)
print("✅ Chapter 5 completed successfully!")
print("=" * 60)
