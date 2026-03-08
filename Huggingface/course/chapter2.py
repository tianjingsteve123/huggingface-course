"""Hugging Face NLP Course -- Chapter 2: Behind the Pipeline

Standalone executable script dissecting tokenizer, model, and post-processing.
Source: https://huggingface.co/learn/nlp-course/chapter2
"""

# ============================================================
# 第2章: Pipeline的内部机制 / Chapter 2: Behind the Pipeline
# ============================================================
#
# 本章概要 / Chapter Summary:
#   本章深入拆解pipeline内部的三大组件: Tokenizer、Model和
#   Post-processing。你将学会如何手动调用AutoTokenizer将文本
#   转化为input_ids和attention_mask, 再通过AutoModel进行前向传播
#   得到logits, 最后用Softmax将logits转换为概率和标签。理解这些
#   底层机制将让你具备自定义推理流程和调试模型的能力。
#
#   This chapter dissects the three core components inside a pipeline:
#   Tokenizer, Model, and Post-processing. You will learn to manually
#   call AutoTokenizer to convert text into input_ids and attention_mask,
#   run a forward pass through AutoModel to obtain logits, and apply
#   Softmax to turn logits into probabilities and labels. Understanding
#   these internals empowers you to customize inference and debug models.
#
# 核心概念 / Key Concepts:
#   1. AutoTokenizer -- 自动选择匹配模型的分词器
#      / Auto-selects the correct tokenizer for a given model
#   2. AutoModel / AutoModelForXxx -- 加载预训练权重, 输出隐藏状态或logits
#      / Loads pretrained weights, outputs hidden states or logits
#   3. 后处理 -- Softmax将logits转化为概率, argmax选出标签
#      / Post-processing: Softmax converts logits to probabilities
#
# 模型架构 / Model Architecture:
#
#   Raw Text
#       |
#       v
#   +---------------------------+
#   | AutoTokenizer             |  自动分词器
#   | .from_pretrained(name)    |
#   +---------------------------+
#       |  input_ids, attention_mask
#       v
#   +---------------------------+
#   | AutoModel (BERT 12-layer) |  Transformer编码器
#   |   Embedding -> Self-Attn  |  词嵌入 -> 自注意力
#   |   -> FFN -> LayerNorm x12 |  -> 前馈 -> 层归一化 x12
#   +---------------------------+
#       |  logits (raw scores)
#       v
#   +---------------------------+
#   | Softmax -> argmax         |  后处理: 概率 -> 标签
#   +---------------------------+
#       |
#       v
#   Labels: POSITIVE / NEGATIVE
#
# 代码示例说明 / Code Examples in This File:
#   - Pipeline直接使用 (Pipeline usage)             ~line 118-125
#   - 手动加载模型 (Manual model loading)            ~line 128-135
#   - 分词器编码/解码 (Tokenizer encode/decode)      ~line 154-161
#   - 批量分词 (Batch tokenization with padding)     ~line 164-174
#   - 模型推理 (Model inference, hidden states)      ~line 177-182
#   - 序列分类模型 (Sequence classification model)   ~line 185-189
#   - 模型配置 (Model configuration & custom init)   ~line 202-227
#   - 模型保存与加载 (Save/load pretrained)          ~line 230-250
#
# 下游应用 / Downstream Applications:
#   - 文本分类 / Text Classification (spam detection, topic labeling)
#   - 特征提取 / Feature Extraction (semantic search, clustering)
#   - 模型调试 / Model Debugging (inspecting hidden states and logits)
#
# 课程链接 / Course Link:
#   https://huggingface.co/learn/nlp-course/chapter2
# ============================================================

# 导入必要的库 (Import necessary libraries)
import warnings
warnings.filterwarnings("ignore")
import os

# 设置计算设备 (Set compute device - GPU if available, else CPU)
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print("=" * 60)
print("🚀 Chapter 2: Using Transformers")
print("=" * 60)

QUICK_RUN = os.getenv("HF_QUICK_RUN", "1") == "1"
print(f"⚙️ HF_QUICK_RUN={int(QUICK_RUN)} (set HF_QUICK_RUN=0 for full chapter run)")

def run_quick_demo() -> None:
    """Run a compact tokenizer/model forward-pass demo without network access."""
    print("\n⚡ Quick Run: inspect tokenizer + model forward pass")
    from transformers import DistilBertConfig, DistilBertForSequenceClassification

    quick_config = DistilBertConfig(
        vocab_size=30522,
        n_layers=2,
        dim=128,
        hidden_dim=256,
        n_heads=4,
        num_labels=2
    )
    quick_model = DistilBertForSequenceClassification(quick_config).to(device)
    quick_input_ids = torch.randint(0, quick_config.vocab_size, (1, 10), device=device)
    quick_attention_mask = torch.ones_like(quick_input_ids)
    print("Tokenized keys:", ["input_ids", "attention_mask"])
    print("Input shape:", quick_input_ids.shape)

    with torch.no_grad():
        quick_outputs = quick_model(input_ids=quick_input_ids, attention_mask=quick_attention_mask)
    print("Logits shape:", quick_outputs.logits.shape)
    print("Logits:", quick_outputs.logits.cpu().tolist())

    print("\n✅ Chapter 2 quick run completed successfully!")


if QUICK_RUN:
    run_quick_demo()
    raise SystemExit(0)

# === Pipeline Usage ===
print("\n📊 1. Using Pipelines")
from transformers import pipeline
classifier = pipeline("sentiment-analysis")
result = classifier([
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!",
])
print("Pipeline result:", result)

# === Models ===
print("\n🤖 2. Working with Models")
from transformers import BertModel
model = BertModel.from_pretrained("bert-base-cased")
print("Model loaded:", type(model))

# Save model locally
model.save_pretrained("local_bert_model")
print("Model saved locally")

# Load model from local directory
from transformers import AutoModel
local_model = AutoModel.from_pretrained("local_bert_model")
print("Model loaded from local directory:", type(local_model))

# === Hub Authentication (Skipped) ===
print("\n🔐 3. Hub Authentication")
print("⚠️ Hub authentication skipped - requires Hugging Face token")
print("To enable Hub features, login with: huggingface_hub.login()")
print("Then uncomment the code below:")
print("""
# from huggingface_hub import notebook_login
# notebook_login()
# model.push_to_hub("my-awesome-model")
""")

# === Tokenizers ===
print("\n🔤 4. Working with Tokenizers")
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
encoded_input = tokenizer("Hello, I'm a single sentence!")
print("Encoded input:", encoded_input)

decoded_text = tokenizer.decode(encoded_input["input_ids"])
print("Decoded text:", decoded_text)

# === Tokenizer with Multiple Sentences ===
print("\n📝 5. Tokenizing Multiple Sentences")
batch_sentences = [
    "But what about second breakfast?",
    "Don't think he knows about second breakfast, Pip.",
]
encoded_inputs = tokenizer(batch_sentences, padding=True, truncation=True, return_tensors="pt")
print("Batch encoded inputs:", encoded_inputs)

# Move to device
encoded_inputs = {k: v.to(device) for k, v in encoded_inputs.items()}
print("Moved to device:", device)

# === Model Inference ===
print("\n🧠 6. Model Inference")
model = model.to(device)  # Move model to device
with torch.no_grad():
    model_outputs = model(**encoded_inputs)
print("Model outputs keys:", model_outputs.keys())
print("Last hidden state shape:", model_outputs.last_hidden_state.shape)

# === Different Model Types ===
print("\n🏗️ 7. Different Model Types")
from transformers import AutoModelForSequenceClassification
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
pt_model = AutoModelForSequenceClassification.from_pretrained(model_name)
print("Sequence classification model loaded:", type(pt_model))

# === Tokenizer Configuration ===
print("\n⚙️ 8. Tokenizer Configuration")
from transformers import AutoTokenizer
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
pt_model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

inputs = tokenizer("We are very happy to show you the 🤗 Transformers library.")
print("Tokenizer inputs:", inputs)

# === Model Configuration ===
print("\n🔧 9. Model Configuration")
from transformers import AutoConfig
config = AutoConfig.from_pretrained("bert-base-cased")
print("Model config:", config)

# === Custom Configuration ===
print("\n🎛️ 10. Custom Configuration")
from transformers import BertConfig
config = BertConfig(
    vocab_size=30_522,
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=3072,
)
print("Custom config created:", config)

# === Model from Configuration ===
print("\n🏗️ 11. Model from Configuration")
from transformers import BertModel
model = BertModel(config)
print("Model from config created:", type(model))

# === Model Parameters ===
print("\n📊 12. Model Parameters")
print("Model parameters:", model.num_parameters())

# === Model Loading ===
print("\n📥 13. Model Loading")
from transformers import BertModel
model = BertModel.from_pretrained("bert-base-cased")
print("Model loaded from pretrained:", type(model))

# === Model Saving ===
print("\n💾 14. Model Saving")
model.save_pretrained("my-bert-model")
print("Model saved to local directory")

# === Tokenizer Saving ===
print("\n💾 15. Tokenizer Saving")
tokenizer.save_pretrained("my-bert-model")
print("Tokenizer saved to local directory")

# === Loading from Local ===
print("\n📂 16. Loading from Local Directory")
from transformers import AutoModel, AutoTokenizer
model = AutoModel.from_pretrained("my-bert-model")
tokenizer = AutoTokenizer.from_pretrained("my-bert-model")
print("Model and tokenizer loaded from local directory")

# === Model Hub (Skipped) ===
print("\n🌐 17. Model Hub")
print("⚠️ Model Hub features skipped - requires authentication")
print("To use Hub features, login and uncomment:")
print("""
# model.push_to_hub("my-awesome-model")
# model = AutoModel.from_pretrained("username/my-awesome-model")
""")

print("\n" + "=" * 60)
print("✅ Chapter 2 completed successfully!")
print("=" * 60)
