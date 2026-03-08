"""Hugging Face NLP Course -- Chapter 4: Sharing Models and Tokenizers

Standalone executable script covering the Model Hub, model/tokenizer saving and loading.
Source: https://huggingface.co/learn/nlp-course/chapter4
"""

# ============================================================
# 第4章: 共享模型与分词器 / Chapter 4: Sharing Models and Tokenizers
# ============================================================
#
# 本章概要 / Chapter Summary:
#   本章讲解如何保存、加载和共享模型与分词器。你将学习使用
#   save_pretrained()将模型保存到本地, 使用push_to_hub()将模型
#   上传到Hugging Face Hub, 以及如何撰写Model Card描述模型用途、
#   训练数据和局限性。模型版本控制(基于Git)确保团队协作中可以
#   追溯每一次模型迭代。
#
#   This chapter explains how to save, load, and share models and
#   tokenizers. You will learn to use save_pretrained() for local
#   storage, push_to_hub() to upload to the Hugging Face Hub, and
#   how to write Model Cards documenting model usage, training data,
#   and limitations. Git-based model versioning ensures every iteration
#   is traceable in team collaboration workflows.
#
# 核心概念 / Key Concepts:
#   1. save_pretrained / from_pretrained -- 本地保存与加载
#      / Local model persistence and loading
#   2. push_to_hub -- 一键上传到Hugging Face Hub
#      / One-command upload to the Hugging Face Hub
#   3. Model Card -- 模型文档卡, 描述用途与局限
#      / Documentation card describing usage, data, and limitations
#
# 模型架构 / Model Architecture:
#
#   +---------------------------+
#   | Local Model + Tokenizer   |
#   | save_pretrained("dir/")   |
#   +---------------------------+
#       |
#       v
#   push_to_hub("username/model")
#       |
#       v
#   +---------------------------+
#   | Hugging Face Hub          |  Git-based model repository
#   | - model weights           |  模型权重 + 配置 + 分词器
#   | - config.json             |
#   | - tokenizer files         |
#   | - README.md (Model Card)  |
#   +---------------------------+
#       |
#       v
#   from_pretrained("username/model")
#       |
#       v
#   +---------------------------+
#   | Anyone can download & use |  任何人可以下载使用
#   +---------------------------+
#
# 代码示例说明 / Code Examples in This File:
#   - 模型与分词器加载 (Load model & tokenizer)        ~line 52-61
#   - 掩码填充推理 (Masked LM inference)               ~line 64-86
#   - Top-5预测 (Top-5 mask predictions)               ~line 89-107
#   - 模型保存与本地加载 (Save & load locally)         ~line 110-127
#   - 自定义配置 (Custom model configuration)          ~line 130-158
#   - Hub功能说明 (Hub features, push_to_hub)          ~line 161-169
#   - 模型信息汇总 (Model parameter summary)           ~line 186-189
#
# 下游应用 / Downstream Applications:
#   - 模型共享 / Model Sharing (open-source community collaboration)
#   - 模型版本管理 / Model Versioning (tracking model iterations)
#   - 团队协作 / Team Collaboration (shared model repositories)
#
# 课程链接 / Course Link:
#   https://huggingface.co/learn/nlp-course/chapter4
# ============================================================

# Import necessary libraries
import warnings
warnings.filterwarnings("ignore")
import os
import tempfile

# Set device
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print("=" * 60)
print("🚀 Chapter 4: Sharing models and tokenizers")
print("=" * 60)

QUICK_RUN = os.getenv("HF_QUICK_RUN", "1") == "1"
print(f"⚙️ HF_QUICK_RUN={int(QUICK_RUN)} (set HF_QUICK_RUN=0 for full chapter run)")

def run_quick_demo() -> None:
    """Run a local model save/load roundtrip using a temporary directory."""
    print("\n⚡ Quick Run: local model save/load roundtrip without remote downloads")
    from transformers import BertConfig, BertForMaskedLM

    cfg = BertConfig(
        vocab_size=2000,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=128
    )
    model = BertForMaskedLM(cfg).to(device)
    input_ids = torch.randint(0, cfg.vocab_size, (1, 8), device=device)
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    print("Logits shape:", tuple(logits.shape))

    with tempfile.TemporaryDirectory(prefix="hf_ch4_quick_") as save_dir:
        model.save_pretrained(save_dir)
        reloaded = BertForMaskedLM.from_pretrained(save_dir).to(device)
        with torch.no_grad():
            reload_logits = reloaded(input_ids=input_ids, attention_mask=attention_mask).logits
    print("Reload logits shape:", tuple(reload_logits.shape))
    print("Roundtrip check:", reload_logits.shape == logits.shape)

    print("\n✅ Chapter 4 quick run completed successfully!")


if QUICK_RUN:
    run_quick_demo()
    raise SystemExit(0)

# === Model Loading ===
print("\n🤖 1. Loading Model")
from transformers import AutoModelForMaskedLM
model = AutoModelForMaskedLM.from_pretrained("camembert-base")
print("Model loaded:", type(model))

# === Tokenizer Loading ===
print("\n🔤 2. Loading Tokenizer")
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("camembert-base")
print("Tokenizer loaded:", type(tokenizer))

# === Text Processing ===
print("\n📝 3. Text Processing")
text = "This is a great <mask>."
inputs = tokenizer(text, return_tensors="pt")
print("Inputs:", inputs)

# === Model Inference ===
print("\n🧠 4. Model Inference")
with torch.no_grad():
    logits = model(**inputs).logits
print("Logits shape:", logits.shape)

# === Mask Prediction ===
print("\n🎭 5. Mask Prediction")
# Find the mask token position
mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
if len(mask_token_index) > 0:
    predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
    predicted_token = tokenizer.decode(predicted_token_id)
    print(f"Original text: {text}")
    print(f"Predicted token: {predicted_token}")
else:
    print("No mask token found in input")
    predicted_token = ""

# === Multiple Predictions ===
print("\n🔮 6. Multiple Predictions")
import torch.nn.functional as F

if len(mask_token_index) > 0:
    # Get top 5 predictions
    mask_token_logits = logits[0, mask_token_index, :]
    top_5_tokens = torch.topk(mask_token_logits, 5, dim=-1).indices[0].tolist()
else:
    print("No mask token found for predictions")
    top_5_tokens = []

print("Top 5 predictions:")
if len(top_5_tokens) > 0:
    for token_id in top_5_tokens:
        token = tokenizer.decode([token_id])
        score = F.softmax(mask_token_logits, dim=-1)[0, token_id].item()
        print(f"  {token}: {score:.4f}")
else:
    print("  No predictions available")

# === Model Saving ===
print("\n💾 7. Model Saving")
model.save_pretrained("my-camembert-model")
tokenizer.save_pretrained("my-camembert-model")
print("Model and tokenizer saved locally")

# === Loading from Local ===
print("\n📥 8. Loading from Local")
from transformers import AutoModelForMaskedLM, AutoTokenizer
local_model = AutoModelForMaskedLM.from_pretrained("my-camembert-model")
local_tokenizer = AutoTokenizer.from_pretrained("my-camembert-model")
print("Model and tokenizer loaded from local directory")

# === Pipeline Usage ===
print("\n🔧 9. Pipeline Usage")
from transformers import pipeline
unmasker = pipeline("fill-mask", model=local_model, tokenizer=local_tokenizer)
result = unmasker("This is a great <mask>.")
print("Pipeline result:", result)

# === Model Configuration ===
print("\n⚙️ 10. Model Configuration")
from transformers import AutoConfig
config = AutoConfig.from_pretrained("camembert-base")
print("Model configuration:", config)

# === Custom Configuration ===
print("\n🎛️ 11. Custom Configuration")
from transformers import CamembertConfig
custom_config = CamembertConfig(
    vocab_size=32005,
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=3072,
    max_position_embeddings=514,
)
print("Custom configuration created:", custom_config)

# === Model from Configuration ===
print("\n🏗️ 12. Model from Configuration")
from transformers import CamembertForMaskedLM
custom_model = CamembertForMaskedLM(custom_config)
print("Custom model created:", type(custom_model))

# === Tokenizer Configuration ===
print("\n🔧 13. Tokenizer Configuration")
from transformers import CamembertTokenizer
custom_tokenizer = CamembertTokenizer.from_pretrained("camembert-base")
print("Custom tokenizer created:", type(custom_tokenizer))

# === Model Hub (Skipped) ===
print("\n🌐 14. Model Hub")
print("⚠️ Model Hub features skipped - requires authentication")
print("To use Hub features, login and uncomment:")
print("""
# from huggingface_hub import notebook_login
# notebook_login()
# model.push_to_hub("my-awesome-camembert")
# model = AutoModelForMaskedLM.from_pretrained("username/my-awesome-camembert")
""")

# === Model Evaluation ===
print("\n📊 15. Model Evaluation")
# Simple evaluation on a test sentence
test_sentences = [
    "This is a <mask> example.",
    "The weather is <mask> today.",
    "I love <mask> programming."
]

print("Testing model on sample sentences:")
for sentence in test_sentences:
    result = unmasker(sentence)
    print(f"  '{sentence}' -> '{result[0]['token_str']}' (score: {result[0]['score']:.4f})")

# === Model Information ===
print("\n📋 16. Model Information")
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Model device: {next(model.parameters()).device}")
print(f"Tokenizer vocab size: {tokenizer.vocab_size}")

print("\n" + "=" * 60)
print("✅ Chapter 4 completed successfully!")
print("=" * 60)
