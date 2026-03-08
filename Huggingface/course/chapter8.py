"""Hugging Face NLP Course -- Chapter 8: How to Ask for Help

Standalone executable script demonstrating debugging techniques and error handling.
Source: https://huggingface.co/learn/nlp-course/chapter8
"""

# ============================================================
# 第8章: 调试训练过程 / Chapter 8: Debugging Training
# ============================================================
#
# 本章概要 / Chapter Summary:
#   本章讲解训练过程中常见的错误和系统化调试方法。你将学习一套
#   结构化的调试流程: 首先检查数据(输入格式、标签对齐)、然后
#   检查模型(配置匹配、权重加载)、接着检查训练超参数(学习率、
#   batch size)、最后检查环境(GPU驱动、库版本兼容性)。本章还
#   教你如何在Hugging Face论坛上撰写有效的问题报告以获得社区帮助。
#
#   This chapter covers common training errors and systematic debugging.
#   You will learn a structured debugging workflow: first check data
#   (input format, label alignment), then check the model (config match,
#   weight loading), next check training hyperparameters (learning rate,
#   batch size), and finally check the environment (GPU drivers, library
#   version compatibility). It also teaches how to write effective bug
#   reports on the Hugging Face forum to get community help.
#
# 核心概念 / Key Concepts:
#   1. 数据检查 -- 验证input_ids形状、标签范围、特殊token
#      / Data check: verify input_ids shape, label range, special tokens
#   2. 模型检查 -- 确认num_labels、权重初始化、设备一致性
#      / Model check: confirm num_labels, weight init, device consistency
#   3. 训练检查 -- 学习率量级、梯度裁剪、loss变化趋势
#      / Training check: learning rate scale, gradient clipping, loss trend
#
# 模型架构 / Model Architecture:
#
#   Error occurred!
#       |
#       v
#   +---------------------------+
#   | 1. Check Data             |  数据是否正确?
#   |    - input_ids shape?     |  输入形状是否匹配?
#   |    - labels in range?     |  标签范围是否合法?
#   |    - special tokens?      |  特殊token是否正确?
#   +---------------------------+
#       |
#       v
#   +---------------------------+
#   | 2. Check Model            |  模型是否配置正确?
#   |    - num_labels match?    |  标签数是否匹配?
#   |    - weights loaded?      |  权重是否加载?
#   |    - correct device?      |  设备是否一致?
#   +---------------------------+
#       |
#       v
#   +---------------------------+
#   | 3. Check Training         |  训练参数是否合理?
#   |    - learning_rate?       |  学习率量级?
#   |    - batch_size?          |  批大小是否合适?
#   |    - loss decreasing?     |  损失是否下降?
#   +---------------------------+
#       |
#       v
#   +---------------------------+
#   | 4. Check Environment      |  环境是否兼容?
#   |    - CUDA / drivers?      |  GPU驱动版本?
#   |    - library versions?    |  库版本兼容性?
#   +---------------------------+
#       |
#       v
#   Fix & Retry  -->  修复后重新训练
#
# 代码示例说明 / Code Examples in This File:
#   - 模型与分词器加载 (Load model & tokenizer)        ~line 62-67
#   - 模型配置查看 (Model configuration inspection)    ~line 70-86
#   - 自定义配置创建 (Custom config creation)          ~line 76-91
#   - 模型保存与加载 (Save & load locally)             ~line 100-109
#   - 模型推理测试 (Inference test with mask)          ~line 112-129
#   - Pipeline验证 (Pipeline validation)               ~line 132-136
#   - 模型导出 (Model export to directory)             ~line 166-181
#   - 一致性验证 (Result consistency check)            ~line 184-193
#   - 性能测试 (Inference speed benchmarking)          ~line 217-228
#
# 下游应用 / Downstream Applications:
#   - 训练错误排查 / Training Error Diagnosis (loss NaN, OOM, convergence)
#   - 模型验证 / Model Validation (sanity checks before full training)
#   - 社区求助 / Community Help (writing effective bug reports)
#
# 课程链接 / Course Link:
#   https://huggingface.co/learn/nlp-course/chapter8
# ============================================================

# Import necessary libraries
import warnings
warnings.filterwarnings("ignore")
import os

# Set device
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print("=" * 60)
print("🚀 Chapter 8: How to share models and tokenizers")
print("=" * 60)

QUICK_RUN = os.getenv("HF_QUICK_RUN", "1") == "1"
print(f"⚙️ HF_QUICK_RUN={int(QUICK_RUN)} (set HF_QUICK_RUN=0 for full chapter run)")

def run_quick_demo() -> None:
    """Run core debugging checks (shape and NaN) on a tiny local model."""
    print("\n⚡ Quick Run: debugging workflow on a tiny local model")
    from transformers import BertConfig, BertForMaskedLM

    config = BertConfig(
        vocab_size=2500,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=128
    )
    model = BertForMaskedLM(config).to(device)
    batch = torch.randint(0, config.vocab_size, (2, 10), device=device)
    mask = torch.ones_like(batch)

    with torch.no_grad():
        logits = model(input_ids=batch, attention_mask=mask).logits
    print("Batch shape:", tuple(batch.shape))
    print("Logits shape:", tuple(logits.shape))
    print("Debug check (no NaN):", not torch.isnan(logits).any().item())

    print("\n✅ Chapter 8 quick run completed successfully!")


if QUICK_RUN:
    run_quick_demo()
    raise SystemExit(0)

# === Model Loading ===
print("\n🤖 1. Loading Model")
from transformers import AutoModelForMaskedLM, AutoTokenizer
model_name = "camembert-base"
model = AutoModelForMaskedLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
print("Model and tokenizer loaded:", type(model), type(tokenizer))

# === Model Configuration ===
print("\n⚙️ 2. Model Configuration")
from transformers import AutoConfig
config = AutoConfig.from_pretrained(model_name)
print("Model configuration:", config)

# === Custom Configuration ===
print("\n🎛️ 3. Custom Configuration")
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
print("\n🏗️ 4. Model from Configuration")
from transformers import CamembertForMaskedLM
custom_model = CamembertForMaskedLM(custom_config)
print("Custom model created:", type(custom_model))

# === Model Parameters ===
print("\n📊 5. Model Parameters")
print(f"Custom model parameters: {sum(p.numel() for p in custom_model.parameters()):,}")
print(f"Original model parameters: {sum(p.numel() for p in model.parameters()):,}")

# === Model Saving ===
print("\n💾 6. Model Saving")
model.save_pretrained("my-camembert-model")
tokenizer.save_pretrained("my-camembert-model")
print("Model and tokenizer saved locally")

# === Loading from Local ===
print("\n📥 7. Loading from Local")
local_model = AutoModelForMaskedLM.from_pretrained("my-camembert-model")
local_tokenizer = AutoTokenizer.from_pretrained("my-camembert-model")
print("Model and tokenizer loaded from local directory")

# === Model Testing ===
print("\n🧪 8. Model Testing")
text = "This is a great <mask>."
inputs = local_tokenizer(text, return_tensors="pt")
print("Input text:", text)
print("Tokenized inputs:", inputs)

# === Model Inference ===
print("\n🧠 9. Model Inference")
with torch.no_grad():
    logits = local_model(**inputs).logits
print("Logits shape:", logits.shape)

# === Mask Prediction ===
print("\n🎭 10. Mask Prediction")
mask_token_index = (inputs.input_ids == local_tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
predicted_token = local_tokenizer.decode(predicted_token_id)
print(f"Predicted token: {predicted_token}")

# === Pipeline Usage ===
print("\n🔧 11. Pipeline Usage")
from transformers import pipeline
unmasker = pipeline("fill-mask", model=local_model, tokenizer=local_tokenizer)
result = unmasker("This is a great <mask>.")
print("Pipeline result:", result)

# === Model Information ===
print("\n📋 12. Model Information")
print(f"Model device: {next(local_model.parameters()).device}")
print(f"Tokenizer vocab size: {local_tokenizer.vocab_size}")
print(f"Model config: {local_model.config}")

# === Tokenizer Testing ===
print("\n🔤 13. Tokenizer Testing")
test_texts = [
    "This is a <mask> example.",
    "The weather is <mask> today.",
    "I love <mask> programming."
]

print("Testing unmasking on sample texts:")
for text in test_texts:
    result = unmasker(text)
    print(f"  '{text}' -> '{result[0]['token_str']}' (score: {result[0]['score']:.4f})")

# === Model Comparison ===
print("\n📊 14. Model Comparison")
print("Comparing original and custom models:")
print(f"Original model type: {type(model)}")
print(f"Custom model type: {type(custom_model)}")
print(f"Original model parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Custom model parameters: {sum(p.numel() for p in custom_model.parameters()):,}")

# === Model Export ===
print("\n📤 15. Model Export")
# Export model to different formats
import os
export_dir = "exported_model"
os.makedirs(export_dir, exist_ok=True)

# Save model in different formats
local_model.save_pretrained(export_dir)
local_tokenizer.save_pretrained(export_dir)
print(f"Model exported to: {export_dir}")

# === Model Loading from Export ===
print("\n📥 16. Model Loading from Export")
exported_model = AutoModelForMaskedLM.from_pretrained(export_dir)
exported_tokenizer = AutoTokenizer.from_pretrained(export_dir)
print("Model loaded from export directory")

# === Model Validation ===
print("\n✅ 17. Model Validation")
# Test that exported model works the same
test_text = "This is a <mask> test."
original_result = unmasker(test_text)
exported_unmasker = pipeline("fill-mask", model=exported_model, tokenizer=exported_tokenizer)
exported_result = exported_unmasker(test_text)

print("Original model result:", original_result[0]['token_str'])
print("Exported model result:", exported_result[0]['token_str'])
print("Results match:", original_result[0]['token_str'] == exported_result[0]['token_str'])

# === Hub Features (Skipped) ===
print("\n🌐 18. Hub Features")
print("⚠️ Hub features skipped - requires authentication")
print("To use Hub features, login with: huggingface_hub.login()")
print("Then uncomment the code below:")
print("""
# from huggingface_hub import notebook_login
# notebook_login()
# model.push_to_hub("my-awesome-camembert")
# model = AutoModelForMaskedLM.from_pretrained("username/my-awesome-camembert")
""")

# === Model Documentation ===
print("\n📚 19. Model Documentation")
print("Model documentation:")
print(f"  Model name: {model_name}")
print(f"  Model type: {type(model).__name__}")
print(f"  Tokenizer type: {type(tokenizer).__name__}")
print(f"  Vocabulary size: {tokenizer.vocab_size}")
print(f"  Max position embeddings: {model.config.max_position_embeddings}")

# === Performance Testing ===
print("\n⚡ 20. Performance Testing")
import time

# Test inference speed
test_text = "This is a <mask> performance test."
start_time = time.time()
result = unmasker(test_text)
end_time = time.time()
inference_time = end_time - start_time

print(f"Inference time: {inference_time:.4f} seconds")
print(f"Result: {result[0]['token_str']} (score: {result[0]['score']:.4f})")

print("\n" + "=" * 60)
print("✅ Chapter 8 completed successfully!")
print("=" * 60)
