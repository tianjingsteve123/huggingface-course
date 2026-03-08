"""Hugging Face NLP Course -- Chapter 7: Main NLP Tasks

Standalone executable script covering token classification, QA, summarization, translation, and more.
Source: https://huggingface.co/learn/nlp-course/chapter7
"""

# ============================================================
# 第7章: 主要NLP任务 / Chapter 7: Main NLP Tasks
# ============================================================
#
# 本章概要 / Chapter Summary:
#   本章全面展示主要NLP任务的实现方式, 包括Token分类(NER)、
#   抽取式问答(Extractive QA)、文本摘要(Summarization)、
#   机器翻译(Translation)和文本生成(Text Generation)。每个任务
#   都使用不同的预训练模型和任务头: BERT用于NER和QA, BART用于
#   摘要和翻译, GPT-2用于文本生成。你将看到同一Transformer框架
#   如何通过更换任务头适配多种NLP任务。
#
#   This chapter comprehensively demonstrates major NLP tasks: token
#   classification (NER), extractive QA, summarization, translation,
#   and text generation. Each task uses a different pretrained model
#   and task head: BERT for NER/QA, BART for summarization/translation,
#   GPT-2 for generation. You will see how the same Transformer
#   framework adapts to diverse NLP tasks by swapping task heads.
#
# 核心概念 / Key Concepts:
#   1. Token分类 (NER) -- 对每个token预测实体标签 (B-PER, I-LOC...)
#      / Predict entity labels for each token
#   2. 抽取式问答 -- 在上下文中定位答案的起止位置
#      / Locate answer start/end positions in context
#   3. Seq2Seq任务 -- 摘要和翻译使用编码器-解码器架构
#      / Summarization & translation use encoder-decoder architecture
#
# 模型架构 / Model Architecture:
#
#   Input Text
#       |
#       v
#   +----------------------------------+
#   | Pretrained Transformer           |
#   | (BERT / BART / GPT-2)           |
#   +----------------------------------+
#       |
#       v
#   +-- [NER Head] ---------> B-PER, I-PER, B-ORG, O ...
#   |   (TokenClassification)
#   |
#   +-- [QA Head] ----------> start_pos=5, end_pos=8
#   |   (QuestionAnswering)
#   |
#   +-- [Seq2Seq Head] -----> "Summary of the article..."
#   |   (Summarization)
#   |
#   +-- [Seq2Seq Head] -----> "This course is produced by..."
#   |   (Translation)
#   |
#   +-- [CausalLM Head] ----> "In this course, we will..."
#       (TextGeneration)
#
# 代码示例说明 / Code Examples in This File:
#   - 文本分类 (Text classification with DistilBERT)     ~line 58-74
#   - 命名实体识别 (NER with BERT-large)                 ~line 77-88
#   - 问答系统 (QA with DistilBERT-SQuAD)               ~line 91-105
#   - 文本摘要 (Summarization with BART-CNN)             ~line 108-128
#   - 文本生成 (Text generation with GPT-2)              ~line 131-142
#   - 机器翻译 (Translation with OPUS-MT)                ~line 145-156
#   - 掩码填充 (Fill-mask with BERT)                     ~line 159-170
#   - 零样本分类 (Zero-shot with BART-MNLI)              ~line 173-187
#   - 图像分类 (Image classification with ViT)           ~line 204-215
#   - 多任务综合演示 (Multi-task demo)                    ~line 249-264
#
# 下游应用 / Downstream Applications:
#   - 命名实体识别 / NER (information extraction, knowledge graphs)
#   - 问答与搜索 / QA (document search, FAQ bots, customer support)
#   - 摘要与翻译 / Summarization & Translation (news, multilingual)
#
# 课程链接 / Course Link:
#   https://huggingface.co/learn/nlp-course/chapter7
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
print("🚀 Chapter 7: Main NLP tasks")
print("=" * 60)

QUICK_RUN = os.getenv("HF_QUICK_RUN", "1") == "1"
print(f"⚙️ HF_QUICK_RUN={int(QUICK_RUN)} (set HF_QUICK_RUN=0 for full chapter run)")

def run_quick_demo() -> None:
    """Run one representative classification task with a tiny local model."""
    print("\n⚡ Quick Run: one representative NLP task with explicit outputs")
    from transformers import DistilBertConfig, DistilBertForSequenceClassification

    config = DistilBertConfig(
        vocab_size=30522,
        n_layers=2,
        dim=128,
        hidden_dim=256,
        n_heads=4,
        num_labels=3
    )
    model = DistilBertForSequenceClassification(config).to(device)
    sample_ids = torch.randint(0, config.vocab_size, (2, 12), device=device)
    sample_mask = torch.ones_like(sample_ids)
    print("Sample input_ids shape:", tuple(sample_ids.shape))

    with torch.no_grad():
        logits = model(input_ids=sample_ids, attention_mask=sample_mask).logits
    result = torch.argmax(logits, dim=-1).cpu().tolist()
    print("Logits:", logits.cpu().tolist())
    print("Predicted class ids:", result)

    print("\n✅ Chapter 7 quick run completed successfully!")


if QUICK_RUN:
    run_quick_demo()
    raise SystemExit(0)

# === Text Classification ===
print("\n📊 1. Text Classification")
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Load IMDB dataset for sentiment analysis
dataset = load_dataset("imdb")
print("IMDB dataset loaded:", dataset)

# Load tokenizer and model
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Create pipeline
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)
result = classifier("This is a great movie!")
print("Text classification result:", result)

# === Named Entity Recognition ===
print("\n🏷️ 2. Named Entity Recognition")
from transformers import AutoModelForTokenClassification

# Load NER model
ner_model_name = "dbmdz/bert-large-cased-finetuned-conll03-english"
ner_tokenizer = AutoTokenizer.from_pretrained(ner_model_name)
ner_model = AutoModelForTokenClassification.from_pretrained(ner_model_name)

# Create NER pipeline
ner_pipeline = pipeline("ner", model=ner_model, tokenizer=ner_tokenizer, aggregation_strategy="simple")
result = ner_pipeline("My name is Sylvain and I work at Hugging Face in Brooklyn.")
print("NER result:", result)

# === Question Answering ===
print("\n❓ 3. Question Answering")
from transformers import AutoModelForQuestionAnswering

# Load QA model
qa_model_name = "distilbert-base-cased-distilled-squad"
qa_tokenizer = AutoTokenizer.from_pretrained(qa_model_name)
qa_model = AutoModelForQuestionAnswering.from_pretrained(qa_model_name)

# Create QA pipeline
qa_pipeline = pipeline("question-answering", model=qa_model, tokenizer=qa_tokenizer)
result = qa_pipeline(
    question="Where do I work?",
    context="My name is Sylvain and I work at Hugging Face in Brooklyn."
)
print("QA result:", result)

# === Text Summarization ===
print("\n📝 4. Text Summarization")
from transformers import AutoModelForSeq2SeqLM

# Load summarization model
sum_model_name = "facebook/bart-large-cnn"
sum_tokenizer = AutoTokenizer.from_pretrained(sum_model_name)
sum_model = AutoModelForSeq2SeqLM.from_pretrained(sum_model_name)

# Create summarization pipeline
summarizer = pipeline("summarization", model=sum_model, tokenizer=sum_tokenizer)
text = """
America has changed dramatically during recent years. Not only has the number of 
graduates in traditional engineering disciplines such as mechanical, civil, 
electrical, chemical, and aeronautical engineering declined, but in most of 
the premier American universities engineering curricula now concentrate on and 
are largely dominated by computer science. However, the fact that there is an 
insufficient number of graduates in computer science is not the only reason 
why the high-tech industry is taking a pass on American workers.
"""
result = summarizer(text, max_length=50, min_length=20, do_sample=False)
print("Summarization result:", result)

# === Text Generation ===
print("\n✍️ 5. Text Generation")
from transformers import AutoModelForCausalLM

# Load text generation model
gen_model_name = "gpt2"
gen_tokenizer = AutoTokenizer.from_pretrained(gen_model_name)
gen_model = AutoModelForCausalLM.from_pretrained(gen_model_name)

# Create text generation pipeline
generator = pipeline("text-generation", model=gen_model, tokenizer=gen_tokenizer)
result = generator("In this course, we will teach you how to", max_length=50, num_return_sequences=2)
print("Text generation result:", result)

# === Translation ===
print("\n🌍 6. Translation")
from transformers import AutoModelForSeq2SeqLM

# Load translation model
trans_model_name = "Helsinki-NLP/opus-mt-fr-en"
trans_tokenizer = AutoTokenizer.from_pretrained(trans_model_name)
trans_model = AutoModelForSeq2SeqLM.from_pretrained(trans_model_name)

# Create translation pipeline
translator = pipeline("translation", model=trans_model, tokenizer=trans_tokenizer)
result = translator("Ce cours est produit par Hugging Face.")
print("Translation result:", result)

# === Fill Mask ===
print("\n🔤 7. Fill Mask")
from transformers import AutoModelForMaskedLM

# Load fill mask model
mask_model_name = "bert-base-uncased"
mask_tokenizer = AutoTokenizer.from_pretrained(mask_model_name)
mask_model = AutoModelForMaskedLM.from_pretrained(mask_model_name)

# Create fill mask pipeline
unmasker = pipeline("fill-mask", model=mask_model, tokenizer=mask_tokenizer)
result = unmasker("This course will teach you all about [MASK] models.", top_k=2)
print("Fill mask result:", result)

# === Zero-shot Classification ===
print("\n🎯 8. Zero-shot Classification")
from transformers import AutoModelForSequenceClassification

# Load zero-shot model
zs_model_name = "facebook/bart-large-mnli"
zs_tokenizer = AutoTokenizer.from_pretrained(zs_model_name)
zs_model = AutoModelForSequenceClassification.from_pretrained(zs_model_name)

# Create zero-shot pipeline
zero_shot_classifier = pipeline("zero-shot-classification", model=zs_model, tokenizer=zs_tokenizer)
result = zero_shot_classifier(
    "This is a course about the Transformers library",
    candidate_labels=["education", "politics", "business"],
)
print("Zero-shot classification result:", result)

# === Text-to-Speech (Skipped) ===
print("\n🎵 9. Text-to-Speech")
print("⚠️ Text-to-Speech skipped - requires additional dependencies")
print("To enable TTS, install: pip install TTS")

# === Automatic Speech Recognition (Skipped) ===
print("\n🎤 10. Automatic Speech Recognition")
print("⚠️ ASR skipped - requires ffmpeg")
print("To enable ASR, install ffmpeg and uncomment the code below:")
print("""
# asr_pipeline = pipeline("automatic-speech-recognition")
# result = asr_pipeline("path/to/audio/file")
""")

# === Image Classification ===
print("\n🖼️ 11. Image Classification")
from transformers import AutoModelForImageClassification, AutoImageProcessor

# Load image classification model
img_model_name = "google/vit-base-patch16-224"
img_processor = AutoImageProcessor.from_pretrained(img_model_name)
img_model = AutoModelForImageClassification.from_pretrained(img_model_name)

# Create image classification pipeline
image_classifier = pipeline("image-classification", model=img_model, feature_extractor=img_processor)
result = image_classifier("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg")
print("Image classification result:", result)

# === Object Detection ===
print("\n🔍 12. Object Detection")
print("⚠️ Object detection skipped - requires timm library")
print("To enable object detection, install: pip install timm")
print("Then uncomment the code below:")
print("""
# from transformers import AutoModelForObjectDetection, AutoImageProcessor
# obj_model_name = "facebook/detr-resnet-50"
# obj_processor = AutoImageProcessor.from_pretrained(obj_model_name)
# obj_model = AutoModelForObjectDetection.from_pretrained(obj_model_name)
# object_detector = pipeline("object-detection", model=obj_model, feature_extractor=obj_processor)
""")

# === Model Comparison ===
print("\n📊 13. Model Comparison")
models = {
    "Text Classification": "distilbert-base-uncased",
    "NER": "dbmdz/bert-large-cased-finetuned-conll03-english",
    "Question Answering": "distilbert-base-cased-distilled-squad",
    "Summarization": "facebook/bart-large-cnn",
    "Text Generation": "gpt2",
    "Translation": "Helsinki-NLP/opus-mt-fr-en",
    "Fill Mask": "bert-base-uncased",
    "Zero-shot": "facebook/bart-large-mnli"
}

print("Available models for different tasks:")
for task, model_name in models.items():
    print(f"  {task}: {model_name}")

# === Task Performance Demo ===
print("\n🎯 14. Task Performance Demo")
test_text = "Hugging Face is a company that provides open-source machine learning tools."

print(f"Testing on: '{test_text}'")
print("\nResults:")

# Text classification
result = classifier(test_text)
print(f"  Sentiment: {result[0]['label']} (score: {result[0]['score']:.4f})")

# NER
result = ner_pipeline(test_text)
print(f"  NER: {result}")

# Zero-shot classification
result = zero_shot_classifier(test_text, candidate_labels=["technology", "business", "education"])
print(f"  Zero-shot: {result['labels'][0]} (score: {result['scores'][0]:.4f})")

print("\n" + "=" * 60)
print("✅ Chapter 7 completed successfully!")
print("=" * 60)
