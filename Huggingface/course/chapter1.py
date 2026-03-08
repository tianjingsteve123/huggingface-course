"""Hugging Face NLP Course -- Chapter 1: Transformers, What Can They Do?

Standalone executable script covering the pipeline API for core NLP tasks.
Source: https://huggingface.co/learn/nlp-course/chapter1
"""

# ============================================================
# 第1章: Transformers简介 / Chapter 1: Introduction to Transformers
# ============================================================
#
# 本章概要 / Chapter Summary:
#   本章是Hugging Face NLP课程的起点, 介绍了Transformer架构及其
#   在自然语言处理中的革命性作用。通过pipeline API, 你可以零代码
#   快速体验情感分析、命名实体识别、问答、摘要、翻译和文本生成等
#   核心NLP任务。本章还展示了BERT、GPT-2、T5、DistilBERT等主流
#   预训练模型, 帮助你理解不同模型适合的任务类型。
#
#   This chapter is the starting point of the Hugging Face NLP course,
#   introducing the Transformer architecture and its revolutionary role
#   in NLP. Through the pipeline API, you can quickly experience core
#   NLP tasks -- sentiment analysis, NER, QA, summarization, translation,
#   and text generation -- with minimal code. It also showcases popular
#   pretrained models like BERT, GPT-2, T5, and DistilBERT, helping you
#   understand which model fits which task.
#
# 核心概念 / Key Concepts:
#   1. Pipeline API -- 一行代码完成NLP任务 / One-liner NLP inference
#   2. Transformer架构 -- 自注意力机制驱动的编码器/解码器
#      / Self-attention-driven encoder/decoder architecture
#   3. 预训练模型 -- BERT(理解), GPT-2(生成), T5(seq2seq),
#      DistilBERT(轻量) / Pretrained: BERT(understanding),
#      GPT-2(generation), T5(seq2seq), DistilBERT(lightweight)
#
# 模型架构 / Model Architecture:
#
#   Text --> [Tokenizer] --> [Transformer Model] --> [Task Head] --> Output
#
#   详细流程 / Detailed Flow:
#
#   "I love NLP"
#       |
#       v
#   +------------------+
#   | Tokenizer        |  分词器: 将文本转换为Token IDs
#   | (Text -> IDs)    |  Converts raw text into numerical IDs
#   +------------------+
#       |  [101, 1045, 2293, 17953, 2361, 102]
#       v
#   +------------------+
#   | Transformer      |  Transformer模型: 编码语义信息
#   | Encoder/Decoder  |  Encodes contextual meaning
#   |   Embeddings     |
#   |   Self-Attention |  自注意力机制: 捕获词间关系
#   |   Feed-Forward   |  前馈网络: 非线性变换
#   |   Layer Norm     |
#   +------------------+
#       |  Hidden States (768-dim vectors)
#       v
#   +------------------+
#   | Task Head        |  任务头: 根据不同任务输出结果
#   | (Classification, |  Classification head, LM head, etc.
#   |  Generation, QA) |
#   +------------------+
#       |
#       v
#   Output: {"label": "POSITIVE", "score": 0.9998}
#
#   支持的任务 / Supported Tasks:
#   +-------------------+-------------------+---------------------+
#   | sentiment-analysis | zero-shot-cls     | text-generation     |
#   | fill-mask          | ner               | question-answering  |
#   | summarization      | translation       | image-classification|
#   +-------------------+-------------------+---------------------+
#
# 代码示例说明 / Code Examples in This File:
#   - 情感分析 (Sentiment Analysis)           ~line 114-124
#   - 零样本分类 (Zero-shot Classification)   ~line 127-135
#   - 文本生成 (Text Generation)              ~line 138-153
#   - 掩码填充 (Fill Mask)                    ~line 156-161
#   - 命名实体识别 (NER)                      ~line 164-169
#   - 问答系统 (Question Answering)           ~line 172-180
#   - 文本摘要 (Summarization)                ~line 183-201
#   - 机器翻译 (Translation)                  ~line 204-209
#   - 图像分类 (Image Classification)         ~line 212-222
#
# 下游应用 / Downstream Applications:
#   - 情感分析 / Sentiment Analysis (product reviews, social media)
#   - 命名实体识别 / NER (information extraction, knowledge graphs)
#   - 问答与搜索 / QA & Search (customer support, document retrieval)
#
# 课程链接 / Course Link:
#   https://huggingface.co/learn/nlp-course/chapter1
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
print("🚀 Chapter 1: Transformers, what can they do?")
print("=" * 60)

QUICK_RUN = os.getenv("HF_QUICK_RUN", "1") == "1"
print(f"⚙️ HF_QUICK_RUN={int(QUICK_RUN)} (set HF_QUICK_RUN=0 for full chapter run)")

def run_quick_demo() -> None:
    """Run a deterministic local smoke test without remote model downloads."""
    print("\n⚡ Quick Run: lightweight smoke test with explicit intermediate outputs")
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

    quick_inputs = torch.randint(0, config.vocab_size, (2, 12), device=device)
    quick_mask = torch.ones_like(quick_inputs)
    print("Quick input_ids shape:", tuple(quick_inputs.shape))

    with torch.no_grad():
        quick_logits = model(input_ids=quick_inputs, attention_mask=quick_mask).logits
    quick_preds = torch.argmax(quick_logits, dim=-1).cpu().tolist()
    print("Quick logits:", quick_logits.cpu().tolist())
    print("Quick predicted labels:", quick_preds)

    print("\n✅ Chapter 1 quick run completed successfully!")


if QUICK_RUN:
    run_quick_demo()
    raise SystemExit(0)

# === 情感分析 (Sentiment Analysis) ===
# 情感分析用于判断文本的正面/负面情绪 (Determines positive/negative sentiment of text)
# 应用场景: 产品评论分析, 社交媒体监控 (Use cases: product reviews, social media monitoring)
print("\n📊 1. Sentiment Analysis")
from transformers import pipeline
classifier = pipeline("sentiment-analysis")
result = classifier("I've been waiting for a HuggingFace course my whole life.")
print("Single text:", result)

result = classifier([
    "I've been waiting for a HuggingFace course my whole life.", 
    "I hate this so much!"
])
print("Multiple texts:", result)

# === 零样本分类 (Zero-shot Classification) ===
# 无需训练数据即可分类文本 (Classify text without task-specific training data)
# 应用场景: 动态内容标签, 新闻分类 (Use cases: dynamic content tagging, news categorization)
print("\n🎯 2. Zero-shot Classification")
classifier = pipeline("zero-shot-classification")
result = classifier(
    "This is a course about the Transformers library",
    candidate_labels=["education", "politics", "business"],
)
print("Zero-shot result:", result)

# === 文本生成 (Text Generation) ===
# 基于提示生成连贯文本 (Generate coherent text from a prompt)
# 应用场景: 聊天机器人, 代码补全, 创意写作 (Use cases: chatbots, code completion, creative writing)
print("\n✍️ 3. Text Generation")
generator = pipeline("text-generation")
result = generator("In this course, we will teach you how to")
print("Generated text:", result)

# 使用不同的模型 (Using a different model)
# SmolLM2是一个轻量级语言模型 (SmolLM2 is a lightweight language model)
generator = pipeline("text-generation", model="HuggingFaceTB/SmolLM2-360M")
result = generator(
    "In this course, we will teach you how to",
    max_length=30,
    num_return_sequences=2,
)
print("Generated with SmolLM2:", result)

# === 掩码填充 (Fill Mask) ===
# 预测被遮盖的词 (Predict masked/hidden words in text)
# 应用场景: 自动补全, 数据增强 (Use cases: autocomplete, data augmentation)
print("\n🔤 4. Fill Mask")
unmasker = pipeline("fill-mask")
result = unmasker("This course will teach you all about <mask> models.", top_k=2)
print("Fill mask result:", result)

# === 命名实体识别 (Named Entity Recognition / NER) ===
# 从文本中提取实体 (Extract entities like names, locations, organizations)
# 应用场景: 信息抽取, 知识图谱构建 (Use cases: information extraction, knowledge graph building)
print("\n🏷️ 5. Named Entity Recognition")
ner = pipeline("ner", aggregation_strategy="simple")
result = ner("My name is Sylvain and I work at Hugging Face in Brooklyn.")
print("NER result:", result)

# === 问答系统 (Question Answering) ===
# 根据上下文回答问题 (Answer questions based on provided context)
# 应用场景: 客服系统, 文档搜索 (Use cases: customer support, document search)
print("\n❓ 6. Question Answering")
question_answerer = pipeline("question-answering")
result = question_answerer(
    question="Where do I work?",
    context="My name is Sylvain and I work at Hugging Face in Brooklyn."
)
print("QA result:", result)

# === 文本摘要 (Summarization) ===
# 将长文本压缩为简短摘要 (Compress long text into a concise summary)
# 应用场景: 新闻摘要, 会议记录, 报告生成 (Use cases: news digests, meeting notes, report generation)
print("\n📝 7. Summarization")
summarizer = pipeline("summarization")
result = summarizer("""
America has changed dramatically during recent years. Not only has the number of 
graduates in traditional engineering disciplines such as mechanical, civil, 
electrical, chemical, and aeronautical engineering declined, but in most of 
the premier American universities engineering curricula now concentrate on and 
are largely dominated by computer science. However, the fact that there is an 
insufficient number of graduates in computer science is not the only reason 
why the high-tech industry is taking a pass on American workers. The fact is 
that the high-tech industry is taking a pass on American workers for a variety 
of reasons, including the fact that there is an insufficient number of graduates 
in computer science. The fact is that the high-tech industry is taking a pass on 
American workers for a variety of reasons, including the fact that there is an 
insufficient number of graduates in computer science.
""")
print("Summarization result:", result)

# === 机器翻译 (Translation) ===
# 将文本从一种语言翻译为另一种 (Translate text between languages)
# 应用场景: 跨语言沟通, 文档本地化 (Use cases: cross-language communication, localization)
print("\n🌍 8. Translation")
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-fr-en")
result = translator("Ce cours est produit par Hugging Face.")
print("Translation result:", result)

# === 图像分类 (Image Classification) ===
# 对图像进行分类识别 (Classify images into categories)
# 应用场景: 医学影像, 质量检测, 自动驾驶 (Use cases: medical imaging, quality control, autonomous driving)
print("\n🖼️ 9. Image Classification")
image_classifier = pipeline(
    task="image-classification", 
    model="google/vit-base-patch16-224"
)
result = image_classifier(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
)
print("Image classification result:", result)

# === 音频处理 (Audio Processing - Skipped due to ffmpeg requirement) ===
# 语音转文本 (Speech-to-text transcription)
# 应用场景: 会议转录, 语音助手, 字幕生成 (Use cases: meeting transcription, voice assistants, subtitles)
print("\n🎵 10. Audio Processing")
print("⚠️ Audio processing skipped - requires ffmpeg installation")
print("To enable audio processing, install ffmpeg and uncomment the code below:")
print("""
# transcriber = pipeline(
#     task="automatic-speech-recognition", 
#     model="openai/whisper-large-v3"
# )
# result = transcriber("path/to/audio/file")
# print("Audio transcription result:", result)
""")

print("\n" + "=" * 60)
print("✅ Chapter 1 completed successfully!")
print("=" * 60)
