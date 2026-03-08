"""Hugging Face NLP Course -- Chapter 9: Building and Sharing Demos

Standalone executable script demonstrating Gradio interfaces for ML model demos.
Source: https://huggingface.co/learn/nlp-course/chapter9
"""

# ============================================================
# 第9章: 使用Gradio构建演示 / Chapter 9: Building Demos with Gradio
# ============================================================
#
# 本章概要 / Chapter Summary:
#   本章介绍如何使用Gradio库为机器学习模型创建交互式Web界面。
#   你将学习两大核心API: gr.Interface用于快速构建单函数界面,
#   gr.Blocks用于构建复杂的多页签、多组件布局。通过设置
#   share=True, 你可以生成一个公共URL让任何人访问你的模型演示。
#   本章还展示了将Gradio应用部署到Hugging Face Spaces的方法。
#
#   This chapter shows how to build interactive web UIs for ML models
#   using the Gradio library. You will learn two core APIs: gr.Interface
#   for quickly wrapping a single function, and gr.Blocks for building
#   complex multi-tab, multi-component layouts. By setting share=True,
#   you can generate a public URL for anyone to access your model demo.
#   The chapter also demonstrates deploying Gradio apps to HF Spaces.
#
# 核心概念 / Key Concepts:
#   1. gr.Interface -- 快速包装: fn + inputs + outputs = Web UI
#      / Quick wrapper: fn + inputs + outputs = Web UI
#   2. gr.Blocks -- 灵活布局: Tabs, Row, Column, 自定义交互
#      / Flexible layout: Tabs, Row, Column, custom interactions
#   3. share=True -- 一键生成公共链接, 72小时有效
#      / One-click public URL, valid for 72 hours
#
# 模型架构 / Model Architecture:
#
#   Python Function (model inference)
#       |
#       v
#   +------------------------------------------+
#   | gr.Interface(                            |
#   |   fn = predict,                          |
#   |   inputs = gr.Textbox(),                 |
#   |   outputs = gr.Label()                   |
#   | )                                        |
#   +------------------------------------------+
#       |
#       v
#   +------------------------------------------+
#   | Auto-generated Web UI                    |
#   | (HTML/JS/CSS, no frontend code needed)   |
#   +------------------------------------------+
#       |
#       v
#   demo.launch(share=True)
#       |
#       v
#   Public URL: https://xxxxx.gradio.live
#
#   Blocks API (高级用法 / Advanced):
#   +------------------------------------------+
#   | gr.Blocks()                              |
#   |   +-- gr.Tabs()                          |
#   |   |     +-- TabItem("Sentiment")         |
#   |   |     +-- TabItem("Generation")        |
#   |   |     +-- TabItem("Summarization")     |
#   |   +-- gr.Row() / gr.Column()             |
#   |   +-- gr.Textbox / gr.Image / gr.Button  |
#   +------------------------------------------+
#
# 代码示例说明 / Code Examples in This File:
#   - Gradio安装检查 (Installation check)               ~line 64-74
#   - 基础Interface (Sentiment analysis interface)      ~line 77-106
#   - 高级Interface (Advanced DistilBERT interface)     ~line 109-163
#   - 多模型Interface (Multi-model: sentiment/gen/sum)  ~line 166-212
#   - 图像分类Interface (Image classification)          ~line 215-241
#   - 多页签Blocks (Tabbed Blocks with 3 tabs)         ~line 244-293
#   - 功能测试 (Function testing without launch)        ~line 296-306
#   - 启动说明 (Launch instructions, commented out)     ~line 309-328
#
# 下游应用 / Downstream Applications:
#   - 模型演示 / Model Demos (interactive prototypes for stakeholders)
#   - 快速原型开发 / Rapid Prototyping (test ideas without frontend code)
#   - Hugging Face Spaces部署 / Spaces Deployment (host on HF Spaces)
#
# 课程链接 / Course Link:
#   https://huggingface.co/learn/nlp-course/chapter9
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
print("🚀 Chapter 9: Creating a web application with Gradio")
print("=" * 60)

QUICK_RUN = os.getenv("HF_QUICK_RUN", "1") == "1"
print(f"⚙️ HF_QUICK_RUN={int(QUICK_RUN)} (set HF_QUICK_RUN=0 for full chapter run)")

# === Gradio Installation Check ===
print("\n📦 1. Checking Gradio Installation")
try:
    import gradio as gr
    print("✅ Gradio is installed and ready to use")
    print(f"Gradio version: {gr.__version__}")
except ImportError:
    print("❌ Gradio not found. Installing...")
    import subprocess
    subprocess.check_call(["pip", "install", "gradio"])
    import gradio as gr
    print("✅ Gradio installed successfully")

def run_quick_demo() -> None:
    """Run a minimal Gradio + local inference smoke test."""
    print("\n⚡ Quick Run: minimal Gradio + inference smoke test")
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
    quick_ids = torch.randint(0, config.vocab_size, (1, 10), device=device)
    quick_mask = torch.ones_like(quick_ids)
    with torch.no_grad():
        quick_logits = model(input_ids=quick_ids, attention_mask=quick_mask).logits
    quick_output = torch.argmax(quick_logits, dim=-1).cpu().tolist()
    print("Quick input_ids shape:", tuple(quick_ids.shape))
    print("Quick logits:", quick_logits.cpu().tolist())
    print("Quick predicted label ids:", quick_output)
    print("Gradio module loaded:", gr.__name__)
    print("\n✅ Chapter 9 quick run completed successfully!")

if QUICK_RUN:
    run_quick_demo()
    raise SystemExit(0)

# === Basic Gradio Interface ===
print("\n🔧 2. Creating Basic Gradio Interface")
from transformers import pipeline

# Load a simple model for demo
classifier = pipeline("sentiment-analysis")

def analyze_sentiment(text: str) -> tuple[str, float]:
    """Run sentiment inference for one input string."""
    result = classifier(text)
    return result[0]['label'], result[0]['score']

# Create Gradio interface
def create_sentiment_interface() -> Any:
    """Build a basic sentiment-analysis Gradio interface."""
    interface = gr.Interface(
        fn=analyze_sentiment,
        inputs=gr.Textbox(label="Enter text to analyze", placeholder="Type your text here..."),
        outputs=[
            gr.Textbox(label="Sentiment"),
            gr.Number(label="Confidence Score")
        ],
        title="Sentiment Analysis",
        description="Analyze the sentiment of your text using Hugging Face Transformers",
        examples=[
            "I love this product!",
            "This is terrible.",
            "It's okay, nothing special."
        ]
    )
    return interface

print("Basic sentiment analysis interface created")

# === Advanced Gradio Interface ===
print("\n🎨 3. Creating Advanced Gradio Interface")
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load model and tokenizer
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

def advanced_sentiment_analysis(text: str) -> tuple[str, float]:
    """Run manual tokenizer/model sentiment inference with confidence output."""
    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    
    # Get model prediction
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    # Get predicted class and confidence
    predicted_class = torch.argmax(predictions, dim=-1).item()
    confidence = torch.max(predictions).item()
    
    # Map to labels
    labels = ["NEGATIVE", "POSITIVE"]
    predicted_label = labels[predicted_class]
    
    return predicted_label, confidence

def create_advanced_interface() -> Any:
    """Build a richer Gradio interface with formatting options."""
    interface = gr.Interface(
        fn=advanced_sentiment_analysis,
        inputs=gr.Textbox(
            label="Enter text to analyze",
            placeholder="Type your text here...",
            lines=3
        ),
        outputs=[
            gr.Textbox(label="Predicted Sentiment"),
            gr.Number(label="Confidence Score", precision=4)
        ],
        title="Advanced Sentiment Analysis",
        description="Advanced sentiment analysis using DistilBERT",
        examples=[
            "I absolutely love this!",
            "This is the worst thing ever.",
            "It's okay, I guess.",
            "Amazing product, highly recommended!",
            "Terrible quality, waste of money."
        ],
        theme=gr.themes.Soft(),
        allow_flagging="never"
    )
    return interface

print("Advanced sentiment analysis interface created")

# === Multi-Model Interface ===
print("\n🤖 4. Creating Multi-Model Interface")
from transformers import pipeline

# Load multiple models
sentiment_classifier = pipeline("sentiment-analysis")
text_generator = pipeline("text-generation", model="gpt2", max_length=50)
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def multi_model_analysis(text: str, task: str) -> str:
    """Route text to a selected NLP task and return a formatted result."""
    if task == "Sentiment Analysis":
        result = sentiment_classifier(text)
        return f"Sentiment: {result[0]['label']} (Score: {result[0]['score']:.4f})"
    
    elif task == "Text Generation":
        result = text_generator(text, max_length=50, num_return_sequences=1)
        return result[0]['generated_text']
    
    elif task == "Summarization":
        result = summarizer(text, max_length=50, min_length=20, do_sample=False)
        return result[0]['summary_text']
    
    else:
        return "Please select a task"

def create_multi_model_interface() -> Any:
    """Build a task-switching interface for multiple NLP pipelines."""
    interface = gr.Interface(
        fn=multi_model_analysis,
        inputs=[
            gr.Textbox(label="Enter text", placeholder="Type your text here...", lines=3),
            gr.Dropdown(
                choices=["Sentiment Analysis", "Text Generation", "Summarization"],
                label="Select Task",
                value="Sentiment Analysis"
            )
        ],
        outputs=gr.Textbox(label="Result", lines=5),
        title="Multi-Model NLP Interface",
        description="Choose from different NLP tasks using Hugging Face models",
        examples=[
            ["I love this product!", "Sentiment Analysis"],
            ["Once upon a time", "Text Generation"],
            ["This is a long text that needs to be summarized for demonstration purposes.", "Summarization"]
        ]
    )
    return interface

print("Multi-model interface created")

# === Image Classification Interface ===
print("\n🖼️ 5. Creating Image Classification Interface")
from transformers import pipeline

# Load image classification model
image_classifier = pipeline("image-classification", model="google/vit-base-patch16-224")

def classify_image(image: Any) -> Any:
    """Classify one image and return model predictions."""
    if image is None:
        return "Please upload an image"
    
    result = image_classifier(image)
    return result

def create_image_interface() -> Any:
    """Build an image-classification demo interface."""
    interface = gr.Interface(
        fn=classify_image,
        inputs=gr.Image(type="pil", label="Upload an image"),
        outputs=gr.Textbox(label="Classification Results", lines=5),
        title="Image Classification",
        description="Classify images using Vision Transformer (ViT)",
        examples=[
            "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
        ]
    )
    return interface

print("Image classification interface created")

# === Tabbed Interface ===
print("\n📑 6. Creating Tabbed Interface")
def create_tabbed_interface() -> Any:
    """Build a multi-tab Gradio app that combines several NLP demos."""
    with gr.Blocks(title="Hugging Face NLP Demo") as demo:
        gr.Markdown("# Hugging Face NLP Demo")
        gr.Markdown("Explore different NLP tasks using Hugging Face Transformers")
        
        with gr.Tabs():
            with gr.TabItem("Sentiment Analysis"):
                gr.Interface(
                    fn=analyze_sentiment,
                    inputs=gr.Textbox(label="Enter text", placeholder="Type your text here..."),
                    outputs=[
                        gr.Textbox(label="Sentiment"),
                        gr.Number(label="Confidence Score")
                    ],
                    title="Sentiment Analysis",
                    examples=[
                        "I love this!",
                        "This is terrible.",
                        "It's okay."
                    ]
                )
            
            with gr.TabItem("Text Generation"):
                gr.Interface(
                    fn=lambda x: text_generator(x, max_length=50, num_return_sequences=1)[0]['generated_text'],
                    inputs=gr.Textbox(label="Enter prompt", placeholder="Type your prompt here..."),
                    outputs=gr.Textbox(label="Generated Text", lines=5),
                    title="Text Generation",
                    examples=[
                        "Once upon a time",
                        "The future of AI is",
                        "In a world where"
                    ]
                )
            
            with gr.TabItem("Text Summarization"):
                gr.Interface(
                    fn=lambda x: summarizer(x, max_length=50, min_length=20, do_sample=False)[0]['summary_text'],
                    inputs=gr.Textbox(label="Enter text to summarize", placeholder="Type your text here...", lines=5),
                    outputs=gr.Textbox(label="Summary", lines=3),
                    title="Text Summarization",
                    examples=[
                        "This is a long text that needs to be summarized for demonstration purposes. It contains multiple sentences and should be reduced to a shorter version while maintaining the key information."
                    ]
                )
    
    return demo

print("Tabbed interface created")

# === Interface Testing ===
print("\n🧪 7. Testing Interfaces")
print("Testing sentiment analysis function:")
test_text = "I love this product!"
result = analyze_sentiment(test_text)
print(f"Input: '{test_text}'")
print(f"Output: {result}")

print("\nTesting advanced sentiment analysis:")
result = advanced_sentiment_analysis(test_text)
print(f"Input: '{test_text}'")
print(f"Output: {result}")

# === Interface Launch (Optional) ===
print("\n🚀 8. Interface Launch")
print("⚠️ Interface launching skipped for demo purposes")
print("To launch interfaces, uncomment the code below:")
print("""
# Launch basic interface
# basic_interface = create_sentiment_interface()
# basic_interface.launch()

# Launch advanced interface
# advanced_interface = create_advanced_interface()
# advanced_interface.launch()

# Launch multi-model interface
# multi_interface = create_multi_model_interface()
# multi_interface.launch()

# Launch tabbed interface
# tabbed_interface = create_tabbed_interface()
# tabbed_interface.launch()
""")

# === Gradio Features Demo ===
print("\n✨ 9. Gradio Features Demo")
print("Available Gradio features:")
print("  - Text input/output")
print("  - Image input/output")
print("  - Number input/output")
print("  - Dropdown selection")
print("  - File upload")
print("  - Examples")
print("  - Themes")
print("  - Tabbed interfaces")
print("  - Real-time updates")

# === Model Information ===
print("\n📋 10. Model Information")
print("Models used in this chapter:")
print(f"  Sentiment Analysis: {classifier.model.config.name_or_path}")
print(f"  Text Generation: gpt2")
print(f"  Summarization: facebook/bart-large-cnn")
print(f"  Image Classification: google/vit-base-patch16-224")

print("\n" + "=" * 60)
print("✅ Chapter 9 completed successfully!")
print("=" * 60)
