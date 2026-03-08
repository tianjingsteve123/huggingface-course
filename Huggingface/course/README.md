# Hugging Face NLP Course - Executable Chapter Scripts

This folder contains runnable scripts `chapter1.py` to `chapter9.py` adapted for WSL Ubuntu development.

## Quality Goals

- clear bilingual comments (EN/ZH) and section headers,
- explicit intermediate prints for easier debugging/learning,
- deterministic quick-run mode for IDE run-button validation,
- clean folder structure suitable for GitHub commits.

## Environment (WSL Ubuntu)

```bash
conda create -n hf_course python=3.11 -y
conda activate hf_course
pip install --index-url https://download.pytorch.org/whl/cu126 torch==2.9.1+cu126 torchvision==0.24.1+cu126 torchaudio==2.9.1+cu126
pip install -r requirements.txt
```

## Quick Start

```bash
cd /mnt/c/coding/Huggingface/course
conda activate hf_course
python chapter1.py
```

By default, scripts run in fast verification mode:

```bash
HF_QUICK_RUN=1 python chapter1.py
```

To run the full chapter flow (slower, more downloads):

```bash
HF_QUICK_RUN=0 python chapter1.py
```

## Run All Chapters

```bash
cd /mnt/c/coding/Huggingface/course
conda activate hf_course
for ch in chapter{1..9}.py; do
  python -u "$ch" && echo "PASS: $ch" || echo "FAIL: $ch"
done
```

## Validate Script Structure

```bash
cd /mnt/c/coding/Huggingface/course
python test_chapters.py
```
