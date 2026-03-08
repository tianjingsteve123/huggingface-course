# huggingface-course

Run-ready and commit-ready Hugging Face course scripts, with repeatable GitHub publishing workflow for open source contributors.

Re-implementations of huggingface NLP course content https://huggingface.co/learn/llm-course

IDE platform - cursor, WSL environment

## What this repository contains

- `Huggingface/course/chapter1.py` to `chapter9.py`: executable learning scripts.
- `Huggingface/course/test_chapters.py`: structural validation for chapter scripts.
- `Huggingface/course/requirements.txt`: pinned Python dependencies.
- `.cursor/rules/` and `.cursor/skills/`: reusable publishing automation guidance.

## Project structure

```text
.
├── Huggingface/
│   ├── README.md
│   └── course/
│       ├── README.md
│       ├── requirements.txt
│       ├── test_chapters.py
│       ├── chapter1.py
│       ├── ...
│       └── chapter9.py
├── .cursor/
│   ├── rules/
│   │   └── folder-publish-guardrails.mdc
│   └── skills/
│       └── folder-publisher/
│           ├── SKILL.md
│           ├── checklist.md
│           └── examples.md
└── LICENSE
```

## Environment setup (Linux/macOS/WSL)

Run commands from repository root, then enter `Huggingface/course`.
Use relative paths in docs so any contributor can run them in their own local clone.

```bash
cd Huggingface/course
conda create -n hf_course python=3.11 -y
conda activate hf_course
pip install --index-url https://download.pytorch.org/whl/cu126 \
  torch==2.9.1+cu126 torchvision==0.24.1+cu126 torchaudio==2.9.1+cu126
pip install -r requirements.txt
```

## Run scripts

Quick run (default mode in scripts, minimal download):

```bash
cd Huggingface/course
conda activate hf_course
HF_QUICK_RUN=1 python chapter1.py
```

Full run (slower, may download larger assets):

```bash
HF_QUICK_RUN=0 python chapter1.py
```

Run all chapters with intermediate output:

```bash
cd Huggingface/course
conda activate hf_course
for ch in chapter{1..9}.py; do
  echo "=== Running ${ch} ==="
  python -u "$ch" && echo "PASS: ${ch}" || echo "FAIL: ${ch}"
done
```

## Validation commands

```bash
cd Huggingface/course
conda activate hf_course
python test_chapters.py
```

## Publishing standard used in this repository

This repository is intentionally published as a **folder-scoped repo** (only target folder plus minimal metadata), so unrelated workspace directories are never pushed by accident.

### Safe folder-only publish workflow

1. Define publishing scope (example: only `Huggingface/`).
2. Build a temporary clean repo directory.
3. Copy only target folder and required root files (for example `LICENSE`, `README.md`).
4. Commit from the clean repo, not from a dirty monorepo root.
5. Push to target GitHub repo.
6. Verify remote root contents after push.

### Why this is important

- Prevents accidental upload of unrelated folders such as `digital_bio/` or `algorithm/`.
- Keeps repository history focused and easier to maintain.
- Makes repeated publishing of other folders predictable and scriptable.

## Reusable automation for next uploads

- Rule: `.cursor/rules/folder-publish-guardrails.mdc`
- Skill: `.cursor/skills/folder-publisher/SKILL.md`
- Checklist: `.cursor/skills/folder-publisher/checklist.md`
- Examples: `.cursor/skills/folder-publisher/examples.md`

Use this same rule + skill set when publishing any other folder in your workspace.
