# Multimodal Conversational Root Cause Analysis System

A Streamlit application that combines ML, LLM, and OCR to perform end-to-end root cause analysis on tabular data.

---

## Features

| # | Feature | Description |
|---|---------|-------------|
| 1 | **Data Input** | Upload CSV or use the built-in demo dataset |
| 2 | **ML Analysis** | Random Forest (feature importance) + KMeans (segmentation) + time trend |
| 3 | **LLM Root Cause Engine** | Structured prompt → Claude/OpenAI explanation with key drivers & actions |
| 4 | **Conversational Interface** | Chat with the system about your data |
| 5 | **OCR + Vision** | Upload an image → extract text → LLM explanation |

---

## Quick Start

### 1. Install system dependency (OCR)

**macOS:**
```bash
brew install tesseract
```

**Ubuntu/Debian:**
```bash
sudo apt-get install tesseract-ocr
```

**Windows:**
Download the installer from: https://github.com/tesseract-ocr/tesseract/releases

---

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

---

### 3. Run the app

```bash
streamlit run app.py
```

---

## API Key Setup

In the **sidebar**, choose your LLM provider and paste your key:

| Provider | Key format | Where to get it |
|----------|-----------|-----------------|
| Claude (Anthropic) | `sk-ant-...` | https://console.anthropic.com |
| OpenAI | `sk-...` | https://platform.openai.com |

The key is never stored — it lives only in the current session.

---

## Project Structure

```
rca_app/
├── app.py            ← Single-file Streamlit application
├── requirements.txt  ← Python dependencies
└── README.md         ← This file
```

---

## Code Architecture

```
app.py
 ├── load_data()          → CSV ingestion + column type detection
 ├── run_ml_models()      → Random Forest + KMeans + time trend
 ├── generate_prompt()    → Structured LLM prompt template
 ├── call_llm()           → Claude or OpenAI API call (modular)
 ├── run_ocr()            → pytesseract image → text
 ├── chart_*()            → Plotly chart helpers
 └── main()               → Streamlit UI layout
```

---

## Notes

- The app runs **entirely locally** — your data never leaves your machine (except for the LLM API call).
- OCR feature requires Tesseract installed at the OS level.
- If you don't have an API key, the ML charts still work — only the LLM sections will show a warning.
