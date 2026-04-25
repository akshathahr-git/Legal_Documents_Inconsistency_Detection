# Legal Document Inconsistency Detection System

## Overview
AI-powered system to detect contradictions and inconsistencies in legal documents, specifically designed for employment contracts.

## Features
- 🔍 Extract clauses from PDF/TXT documents
- 🧠 Semantic similarity comparison using Sentence Transformers
- 🚨 Detect contradictions (negations, numerical conflicts, term mismatches)
- 📊 Confidence scores for each detection
- 🎨 Interactive Streamlit UI with color-coded results

## Installation

```bash
# Clone repository
git clone <repository-url>
cd legal-inconsistency-detector

# Install dependencies
pip install -r requirements.txt

# Download NLTK data (first run only)
python -c "import nltk; nltk.download('punkt')"