# Mirage: A Unified Library for Benchmarking Factual Consistency

**Authors**: _Your Name_, _Your Advisor_  
**Affiliation**: _Your Institution / Research Group_  
**Status**: Research Resource Paper Submission (2025)  
**License**: Apache 2.0

---

## 🧭 Overview

**Mirage** is a modular Python library for benchmarking factual consistency in natural language generation. It offers a standardized interface to load, evaluate, and visualize results across multiple classes of factuality metrics including:

- Classification-based (e.g., FactCC, TrueTeacher)
- QA/QG-based (e.g., QAGS, FEQA)
- Entity-based (e.g., FactAcc)

This library is designed for researchers and practitioners aiming to evaluate model outputs (e.g., summaries, translations, generated text) against reference/source texts for factual consistency.

---

## 📦 Installation

```bash
git clone https://github.com/YOUR_USERNAME/mirage.git
cd mirage
pip install -r requirements.txt
python -m spacy download en_core_web_sm  # Required for FEQA
