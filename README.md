# Paper Intelligence Agent — using Agentic AI

![Python](https://img.shields.io/badge/Python-3.8-blue)
![LangChain](https://img.shields.io/badge/Agent-LangChain-green)
![LLM](https://img.shields.io/badge/LLM-Groq%20Llama3.1-purple)
![NER](https://img.shields.io/badge/NER-Hybrid%20(Rules%2BBERT)-orange)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)
![Dataset](https://img.shields.io/badge/Dataset-15000%20ArXiv%20Papers-orange)

> An autonomous research agent that searches, summarizes, compares, and explains ML research papers using a hybrid NER system and a LangChain tool-calling agent powered by Groq's LLaMA 3.1.

---

## 📋 Table of Contents
- [About the Project](#about-the-project)
- [Dataset](#dataset)
- [Features](#features)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [How It Works](#how-it-works)
- [Results](#results)
- [How to Run](#how-to-run)
- [What I Learned](#what-i-learned)
- [Author](#author)

---

## 📌 About the Project

Finding, understanding, and comparing research papers manually is slow. This project builds an **autonomous agent** that understands natural language questions and automatically decides which capability to use — search, summarize, compare, or explain — instead of requiring the user to call each function manually.

At its core is a **hybrid Named Entity Recognition system**, combining a custom rule-based tagger with a HuggingFace transformer model, actually integrated into the paper retrieval pipeline so every result comes tagged with real entities (Frameworks, Models, Programming Languages, Organizations).

---

## 📂 Dataset
| Property | Details |
|----------|---------|
| Source | ArXiv ML Papers (via HuggingFace, `CShorten/ML-ArXiv-Papers`) |
| Size | 15,000 research papers (subset of ~100,000 ML-tagged ArXiv papers) |
| Fields | Title, Abstract |
| Domain | Machine Learning and AI |

---

## ✨ Features

**1. 🔍 Semantic Search**
Finds relevant papers by meaning, not just keywords, using SentenceTransformer embeddings and FAISS.

**2. 📝 AI Summarization**
Generates concise summaries of each paper's abstract using Facebook's BART model.

**3. 🏷️ Keyword Extraction**
Extracts top keyphrases using KeyBERT with confidence filtering (score > 0.5).

**4. 🤖 Hybrid Named Entity Recognition**
Combines two methods to tag technical entities:
- **Rule-based tagger** — matches curated lists: Frameworks, Models, Programming Languages, Organizations
- **HuggingFace transformer** (`dslim/bert-base-NER`) — catches organizations the rule-based list can't anticipate

Filtered for accuracy using a confidence threshold, subword-fragment detection, self-reference filtering, and a known-dataset exclusion list (to avoid mistagging things like TIMIT or DAQUAR as companies).

**5. 🤝 Tool-Calling Agent**
Four tools, auto-selected by the LLM based on your question:
| Tool | Purpose |
|------|---------|
| `search_papers_tool` | Search + summarize + tag entities |
| `extract_keywords_tool` | Extract keyphrases from any text |
| `compare_papers_tool` | Structured LLM comparison of 2 papers (Objective, Methodology, Contributions, Limitations) |
| `explain_paper_tool` | Beginner-friendly explanation of a paper |

**6. 🛡️ Error Handling + Logging**
Every question is wrapped in error handling, and every successful answer is logged (question, tool used, answer) to `agent_log.json`.

---

## 📊 Project Structure

📦 Paper-Intelligence-Agent-using-Agentic-AI
┣ 📁 Notebook
┃ ┗ Paper_Intelligence_Agent.ipynb
┣ 📁 Docs
┃ ┣ architecture.md
┃ ┗ design_decisions.md
┣ 📁 Sample_Outputs
┃ ┗ agent_log.json
┣ 📄 requirements.txt
┗ 📄 README.md

---

## 🔧 Technologies Used
| Tool | Purpose |
|------|---------|
| Python | Core programming language |
| HuggingFace Datasets | Load 15,000 ArXiv ML papers |
| SentenceTransformers | Semantic embeddings (all-MiniLM-L6-v2) |
| FAISS | Fast similarity search |
| Facebook BART | Abstractive summarization |
| KeyBERT | Keyword extraction |
| HuggingFace Transformers | `dslim/bert-base-NER` for hybrid NER |
| LangChain | Tool wrapping + agent orchestration |
| Groq (LLaMA 3.1) | LLM powering the agent and tool-calling |
| Regex (re) | Rule-based NER + LaTeX text cleaning |

---

## ⚙️ How It Works

1. User asks a natural language question via `ask_agent(question)`
2. The Groq LLM (bound to 4 tools) decides which tool fits the question
3. The selected tool runs:
   - `search_papers_tool` → FAISS search → summarize → hybrid NER on each paper
   - `extract_keywords_tool` → KeyBERT on given text
   - `compare_papers_tool` → finds 2 papers, asks LLM for structured comparison
   - `explain_paper_tool` → finds 1 paper, asks LLM for a beginner-friendly explanation
4. The tool's raw result is passed back to the LLM, which writes a natural language final answer
5. The answer is printed, and the interaction is logged to `agent_log.json`

---

## 📈 Results

Sample query:
```python
ask_agent("Compare papers about GANs")
```

Output (abridged):

Paper 1: Learning to Avoid Errors in GANs by Manipulating Input Spaces

Focus: Reducing visual artifacts via input space manipulation

Paper 2: The Numerics of GANs

Focus: Analyzing convergence properties of GAN training algorithms

Comparison: Both aim to improve GAN stability, but Paper 1 targets
image-quality artifacts while Paper 2 addresses training convergence.
Tool used: compare_papers_tool
---

## ▶️ How to Run

1. Clone the repository : git clone https://github.com/KunalSingh84/Paper-Intelligence-Agent-using-Agentic-AI.git
2. Install required libraries : pip install -r requirements.txt
3. Open `Notebook/Paper_Intelligence_Agent.ipynb` in Google Colab

4. Add your Groq API key to Colab secrets as `GROQ_API_KEY`

5. Run all cells sequentially from top to bottom

6. Call the agent with your question:
```python
ask_agent("your question here")
```

> **Note:** First run will generate embeddings for all 15,000 papers (~30-35 minutes). After that, it loads instantly from saved files.

---

## 💡 What I Learned
- How to combine rule-based and transformer-based NER into a working hybrid system, and why filtering (confidence, subwords, self-reference) matters for real accuracy
- How LangChain tool-calling works, and how to bind tools to an LLM so it can reason about which one to use
- How to debug real issues in NLP pipelines — tokenization artifacts, false-positive entities, LaTeX-polluted text
- How to structure an agent so each tool is independently testable, with the LLM only handling routing and synthesis
- How to design for failure — adding error handling and logging instead of assuming every call succeeds

---

## 👤 Author
Kunal Singh

[![GitHub](https://img.shields.io/badge/GitHub-KunalSingh84-black?logo=github)](https://github.com/KunalSingh84)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-kunalsingh2127-blue?logo=linkedin)](https://www.linkedin.com/in/kunalsingh2127/)
