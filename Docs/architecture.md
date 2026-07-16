# Architecture

## Overview

This project extends a research paper search/summarization pipeline into an autonomous agent. Data flows through five stages: retrieval, enrichment, tool wrapping, agent reasoning, and response synthesis.

## Pipeline Flow

User Query
│
▼
[FAISS Similarity Search] ──> Top-k relevant papers (title + abstract)
│
▼
[Enrichment Stage — runs per paper]
├── BART Summarizer      → concise summary
├── KeyBERT              → top keyphrases
└── Hybrid NER           → entities (Frameworks, Models, Languages, Orgs)
│
▼
[4 LangChain Tools]
├── search_papers_tool     → search + summarize + entities
├── extract_keywords_tool  → keyword extraction only
├── compare_papers_tool    → structured LLM comparison of 2 papers
└── explain_paper_tool     → beginner-friendly explanation of 1 paper
│
▼
[Groq LLM bound to tools] ──> LLM decides which tool fits the question
│
▼
[Tool executes] ──> raw result returned
│
▼
[LLM synthesizes final answer] ──> natural language response
│
▼
[ask_agent()] ──> prints answer + logs to agent_log.json

## Hybrid NER Sub-Architecture

Text Input
│
├──> Rule-Based Classifier (classify_query_entities)
│      matches against curated lists: Frameworks, Models,
│      Languages, Organizations
│
└──> HuggingFace NER (dslim/bert-base-NER)
detects general ORG/PERSON/LOC entities
│
▼
Filtering Layer
├── discard subword fragments (## prefix)
├── discard confidence < 0.95
├── discard length < 4 chars
├── discard if entity matches paper's own title
├── discard known dataset names (TIMIT, DAQUAR, etc.)
└── discard generic architecture terms
│
▼
Merged Entity Dictionary

## Why This Structure

- **Retrieval is separated from enrichment** so any of the 4 tools can reuse the same search logic without duplicating FAISS query code.
- **Hybrid NER runs on cleaned abstract text**, not the raw query — this was a deliberate change from the original rule-based-only system, which only tagged entities in the user's query, not the actual paper content.
- **The agent layer is a thin decision layer** — it doesn't contain business logic itself, it only routes to the correct tool and passes the result to the LLM for final synthesis. This keeps each tool independently testable.
