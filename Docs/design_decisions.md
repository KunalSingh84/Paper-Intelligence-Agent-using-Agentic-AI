# Design Decisions

This document explains the reasoning behind key engineering choices made while building this project, based on issues actually encountered during development and testing.

## Why Hybrid NER Instead of One Method

A purely rule-based tagger only catches entities already in its curated lists — it has no way to detect an organization it wasn't told about in advance. A purely transformer-based tagger (HuggingFace's `dslim/bert-base-NER`) can catch unlisted entities, but is a general-purpose model with no knowledge of ML-specific vocabulary, and is prone to tokenization issues on domain-specific terms.

Combining both means: reliable detection of known ML vocabulary (frameworks, models, languages) via rules, plus the ability to catch previously-unlisted organizations via the transformer model.

## Why a 0.95 Confidence Threshold

Initial testing with the default aggregation used a lower implicit threshold, which let through incorrect detections such as tagging "Torch" (a framework, not a company) and a broken subword fragment "Ten" (a leftover piece of a mis-tokenized "TensorFlow") as organizations. Raising the confidence threshold to 0.95 filtered these out while still correctly retaining high-confidence detections like "Google" (0.998 confidence).

## Why Subword Fragment Filtering

BERT-based tokenizers split unfamiliar words into subword pieces (for example, "TensorFlow" was sometimes split into `Ten`, `##sor`, `##Flow`). These fragments are marked with a `##` prefix by the tokenizer. Any word starting with `##` is discarded, since it represents an incomplete piece of a larger word rather than a standalone entity.

## Why Self-Reference Filtering

Testing on the paper "LightNet: A Versatile, Standalone Matlab-based Environment for Deep Learning" showed the NER model tagging "LightNet" as an organization — but LightNet is the paper's own subject, not an external company it mentions. Entities matching words in the paper's own title are now excluded from the Organization category.

## Why a Known-Dataset Exclusion List

Benchmark dataset names are frequently mistagged as organizations by general-purpose NER models, since they are capitalized proper nouns with no other distinguishing signal. This was observed directly with "DAQUAR" (an image question-answering dataset) and "TIMIT" (a speech dataset), both initially tagged as Organizations. A small exclusion list (`daquar`, `timit`, `mnist`, `imagenet`, `cifar`, `vqa`) filters these out.

## Why `llm.bind_tools()` Instead of `create_tool_calling_agent()`

LangChain's `create_tool_calling_agent()` and `AgentExecutor` classes (used in some LangChain tutorials) are not available in LangChain 1.x, which restructured how agents are built. Since the installed environment used LangChain 1.3.13, `llm.bind_tools(tools)` was used instead — a more direct approach where the LLM is given the tool definitions and returns which tool it wants to call, which is then executed manually. This achieves the same outcome with fewer dependencies on a specific LangChain agent class.

## Why a Separate Structured Prompt for Paper Comparison

An early version of `compare_papers_tool` simply returned the raw abstracts of two papers, leaving the LLM to interpret them however it chose in the final synthesis step. This produced comparisons of inconsistent quality and structure. The tool was rewritten to build an explicit prompt asking the LLM to compare the two papers across four fixed criteria — Research Objective, Methodology, Key Contributions, and Limitations — producing a consistently structured comparison every time.

## Why Error Handling Was Added

The initial version of `ask_agent()` had no error handling — if the LLM ever returned zero tool calls for an ambiguous question, the function would crash with an unhandled `IndexError`. A `try/except` block was added around the core logic so the function fails gracefully with a readable message instead of an unhandled traceback.
