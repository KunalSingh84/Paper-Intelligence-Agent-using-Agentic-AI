# Design Decisions

This document explains the reasoning behind key engineering choices made while building this project, based on issues actually encountered during development and testing.

## Why Hybrid NER Instead of One Method

A purely rule-based tagger only catches entities already in its curated lists — it has no way to detect an organization it wasn't told about in advance. A purely transformer-based tagger (HuggingFace's `dslim/bert-base-NER`) can catch unlisted entities, but is a general-purpose model with no knowledge of ML-specific vocabulary, and is prone to tokenization issues on domain-specific terms.

Combining both means: reliable detection of known ML vocabulary (frameworks, models, languages) via rules, plus the ability to catch previously unlisted organizations via the transformer model.

## Why a 0.95 Confidence Threshold

Initial testing with the default aggregation used a lower implicit threshold, which let incorrect detections such as tagging "Torch" (a framework, not a company) and a broken subword fragment "Ten" (a leftover piece of a mis-tokenized "TensorFlow") as organizations. Raising the confidence threshold to 0.95 filtered these out while still correctly retaining high-confidence detections like "Google" (0.998 confidence).

## Why Subword Fragment Filtering

BERT-based tokenizers split unfamiliar words into subword pieces (for example, "TensorFlow" was sometimes split into `Ten`, `##sor`, `##flow`). These fragments are marked with a `##` prefix by the tokenizer. Any word starting with `##` is discarded, since it represents an incomplete piece of a larger word rather than a standalone entity.

## Why Self-Reference Filtering

Testing on the paper "LightNet: A Versatile, Standalone Matlab-based Environment for Deep Learning" showed the NER model tagging "LightNet" as an organization — but LightNet is the paper's own subject, not an external company it mentions. Entities matching words in the paper's own title are now excluded from the Organization category.

## Why a Known-Dataset Exclusion List

Benchmark dataset names are frequently misdiagnosed as organizations by general-purpose NER models, since they're capitalized proper nouns with no other distinguishing signal. This was observed directly with "DAQUAR" (an image question-answering dataset) and "TIMIT" (a speech dataset), both initially tagged as Organizations. A small exclusion list (`daquar`, `timit`, `mnist`, `imagenet`, `cifar`, `vqa`) filters these out.

## Why `llm.bind_tools()` Instead of `create_tool_calling_agent()`

LangChain's `create_tool_calling_agent()` and `AgentExecutor` classes (used in some LangChain 1.x tutorials) are not available in LangChain 1x, which restructured how agents are built. The installed environment used LangChain 1.3.13, so `llm.bind_tools()` was used instead — a more direct approach where the LLM is given the tool definitions and returns tool calls, which is then executed manually. This achieves the same outcome with fewer dependencies.

## Why a Separate Structured Prompt for Paper Comparison

An early version of `compare_papers_tool` simply returned the raw abstracts of two papers, leaving the LLM to interpret them however it chose in the first synthesis step. This produced comparisons of inconsistent quality and structure. The tool was rewritten to build an explicit prompt asking the LLM to compare the two papers across fixed criteria — Research Objective, Methodology, Key Contributions, and Limitations — producing a consistently structured comparison every time.

## Why Error Handling Was Added

The initial version of `ask_agent()` had no error handling — if the LLM ever returned zero tool calls for an ambiguous question, the function would crash with an unhandled `IndexError`. Two other failure modes surfaced during testing: network/timeout errors from the Groq API on longer queries, and malformed tool-call arguments when the LLM's output didn't match the expected schema. Rather than handle each case separately, a broad `try/except` block was added around the core logic so the function fails gracefully with a readable message and logs the failure, instead of raising an unhandled traceback. This trades some diagnostic precision for reliability — a reasonable choice for a portfolio-stage project, though a production version would likely catch these cases separately to give more actionable error messages.

## Evaluation Notes and Limitations

These decisions were validated through manual, ad hoc testing rather than a formal labeled evaluation set — thresholds like the 0.95 confidence cutoff were tuned against a handful of known problem cases (e.g., "Google" vs. "Torch"), not measured against precision/recall on a held-out dataset. This is a reasonable tradeoff for a project at this stage, but means edge cases outside the tested examples (e.g., organization names with lower model confidence, or rare frameworks not in the curated list) may still be misclassified.
