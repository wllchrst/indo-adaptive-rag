📋 Major Revision Plan for "Bridging Language Gaps with Adaptive RAG"
Overview
The paper has received major revision feedback from 3 reviewers with ~50 specific points. The main weaknesses are:
1. No translation quality validation (BLEU, chrF, human evaluation)
2. Insufficient multi-retrieval failure analysis (superficial post-hoc analysis only)
3. Circular labeling dependency (classifier trained on imperfect RAG performance)
4. Missing critical baselines (always-multi, dense retrieval, hybrid approaches)
5. No statistical significance testing and reproducibility issues
---
🎯 Priority 1: Critical Revisions (Must Complete)
1. Translation Quality Evaluation
Time Estimate: 1-2 weeks
Tasks:
- Calculate BLEU and chrF scores for translated HotpotQA
- Implement human evaluation protocol:
  - Sample 100-200 translated QA pairs
  - Rate fluency and semantic accuracy
  - Measure inter-annotator agreement
  - Correlate translation quality with QA performance
- Document translation artifacts that affect reasoning
- Update Section 3.1 with quantitative metrics
Files to create/modify:
- analysis/translation_evaluation.py (new)
- analysis/human_evaluation_protocol.md (new)
- XXXX_SOIC-template.tex (update Section 3.1, 4.1)
---
2. Deep Multi-Retrieval Failure Analysis
Time Estimate: 2-3 weeks
Tasks:
- Ablation studies:
  - Test with gold retrieved documents (English, translated) → isolates retrieval vs reasoning
  - Limit retrieved documents per step (1, 3, 5)
  - Implement re-ranking for retrieved snippets
  - Test alternative multi-hop strategies (SelfAsk, Decompose)
- Categorize error types:
  - Hallucinations count
  - Failed keyword generation
  - Irrelevant context retrieval
  - Answer format issues
- Fix termination logic:
  - Replace keyword matching with evidence convergence detection
  - Implement repetition detection
  - Add logical stopping rules
- Formalize algorithm (pseudocode block)
- Add retrieval hit-rate analysis
Files to create/modify:
- analysis/multiretrieval_ablation.py (new)
- analysis/error_categorization.py (new)
- methods/multistep_retrieval.py (major refactor)
- XXXX_SOIC-template.tex (add algorithm block, update Section 4.4, 4.5)
---
3. Label Validation Process
Time Estimate: 1-2 weeks
Tasks:
- Human audit of labels (50 examples per class A/B/C)
- Validate automated labels align with human intuition
- Discuss circular dependency limitation explicitly in paper
- Report train/validation/test splits
- Ensure no data leakage between classifier training and QA evaluation
- Report results on both balanced and original distributions
Files to create/modify:
- analysis/label_validation.py (new)
- analysis/human_audit_protocol.md (new)
- training_classifier/data_loader.py (explicit splits)
- XXXX_SOIC-template.tex (add limitation discussion, Section 3.4)
---
4. Add Critical Baselines
Time Estimate: 2-3 weeks
Tasks:
- Always-Multi baseline for HotpotQA
- Single-Retrieval-Only baseline for IndoQA/QASiNa
- Stronger LLM baseline:
  - GPT-4o-mini (API-based) as performance ceiling
  - Or LLaMA 3 / Aya-23 (multilingual)
- Cross-lingual analysis:
  - Run Adaptive RAG on original English HotpotQA
  - Compare performance penalty from language shift
Files to create/modify:
- final_experiment/baseline_comparison.py (new)
- final_experiment/config.py (add new models)
- llm/openai_wrapper.py (new if using GPT-4o-mini)
- XXXX_SOIC-template.tex (new baseline results table)
---
5. Dense Retrieval + Hybrid Retrieval
Time Estimate: 1-2 weeks
Tasks:
- Implement DPR (Dense Passage Retrieval) or Contriever
- Use multilingual embeddings (mBERT, XLM-R, or Indonesian-specific)
- Implement hybrid retrieval (BM25 + dense, weighted or reranked)
- Compare retrieval recall before QA generation
- Justify BM25 for Indonesian or add normalization/stemming
Files to create/modify:
- bm25/indonesian_preprocessing.py (new - stemming, normalization)
- vector_database/dense_retrieval.py (enable/disable dense retrieval)
- methods/hybrid_retrieval.py (new)
- XXXX_SOIC-template.tex (update Section 3.2, add comparison)
---
6. Statistical Significance Testing
Time Estimate: 1 week
Tasks:
- Implement bootstrap testing for EM/F1 differences
- Or paired t-tests (bootstrapping recommended)
- Report p-values for key comparisons
- Run repeated experiments with fixed random seeds
Files to create/modify:
- analysis/statistical_significance.py (new)
- final_experiment/system.py (fix random seeds, report them)
- XXXX_SOIC-template.tex (add significance column to tables)
---
7. Reproducibility Fixes
Time Estimate: 3-5 days
Tasks:
- Fix random seeds in all experiments (explicitly report)
- Use deterministic decoding (temperature=0) during labeling
- Fully specify all decoding parameters (temperature, top_k, top_p)
- Clarify time measurement (end-to-end vs generation-only)
- Report dataset sizes before/after balancing
Files to create/modify:
- helpers/random_seed.py (new utility)
- classification/classify.py (set temperature=0)
- methods/base_method.py (specify params)
- XXXX_SOIC-template.tex (add parameter table)
---
🎯 Priority 2: High Priority Improvements
8. Reposition HotpotQA as Failure Analysis
- Rewrite Section 4.3.3 as diagnostic analysis, not performance evaluation
- Move from "Results" to "Error Analysis" section
- Discuss as limitation case study
9. Formalize Multi-Retrieval Algorithm
- Add Algorithm 1 pseudocode in paper
- Include query update, top-k selection, context aggregation
- Specify termination conditions
10. Per-Class Classifier Metrics
- Report precision, recall, F1 for each class A/B/C
- Analyze misclassification cost between classes
- Add calibration analysis or confidence thresholding
11. Redefine Adaptive Objective
- Frame as cost-quality trade-off problem
- Introduce utility function: U = α·Accuracy - β·Cost
- Report on utility improvement vs pure accuracy
---
🎯 Priority 3: Medium/Low Priority
12. Remove/Improve Synonym Augmentation
- Replace with linguistically constrained augmentation
- Or remove entirely
- If kept: specify source, POS filtering, entity-preservation
13. Strengthen Novelty Positioning
- Explicitly contrast with existing adaptive RAG frameworks
- State what's fundamentally different
- Isolate classifier contribution as main technical outcome
14. Improve Discussion Section
- State main bottleneck: multi-retrieval in Indonesian
- Specific future work directions:
  - Fine-tune LLMs on Indonesian CoT data
  - Hybrid retrieval improvements
  - Native Indonesian multi-hop dataset
15. Paper-Writing Improvements
- English proofreading
- Replace blog references with authoritative sources
- Add dataset licensing and ethical considerations
- Separate "problem gap" from "research objectives" (Reviewer 3)
- Add English captions to all figures (Reviewer 3)
---
📅 Suggested Timeline (12-16 weeks total)
| Week | Tasks | Deliverables |
|------|-------|--------------|
| 1-2 | Translation evaluation | BLEU/chrF scores, human eval protocol |
| 3-4 | Label validation | Human audit results, explicit splits |
| 5-7 | Multi-retrieval ablation | Ablation study results, error analysis |
| 8-9 | Add baselines (always-multi, strong LLM) | Baseline comparison results |
| 10-11 | Dense/hybrid retrieval | Retrieval comparison, recall analysis |
| 12 | Statistical significance | p-values, confidence intervals |
| 13-14 | Reproducibility & minor improvements | Fixed seeds, parameter docs |
| 15-16 | Paper updates & proofreading | Revised manuscript, rebuttal letter |
---
🚦 Immediate Next Steps
1. Start with translation evaluation - foundational for all downstream analysis
2. Set up reproducibility framework - fix random seeds, add logging
3. Design ablation study experiments - plan before implementing