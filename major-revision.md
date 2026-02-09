Reviewer 1

Bridging Language Gaps with Adaptive RAG: Improving Indonesian Language
Question Answering
The paper presents Bridging Language Gaps with Adaptive RAG: Improving
Indonesian Language Question Answering. The manuscript is clear and follows a
logical structure. However, several technical and methodological weaknesses need
to be addressed before the manuscript can be considered for publication.
1. Add a dedicated subsection evaluating translation quality using automatic metrics
(e.g., COMET or BLEU) and include a small-scale human adequacy/fluency
annotation protocol.
2. Explicitly quantify translation-induced semantic drift and discuss how it affects
reasoning complexity.
3. Remove the assumption that all HotpotQA questions remain multi-hop after
translation; perform post-translation verification or sampling-based validation.
4. Redefine question complexity labels using intrinsic criteria (number of evidence
sentences, document hops, entity transitions) instead of performance-based
labeling.
5. Decouple label generation from evaluation models to eliminate circular
dependency.
6. Fix stochastic instability by using deterministic decoding (temperature = 0) during
labeling and report this explicitly.
7. Introduce dense retrieval baselines (e.g., DPR, Contriever, or multilingual
embeddings) alongside BM25.
8. Add hybrid retrieval (BM25 + dense) and compare retrieval recall before QA
generation.
9. Justify retrieval design for Indonesian morphology or add normalization and
stemming strategies.
10. Fully specify document chunk size, overlap, indexing fields, and preprocessing
pipeline.
11. Provide a formal algorithm block for the multi-retrieval procedure, including query
update, top-k selection, and context aggregation.
12. Redesign the termination condition using logical stopping rules (evidence
convergence, repetition detection), not keyword matching.
13. Clearly specify tokenization, normalization, and evaluation scripts for Indonesian
EM/F1.
14. Add answer alias handling and normalization rules.
15. Provide systematic error analysis explaining why multi-retrieval degrades
performance.
16. Introduce retrieval hit-rate analysis (whether gold evidence appears in retrieved
documents).
17. Add ablation comparing: no-retrieval vs single-retrieval vs multi-retrieval using
identical prompts.
18. Reposition HotpotQA results as failure analysis, not performance evaluation.
19. Redefine the adaptive objective as a cost–quality trade-off problem instead of pure
accuracy optimization.
20. Introduce a utility function combining accuracy and inference cost.
21. Add statistical significance testing (bootstrap or paired tests) for EM/F1
differences.
22. Explicitly report dataset splits, class sizes before/after balancing, and
train/validation/test partitions.
23. Ensure that classifier training data does not overlap with QA evaluation samples.
24. Report results on both balanced and original class distributions.
25. Replace synonym replacement augmentation with linguistically constrained
augmentation or remove it.
26. Specify synonym source, POS filtering, and entity-preservation constraints.
27. Justify or tune weight decay through validation experiments.
28. Add hyperparameter search description or provide rationale for fixed values.
29. Report per-class precision, recall, and F1 instead of only confusion matrix
visualization.
30. Analyze misclassification cost between A↔B↔C classes.
31. Add calibration analysis or confidence thresholding for adaptive routing.
32. Fully report decoding parameters for all LLMs.
33. Fix random seeds and state them explicitly.
34. Clarify time measurement methodology (end-to-end vs generation-only).
35. Perform repeated random sampling experiments for reduced test sets and report
variance.
36. Add comparison with at least one strong multilingual or adaptive RAG baseline.
37. Soften claims about closing language gaps; reposition findings as identifying
current limitations.
38. Replace blog-based references with authoritative linguistic or demographic
sources.
39. Perform professional English proofreading to correct grammar and terminology
inconsistencies.
40. Add dataset licensing and ethical considerations for translated corpora.
41. Rewrite the conclusion to emphasize diagnostic insights, not system superiority.
42. Clearly isolate the classifier contribution as the main technical outcome.
43. Strengthen novelty positioning by explicitly contrasting with existing adaptive
RAG frameworks and stating what is fundamentally different.

Reviewer 2

his manuscript presents a timely and relevant study on adapting Adaptive Retrieval-Augmented Generation (RAG) systems for a low-resource language, Indonesian. The authors address a significant research gap by constructing an Indonesian multi-hop QA dataset via machine translation, developing a question complexity classifier, and implementing a strategy-selection mechanism for non-retrieval, single-retrieval, and multi-retrieval answer generation. The work is methodologically sound, well-structured, and provides valuable empirical insights. However, several critical weaknesses in experimental depth, data validation, and analytical rigor must be addressed to strengthen the paper's contributions and reliability.

Major Weaknesses:

The following weaknesses currently limit the impact and scientific rigor of the study:

1. Insufficient Analysis of Core Failure Mode (Multi-Retrieval)
The paper identifies the multi-retrieval method's failure as the primary reason for overall system underperformance but provides only a superficial, post-hoc analysis. The claim that LLMs "hallucinate more frequently" is descriptive, not diagnostic. A deeper, mechanistic investigation is absent:

Lack of Root-Cause Analysis: Is the failure due to (a) the LLM's limited Indonesian comprehension in reasoning tasks, (b) ineffective prompt design for the Indonesian Chain-of-Thought, (c) information overload from concatenated documents, (d) the retrieval engine (BM25/ElasticSearch) returning poor-quality snippets for Indonesian queries, or (e) a combination thereof?

Missing Ablation Studies: No experiments modify components of the multi-retrieval pipeline (e.g., changing the CoT prompt language/style, limiting retrieved documents per step, implementing re-ranking, or trying a different retrieval algorithm like DPR) to isolate the point of failure.

2. Unquantified and Unvalidated Translation Data Quality
The entire study's foundation—the translated HotpotQA dataset—lacks rigorous quality assurance.

No Quantitative Metrics: There is no reporting of standard machine translation evaluation scores (e.g., BLEU, chrF, TER) to quantify the translation's fidelity to the source.

No Impact Assessment: The potential effect of translation errors on downstream task performance is not measured. Are low scores on HotpotQA due to the Adaptive RAG system's flaws or noisy input data? A small-scale human evaluation of translation quality and its correlation with QA accuracy is needed.

Potential Data Leakage: The translation process for the training and test sets is not described in sufficient detail to rule out contamination.

3. Limited and Unjustified Experimental Scope
The experimental design lacks comparisons that are essential for contextualizing the findings.

Lack of Baseline Comparisons: The Adaptive RAG system is not compared against a non-adaptive, always-multi-retrieval baseline on the HotpotQA dataset. This makes it impossible to conclude whether the "adaptive" component provides any benefit for complex questions.

Narrow Model Selection: Experiments are conducted only with Gemma 3-4B and Qwen 3-8B. No justification is provided for omitting:

Larger, more capable multilingual models (e.g., GPT-4, Claude-3) as an upper-bound reference.

Other open-source models optimized for Indonesian (e.g., NusaBERT, IndoLLM variants).

The original Adaptive RAG results on English data for a cross-lingual performance gap analysis.

Incomplete Retrieval Strategy Evaluation: Only the IRCoT-based multi-retrieval is tested. Other published multi-hop strategies (e.g., Self-Ask, Decomposed Prompting) are not implemented for comparison on the Indonesian dataset.

4. Questionable Labeling Protocol for Classifier Training
The "silver standard" labels (A, B, C) for question complexity are generated automatically based on the performance of the three RAG strategies.

Circular Dependency Risk: The classifier is trained to predict which strategy works best, which is determined by the same imperfect RAG components being evaluated. This creates a potential feedback loop.

No Human Validation: The automated labels are not sampled and validated by human annotators to ensure they correspond to intuitive notions of "complexity" (e.g., number of reasoning hops, need for intersectional knowledge).

5. Underdeveloped Discussion of Real-World Applicability
The conclusion and future work sections are generic. The study misses an opportunity to discuss:

The practical implications of the high latency and cost associated with the failing multi-retrieval method.

Specific deployment challenges for Indonesian Adaptive RAG in real-world applications (e.g., integrating with local knowledge bases, handling colloquial language).

Recommendations for Improvement:

To address the weaknesses above, the authors should undertake the following major revisions:

1. Conduct a Deep Dive into Multi-Retrieval Failures.

Perform ablation studies on the multi-retrieval pipeline. For example, test if providing gold retrieved documents (from the English dataset, translated) improves performance, isolating the retrieval vs. reasoning problem.

Analyze error cases categorically: count instances of hallucination, failure to generate keywords, and retrieval of irrelevant context.

Experiment with at least one alternative multi-hop QA strategy (e.g., Self-Ask) as a point of comparison.

2. Rigorously Evaluate the Translated Dataset.

Report BLEU and chrF scores for the OPUS-MT translation of a held-out portion of HotpotQA.

Conduct a human evaluation of 100-200 randomly sampled translated QA pairs, rating for fluency and semantic accuracy. Report the agreement rate and correlation with model performance on those samples.

Briefly discuss the limitations of using translated data versus natively authored Indonesian multi-hop questions.

3. Expand the Experimental Framework.

Add Critical Baselines: Include an "Always-Multi" baseline for HotpotQA. Compare Adaptive RAG against a simple "Single-Retrieval-Only" system on IndoQA and QASiNa to quantify the adaptive component's added value.

Broaden Model Comparisons: Include results from one larger API-based model (e.g., GPT-4o-mini) to establish a performance ceiling. If possible, include a multilingual model like LLaMA 3 or Aya-23.

Perform Cross-Lingual Analysis: Run the same Adaptive RAG system pipeline (with an English classifier) on the original English HotpotQA test set. This will clearly show the performance penalty attributable to the language shift versus the methodology itself.

4. Validate and Refine the Labeling Process.

Perform a human audit on a stratified sample (e.g., 50 examples per class A, B, C) to validate that the automated labels align with human judgment of complexity.

In the manuscript, explicitly discuss the limitations of the automated labeling protocol and its potential impact on classifier reliability.

5. Strengthen the Discussion and Future Work.

Explicitly state the main bottleneck identified: The primary challenge for Indonesian Adaptive RAG is not the adaptive classification, but the weakness of current LLMs in conducting iterative retrieval-and-reasoning in Indonesian.

Future work should be more specific. Instead of "improving the multi-retrieval method," propose concrete directions: e.g., "fine-tuning LLMs on Indonesian CoT data," "developing hybrid retrieval that combines lexical search with multilingual embeddings," or "creating a natively Indonesian multi-hop dataset to avoid translation artifacts."

Overall Recommendation

This manuscript tackles an important problem and presents a solid foundational framework. However, in its current form, it provides more of a proof-of-concept and problem identification than a conclusive study. The significant weaknesses related to experimental depth, validation, and analysis undermine the strength of its conclusions.

Decision: Major Revision Required.
I recommend acceptance only after the authors have thoroughly addressed the concerns above. The revisions should include new experiments, data analyses, and a substantially rewritten results/discussion section that provides deeper insight rather than just reporting scores.

Reviewer 3:
1. Explicitly separate “problem gap” from “research objectives” in the Introduction. for example, Add a short paragraph that explicitly states: - what is unknown or insufficient in prior work (Adaptive RAG studied only in English, no Indonesian multi-hop benchmark, unclear behavior of multi-retrieval in low-resource settings),- why this gap matters scientifically and practically. 2. all figures should also contains english language (so reader do not need to re read the description) 3. The paper assumes Adaptive RAG is the natural choice, but the rationale is indirect. Add a brief justification explaining:- why static RAG is insufficient,- why cost-aware or complexity-aware retrieval is the correct axis of adaptation,- why Adaptive RAG is suitable for Indonesian despite language imbalance. 4. Currently, the paper mixes all three. State explicitly that the study is:- primarily evaluative (testing Adaptive RAG behavior in Indonesian),- secondarily diagnostic (identifying why multi-retrieval fails),- not proposing a new retrieval algorithm.This prevents misinterpretation of novelty claims. 5. The rationale focuses heavily on dataset availability. Add one or two sentences highlighting linguistic and modeling challenges:- Indonesian morphology and paraphrasing,- translation noise propagation into retrieval,- reasoning degradation under long Indonesian prompts.This shows the problem is structural, not merely data-driven. 6. regarding Replicability / Reproducibility. The paper describes components separately (translation, labeling, classifier training, RAG evaluation). Add one consolidated algorithmic description or pseudocode that shows:- input data flow,- decision points,- outputs at each stage.This helps readers reproduce the full system behavior, not only individual parts.