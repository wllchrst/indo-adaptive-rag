# Guide for Human Evaluation of Labeling Process

## Overview
You will evaluate the quality of automatically generated labels for an Indonesian Question Answering dataset. The labels (A, B, C) classify questions based on their retrieval complexity.

## Your Role
As a native Indonesian speaker, you will assess whether the automatically assigned labels are appropriate and accurate for each question.

## What You Need to Do

### 1. Review 150 Questions (50 per class)
- Each question has been automatically labeled as Class A, B, or C
- You will determine if the label is correct based on the question's complexity

### 2. Label Class Definitions

**Class A (Simple Retrieval)**
- Single fact answer
- Direct question-answer mapping
- Example: "Siapa presiden pertama Indonesia?"
- Expected: Answer directly available in one context passage

**Class B (Multi-hop Simple)**
- Requires 2-3 reasoning steps
- Multiple related facts needed
- Example: "Siapa presiden yang menandatangani kemerdekaan Indonesia dan kapan itu terjadi?"
- Expected: Multiple retrievals needed but straightforward chain

**Class C (Complex Multi-hop)**
- Requires 4+ reasoning steps
- Involves bridging entities, comparisons, or complex inference
- Example: "Di antara presiden Indonesia, siapa yang memiliki masa jabatan terpanjang dan berapa lama?"
- Expected: Multi-step reasoning with entity connections

### 3. For Each Question, Evaluate:

**a) Label Correctness**
- ✓ Correct: The automated label matches your judgment
- ✗ Incorrect: The question belongs to a different class
- ? Ambiguous: The question could reasonably fit multiple classes

**b) Justification** (Briefly explain your reasoning)
- Why is this label correct/incorrect?
- What aspects make it simple or complex?

### 4. Evaluation Process

**Step 1:** Read the question in Indonesian
**Step 2:** Consider what's needed to answer it
**Step 3:** Determine the appropriate class (A/B/C)
**Step 4:** Compare with the automated label
**Step 5:** Mark Correct/Incorrect/Ambiguous
**Step 6:** Add brief justification

### 5. Example Evaluation

**Question:** "Apa ibu kota Indonesia?"
**Automated Label:** A

**Your Evaluation:**
- Label Correctness: ✓ Correct
- Justification: This is a simple factual question requiring one piece of information directly available in a single context passage.

---

**Question:** "Siapa penemu bola lampu dan kapan itu terjadi?"
**Automated Label:** B

**Your Evaluation:**
- Label Correctness: ✓ Correct
- Justification: Requires two pieces of information (who and when) but they're directly related and straightforward to retrieve.

---

**Question:** "Di antara presiden Indonesia, siapa yang memiliki masa jabatan terpanjang dan berapa lama, serta bagaimana ini dibandingkan dengan presiden lainnya?"
**Automated Label:** C

**Your Evaluation:**
- ✗ Incorrect (Should be B)
- Justification: While it mentions comparison, the core question is straightforward - find the longest-serving president. The comparison is secondary and doesn't require complex bridging reasoning.

### 6. Important Guidelines

- **Trust your native intuition**: If a question feels complex, it probably is
- **Focus on retrieval needs**: What would a system need to look up to answer this?
- **Be consistent**: Apply the same criteria across all questions
- **Note ambiguity**: Mark unclear cases separately
- **Consider real-world QA**: What would a human need to search for?

### 7. What Not to Do

- ❌ Don't judge the translation quality (that's a separate evaluation)
- ❌ Don't evaluate the answer quality (we're only checking labels)
- ❌ Don't overthink - trust your first instinct
- ❌ Don't spend more than 1-2 minutes per question

### 8. Format Your Response

For each question, provide:

```
Question: [text]
Automated Label: [A/B/C]
Your Label: [A/B/C]
Correctness: [✓ Correct / ✗ Incorrect / ? Ambiguous]
Justification: [1-2 sentences]
```

### 9. Completion Checklist

- [ ] Evaluated all 150 questions
- [ ] Provided label for each question
- [ ] Added justification for each evaluation
- [ ] Flagged ambiguous cases
- [ ] Noted any patterns in misclassifications

If there are any further questions, please ask first!
