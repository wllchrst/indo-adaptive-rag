# Planning of what should be done in each task.

## HotpotQA Translation task

### Evaluating the translation result of HotpotQA dataset.

The evaluation process can be done automatically using score such as BLEU, chrF, or COMET scores. The evaluation should be done separately for each column and be reported for the table

### Conduct human evaluation

Automatic evaluation process is sometime not enough for looking at the result of the translation and the resulting text are not natural which is not enough for a real implementation of question answering making the dataset not valid for the research.

Randomly pick rows in the translated dataset and have native Indonesian speakers evaluate things like fluency, semantic accuracy, and entity representation (are names and entities from the original text kept by the translation result)

A better result from this is the human evaluated texts can be used later in the next section.

### Does the translation process impact the performance of the RAG system.

By using the rows evaluated manually by human it can be used to show whether the translation process or RAG system that have been created that causes the error.

### Does the current translation model being used is the best one for translating english to indonesia?

The reviewer is afraid that the translation result is bad due to the use of a bad translation model.

### Multihop preservation after translation.

Does the multihop question keep being a multihop question after being translated to Indonesian language?

---
## Labeling Process

### Human evaluation

The first problem of labeling process is happening because the circular dependency problem, dataset being used to train the classifier depends on the imperfect RAG system that is being used at the end of the paper. By using the same RAG system this can create a domino effect where the labeled dataset is not really good for performing the next steps just because the RAG system used is not a good one.

Hence the need of human evaluation, to find out whether the labeling process is a beneficial one the use of human evaluation is important to judge whether the result of labeling process is good or not.

After there is a manually labeled dataset by humans, it can be shown by looking at if its accuracy is high or not when the human labeled dataset is used as the golden answer.

! Needs to explicitly report the limitation of the labeling process because of the circular dependency

### Use deterministic decoding for labeling process

There is variable like temperature and other things that make the result of LLM to be not consistent and have a variation, to ensure the LLM's result is consistent parameters like temperature, top_p, top_k is set to values that make the LLM to be consistent (and this also need to be reported in the paper after the implementation)

### Report the class distribution balancing for the dataset

The reviewer needs transparency of the dataset used for training the LLM.

### Report the classifier performance but per class

Current paper only show the overall accuracy and f1 of the classifier, not the overall performance specific for each class.

---
## Final Experiment changes

### Bootstrap testing

Because the natural variability of LLM's output the reviewer ask to do bootstrap testing, essentially removing randomness for the evaluation process.

### Fix reproduciblity issues

The big problem is the runs done on the final experiment is not documented perfectly there are lot of parameters of the model that is not mentioned in the research paper (to be honest i forgot about this while doing the experiment so there is a need to redo all the experiment).

### LLM or the RAG System?

Because of the bad result especially in the method for multi retrieval. There is a need to find out what causes the problem of the bad result the LLM performance in understanding Indonesian langauge or the RAG system is the one at fault.

### Retrieval hit rate analysis

To find out whether the bad result is caused by the retrieval method used for this system. For each retrieval method needs to find how many golden facts were in top-k retrieved documents.

If it turns out that the hit rate is high the problem is on the side of the LLM or the prompting used not a problem from retrieval method.

### Reposition HotpotQA results as a failure analysis

Explain more in the paper that the result in HotpotQA is a problem because of what....., what pattern is found in failures, which entity confuse the system, do automatic translation results in a bad experiment result for hotpot qa.

### Shifts from Accuracy to efficiency

Adaptive RAG is always about efficiency, how to always pick the best retrieval system for getting the best accuracy. There is still a need to showcase the accuracy result of each model. But the most important part is to graph the efficiency and accuracy correlation between using this RAG system types.

### Detail analysis on the failure on multi-retrieval method for this research

We can sample 100 multi retrieval results and find out what is the main problem and pattern experienced by the multi-retrieval method. It can be categorized into this list:

- Hallucination
- Keyword failure: fails to generate search keywords
- Wrong format for answering
- Irrelevant context from retrieved documents.
- Termination early: stops before finding answer
- Termination late: loops excessively
- Answer present: but not extracted correctly

### Report in detail dataset splits and class distributions.

### Report per class performance for all the experiment.

### Retrieval method

Use semantic approaches, the reviewer demands it. Or even combine both approach to usually the combination of semantic and lexical approach is really good