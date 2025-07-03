import collections
from helpers.word_helper import WordHelper
class EvaluationHelper:
    def __init__(self):
        pass
    
    @staticmethod
    def compute_exact(a_gold: str, a_pred: str):
        return int(WordHelper.normalize_text(a_gold) == WordHelper.normalize_text(a_pred))

    @staticmethod
    def compute_f1(a_gold: str, a_pred: str):
        gold_toks = WordHelper.get_tokens(a_gold)
        pred_toks = WordHelper.get_tokens(a_pred)
        common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
        num_same = sum(common.values())
        if len(gold_toks) == 0 or len(pred_toks) == 0:
            # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
            return int(gold_toks == pred_toks)
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(pred_toks)
        recall = 1.0 * num_same / len(gold_toks)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1
    
    @staticmethod
    def compute_scores(a_gold: str, a_pred: str):
        exact = EvaluationHelper.compute_exact(a_gold, a_pred)
        f1 = EvaluationHelper.compute_f1(a_gold, a_pred)
        return {
            'exact_match': exact,
            'f1_score': f1
        }