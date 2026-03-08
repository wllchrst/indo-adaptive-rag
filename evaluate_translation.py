import pandas as pd
import json
from datasets import load_dataset
from evaluate import load as load_metric
from comet import download_model, load_from_checkpoint
import numpy as np
from typing import Dict, List, Tuple
import ast


class TranslationEvaluator:
    def __init__(self):
        print("Loading evaluation metrics...")
        self.bleu = load_metric("bleu")
        self.chrf = load_metric("chrf")
        print("Loading COMET model...")
        self.comet_model = load_from_checkpoint(download_model("Unbabel/wmt22-comet-da"))
        print("All metrics loaded!")

    def transform_context(self, context_dict: dict) -> list:
        """Transform English context dict to Indonesian format (list of dicts)"""
        return [
            {'title': t, 'sentences': s}
            for t, s in zip(context_dict['title'], context_dict['sentences'])
        ]

    def transform_supporting_facts(self, sf_dict: dict) -> list:
        """Transform English supporting_facts dict to Indonesian format (list of titles)"""
        return sf_dict['title']

    def load_validation_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load Indonesian validation data and English original data"""
        print("\nLoading Indonesian validation data...")
        indonesian_df = pd.read_csv('hotpot/validation.csv')
        print(f"Loaded {len(indonesian_df)} rows from Indonesian validation data")

        print("Loading English validation data from hotpot_qa dataset...")
        hotpot_qa = load_dataset('hotpot_qa', 'fullwiki', trust_remote_code=True)
        english_data = hotpot_qa['validation']
        print(f"Loaded {len(english_data)} rows from English validation data")

        print("Creating English DataFrame...")
        english_list = []
        for row in english_data:
            english_list.append({
                'id': row['id'],
                'question': row['question'],
                'contexts': self.transform_context(row['context']),
                'supporting_facts': self.transform_supporting_facts(row['supporting_facts']),
                'answer': row['answer']
            })
        english_df = pd.DataFrame(english_list)

        print("Merging English and Indonesian data by ID...")
        merged_df = english_df.merge(indonesian_df, on='id', suffixes=('_en', '_id'))
        print(f"Merged to {len(merged_df)} rows")

        return english_df, indonesian_df

    def flatten_contexts(self, contexts: list, field: str) -> List[str]:
        """Flatten contexts list to extract titles or sentences"""
        result = []
        for context in contexts:
            if isinstance(context, dict) and field in context:
                if field == 'title':
                    result.append(context[field])
                elif field == 'sentences':
                    result.extend(context[field])
        return result

    def evaluate_text_column(self, references: List[str], predictions: List[str]) -> Dict:
        """Evaluate a text column using BLEU, chrF, and COMET"""
        if not references or not predictions:
            return {
                'bleu': None,
                'chrf': None,
                'comet': None,
                'count': 0
            }

        # Filter out None values
        valid_pairs = [(r, p) for r, p in zip(references, predictions) if r is not None and p is not None]
        if not valid_pairs:
            return {
                'bleu': None,
                'chrf': None,
                'comet': None,
                'count': 0
            }

        valid_refs = [r for r, _ in valid_pairs]
        valid_preds = [p for _, p in valid_pairs]

        result = {'count': len(valid_pairs)}

        try:
            # BLEU score
            bleu_result = self.bleu.compute(
                predictions=[[p] for p in valid_preds],
                references=[[r] for r in valid_refs]
            )
            result['bleu'] = bleu_result['bleu']
        except Exception as e:
            print(f"Error computing BLEU: {e}")
            result['bleu'] = None

        try:
            # chrF score
            chrf_result = self.chrf.compute(
                predictions=valid_preds,
                references=valid_refs
            )
            result['chrf'] = chrf_result['score']
        except Exception as e:
            print(f"Error computing chrF: {e}")
            result['chrf'] = None

        try:
            # COMET score
            comet_data = [
                {"src": "", "mt": pred, "ref": ref}
                for ref, pred in zip(valid_refs, valid_preds)
            ]
            comet_result = self.comet_model.predict(comet_data, batch_size=8, gpus=1)
            result['comet'] = np.mean(comet_result.scores)
        except Exception as e:
            print(f"Error computing COMET: {e}")
            result['comet'] = None

        return result

    def evaluate_all_columns(self, english_df: pd.DataFrame, indonesian_df: pd.DataFrame) -> Dict:
        """Evaluate all translated columns"""
        print("\n" + "="*50)
        print("Evaluating Translation Quality")
        print("="*50)

        merged_df = english_df.merge(indonesian_df, on='id', suffixes=('_en', '_id'))

        results = {}

        # 1. Evaluate question column
        print("\n[1/5] Evaluating 'question' column...")
        question_results = self.evaluate_text_column(
            merged_df['question_en'].tolist(),
            merged_df['question_id'].tolist()
        )
        results['question'] = question_results
        self.print_column_results('question', question_results)

        # 2. Evaluate answer column
        print("\n[2/5] Evaluating 'answer' column...")
        answer_results = self.evaluate_text_column(
            merged_df['answer_en'].tolist(),
            merged_df['answer_id'].tolist()
        )
        results['answer'] = answer_results
        self.print_column_results('answer', answer_results)

        # 3. Evaluate contexts titles
        print("\n[3/5] Evaluating 'contexts titles' column...")
        context_titles_en = []
        context_titles_id = []
        for _, row in merged_df.iterrows():
            contexts_en = row['contexts_en'] if pd.notna(row['contexts_en']) else []
            contexts_id = row['contexts_id'] if pd.notna(row['contexts_id']) else []
            
            try:
                if isinstance(contexts_en, str):
                    contexts_en = ast.literal_eval(contexts_en)
                if isinstance(contexts_id, str):
                    contexts_id = ast.literal_eval(contexts_id)
            except:
                contexts_en, contexts_id = [], []

            titles_en = self.flatten_contexts(contexts_en, 'title')
            titles_id = self.flatten_contexts(contexts_id, 'title')
            
            if len(titles_en) == len(titles_id):
                context_titles_en.extend(titles_en)
                context_titles_id.extend(titles_id)

        context_title_results = self.evaluate_text_column(context_titles_en, context_titles_id)
        results['context_titles'] = context_title_results
        self.print_column_results('context_titles', context_title_results)

        # 4. Evaluate contexts sentences
        print("\n[4/5] Evaluating 'contexts sentences' column...")
        context_sentences_en = []
        context_sentences_id = []
        for _, row in merged_df.iterrows():
            contexts_en = row['contexts_en'] if pd.notna(row['contexts_en']) else []
            contexts_id = row['contexts_id'] if pd.notna(row['contexts_id']) else []
            
            try:
                if isinstance(contexts_en, str):
                    contexts_en = ast.literal_eval(contexts_en)
                if isinstance(contexts_id, str):
                    contexts_id = ast.literal_eval(contexts_id)
            except:
                contexts_en, contexts_id = [], []

            sentences_en = self.flatten_contexts(contexts_en, 'sentences')
            sentences_id = self.flatten_contexts(contexts_id, 'sentences')
            
            if len(sentences_en) == len(sentences_id):
                context_sentences_en.extend(sentences_en)
                context_sentences_id.extend(sentences_id)

        context_sentence_results = self.evaluate_text_column(context_sentences_en, context_sentences_id)
        results['context_sentences'] = context_sentence_results
        self.print_column_results('context_sentences', context_sentence_results)

        # 5. Evaluate supporting facts titles
        print("\n[5/5] Evaluating 'supporting_facts titles' column...")
        fact_titles_en = []
        fact_titles_id = []
        for _, row in merged_df.iterrows():
            facts_en = row['supporting_facts_en'] if pd.notna(row['supporting_facts_en']) else []
            facts_id = row['supporting_facts_id'] if pd.notna(row['supporting_facts_id']) else []
            
            try:
                if isinstance(facts_en, str):
                    facts_en = ast.literal_eval(facts_en)
                if isinstance(facts_id, str):
                    facts_id = ast.literal_eval(facts_id)
            except:
                facts_en, facts_id = [], []

            if 'title' in facts_en and 'title' in facts_id:
                if len(facts_en['title']) == len(facts_id['title']):
                    fact_titles_en.extend(facts_en['title'])
                    fact_titles_id.extend(facts_id['title'])

        fact_title_results = self.evaluate_text_column(fact_titles_en, fact_titles_id)
        results['supporting_fact_titles'] = fact_title_results
        self.print_column_results('supporting_fact_titles', fact_title_results)

        return results

    def print_column_results(self, column_name: str, results: Dict):
        """Print results for a single column"""
        print(f"  Count: {results['count']}")
        if results['bleu'] is not None:
            print(f"  BLEU:  {results['bleu']:.4f}")
        else:
            print(f"  BLEU:  N/A")
        if results['chrf'] is not None:
            print(f"  chrF:  {results['chrf']:.4f}")
        else:
            print(f"  chrF:  N/A")
        if results['comet'] is not None:
            print(f"  COMET: {results['comet']:.4f}")
        else:
            print(f"  COMET: N/A")

    def generate_summary_table(self, results: Dict) -> pd.DataFrame:
        """Generate a summary table of all results"""
        summary_data = []
        for column_name, metrics in results.items():
            summary_data.append({
                'Column': column_name,
                'Count': metrics['count'],
                'BLEU': f"{metrics['bleu']:.4f}" if metrics['bleu'] is not None else 'N/A',
                'chrF': f"{metrics['chrf']:.4f}" if metrics['chrf'] is not None else 'N/A',
                'COMET': f"{metrics['comet']:.4f}" if metrics['comet'] is not None else 'N/A'
            })
        return pd.DataFrame(summary_data)

    def save_results(self, results: Dict, summary_df: pd.DataFrame):
        """Save results to files"""
        # Save detailed results as JSON
        with open('hotpot/translation_evaluation_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        print("\n✓ Saved detailed results to hotpot/translation_evaluation_results.json")

        # Save summary table as CSV
        summary_df.to_csv('hotpot/translation_evaluation_summary.csv', index=False)
        print("✓ Saved summary table to hotpot/translation_evaluation_summary.csv")

    def run_evaluation(self):
        """Run the complete evaluation pipeline"""
        print("\n" + "="*50)
        print("HotpotQA Translation Quality Evaluation")
        print("="*50)

        english_df, indonesian_df = self.load_validation_data()
        results = self.evaluate_all_columns(english_df, indonesian_df)
        summary_df = self.generate_summary_table(results)

        print("\n" + "="*50)
        print("SUMMARY TABLE")
        print("="*50)
        print(summary_df.to_string(index=False))

        self.save_results(results, summary_df)

        print("\n" + "="*50)
        print("Evaluation completed!")
        print("="*50)


if __name__ == "__main__":
    evaluator = TranslationEvaluator()
    evaluator.run_evaluation()
