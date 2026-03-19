#!/usr/bin/env python3
"""
Create Human Evaluation Dataset for Labeling Process

This script samples questions from multiple sources (IndoQA, QASiNa, Hotpot)
to generate a balanced dataset for human evaluation of automatic labeling.

Usage:
    python create_human_evaluation_dataset.py [options]
"""

import pandas as pd
import argparse
import sys
import os
from pathlib import Path


def load_and_normalize_indoqa(filepath, dataset_name):
    """Load IndoQA CSV file and normalize column names."""
    df = pd.read_csv(filepath)
    
    normalized = pd.DataFrame()
    normalized['ID'] = df['id']
    normalized['Question'] = df['question']
    normalized['Answer'] = df['answer']
    normalized['Label'] = df['classification']
    normalized['Source_Dataset'] = dataset_name
    
    return normalized


def load_and_normalize_qasina(filepath):
    """Load QASiNa CSV file and normalize column names."""
    df = pd.read_csv(filepath)
    
    normalized = pd.DataFrame()
    normalized['ID'] = df['ID']
    normalized['Question'] = df['question']
    normalized['Answer'] = df['answer']
    normalized['Label'] = df['classifications']
    normalized['Source_Dataset'] = 'QASiNa'
    
    return normalized


def load_and_normalize_hotpot(filepath):
    """Load Hotpot validation CSV file and normalize column names."""
    df = pd.read_csv(filepath)
    
    normalized = pd.DataFrame()
    normalized['ID'] = df['id']
    normalized['Question'] = df['question']
    normalized['Answer'] = df['answer']
    normalized['Label'] = 'C'  # All Hotpot questions are multi-hop (class C)
    normalized['Source_Dataset'] = 'Hotpot-Validation'
    
    return normalized


def sample_questions(df, label, n_samples, seed):
    """Sample n_samples from dataframe, stratified if needed."""
    label_df = df[df['Label'] == label].copy()
    
    if len(label_df) < n_samples:
        print(f"Warning: Only {len(label_df)} samples available for class {label}")
        print(f"Using all available samples instead of {n_samples}")
        n_samples = len(label_df)
    
    sampled = label_df.sample(n=n_samples, random_state=seed)
    return sampled


def sample_class_c(class_c_indo, hotpot_df, n_samples, hotpot_ratio, seed):
    """Sample class C questions from multiple sources with specified ratio."""
    n_from_hotpot = int(n_samples * hotpot_ratio)
    n_from_indo = n_samples - n_from_hotpot
    
    samples = []
    
    # Sample from existing class C (IndoQA/QASiNa)
    if n_from_indo > 0:
        if len(class_c_indo) >= n_from_indo:
            indo_samples = class_c_indo.sample(n=n_from_indo, random_state=seed)
            samples.append(indo_samples)
        else:
            print(f"Warning: Only {len(class_c_indo)} class C samples from IndoQA/QASiNa")
            print(f"Using all available and increasing Hotpot ratio")
            samples.append(class_c_indo)
            n_from_hotpot = n_samples - len(class_c_indo)
    
    # Sample from Hotpot
    if n_from_hotpot > 0:
        if len(hotpot_df) >= n_from_hotpot:
            hotpot_samples = hotpot_df.sample(n=n_from_hotpot, random_state=seed + 1)
            samples.append(hotpot_samples)
        else:
            print(f"Warning: Only {len(hotpot_df)} Hotpot samples available")
            samples.append(hotpot_df)
    
    # Combine samples
    if samples:
        result = pd.concat(samples, ignore_index=True)
        return result
    else:
        return pd.DataFrame()


def main():
    parser = argparse.ArgumentParser(
        description='Create human evaluation dataset for labeling process',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python create_human_evaluation_dataset.py
  python create_human_evaluation_dataset.py --output custom_dataset.csv --seed 123
  python create_human_evaluation_dataset.py --samples-per-class 30 --c-split-hotpot-ratio 0.7
        """
    )
    
    parser.add_argument('--output', type=str, 
                        default='human-evaluation/evaluation_dataset.csv',
                        help='Output CSV file path (default: human-evaluation/evaluation_dataset.csv)')
    parser.add_argument('--samples-per-class', type=int, default=50,
                        help='Number of samples to generate for each class (default: 50)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--c-split-hotpot-ratio', type=float, default=0.5,
                        help='Ratio of Hotpot questions in class C sampling (0.0-1.0, default: 0.5)')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Human Evaluation Dataset Generator")
    print("=" * 80)
    print(f"Samples per class: {args.samples_per_class}")
    print(f"Random seed: {args.seed}")
    print(f"Class C Hotpot ratio: {args.c_split_hotpot_ratio}")
    print(f"Output file: {args.output}")
    print()
    
    base_dir = Path(__file__).parent
    classification_dir = base_dir / 'classification_result' / 'stable-qwen'
    hotpot_dir = base_dir / 'hotpot'
    
    input_files = {
        'indoqa_test': classification_dir / 'indoqa__test.csv',
        'indoqa_train': classification_dir / 'indoqa__train.csv',
        'qasina': classification_dir / 'qasina__full.csv',
        'hotpot': hotpot_dir / 'validation.csv'
    }
    
    for name, filepath in input_files.items():
        if not filepath.exists():
            print(f"Error: Input file not found: {filepath}")
            sys.exit(1)
    
    print("Loading input files...")
    
    indoqa_test = load_and_normalize_indoqa(input_files['indoqa_test'], 'IndoQA-Test')
    print(f"  - IndoQA Test: {len(indoqa_test)} questions")
    
    indoqa_train = load_and_normalize_indoqa(input_files['indoqa_train'], 'IndoQA-Train')
    print(f"  - IndoQA Train: {len(indoqa_train)} questions")
    
    qasina = load_and_normalize_qasina(input_files['qasina'])
    print(f"  - QASiNa: {len(qasina)} questions")
    
    hotpot = load_and_normalize_hotpot(input_files['hotpot'])
    print(f"  - Hotpot Validation: {len(hotpot)} questions")
    print()
    
    indoqa_qasina = pd.concat([indoqa_test, indoqa_train, qasina], ignore_index=True)
    
    print("Class distribution in IndoQA + QASiNa:")
    for label in ['A', 'B', 'C']:
        count = (indoqa_qasina['Label'] == label).sum()
        print(f"  - Class {label}: {count}")
    print()
    
    print(f"Class C in Hotpot: {len(hotpot)} (all questions)")
    print()
    
    print("Sampling questions...")
    
    class_a_samples = sample_questions(indoqa_qasina, 'A', args.samples_per_class, args.seed)
    print(f"  - Class A: {len(class_a_samples)} questions sampled")
    
    class_b_samples = sample_questions(indoqa_qasina, 'B', args.samples_per_class, args.seed + 2)
    print(f"  - Class B: {len(class_b_samples)} questions sampled")
    
    class_c_indo = indoqa_qasina[indoqa_qasina['Label'] == 'C']
    class_c_samples = sample_class_c(class_c_indo, hotpot, args.samples_per_class, 
                                    args.c_split_hotpot_ratio, args.seed + 3)
    print(f"  - Class C: {len(class_c_samples)} questions sampled")
    
    class_c_breakdown = class_c_samples['Source_Dataset'].value_counts()
    for source, count in class_c_breakdown.items():
        print(f"    - {source}: {count}")
    print()
    
    evaluation_dataset = pd.concat([class_a_samples, class_b_samples, class_c_samples], 
                                ignore_index=True)
    evaluation_dataset = evaluation_dataset.sample(frac=1, random_state=args.seed + 10)
    evaluation_dataset = evaluation_dataset.reset_index(drop=True)
    
    print("Creating output directory...")
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving evaluation dataset to {args.output}...")
    evaluation_dataset.to_csv(args.output, index=False, encoding='utf-8')
    
    print()
    print("=" * 80)
    print("Dataset Statistics")
    print("=" * 80)
    print(f"Total questions: {len(evaluation_dataset)}")
    print()
    
    print("By class:")
    class_counts = evaluation_dataset['Label'].value_counts().sort_index()
    for label in ['A', 'B', 'C']:
        count = class_counts.get(label, 0)
        pct = (count / len(evaluation_dataset)) * 100
        print(f"  Class {label}: {count} ({pct:.1f}%)")
    print()
    
    print("By source dataset:")
    source_counts = evaluation_dataset['Source_Dataset'].value_counts()
    for source, count in source_counts.items():
        pct = (count / len(evaluation_dataset)) * 100
        print(f"  {source}: {count} ({pct:.1f}%)")
    print()
    
    print("Class breakdown by source:")
    breakdown = evaluation_dataset.groupby(['Label', 'Source_Dataset']).size().unstack(fill_value=0)
    for label in ['A', 'B', 'C']:
        if label in breakdown.index:
            print(f"  Class {label}:")
            for source, count in breakdown.loc[label].items():
                if count > 0:
                    print(f"    - {source}: {count}")
    print()
    
    stats_file = Path(args.output).with_suffix('.txt')
    print(f"Saving statistics to {stats_file}...")
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write("Human Evaluation Dataset Statistics\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Generation parameters:\n")
        f.write(f"  - Samples per class: {args.samples_per_class}\n")
        f.write(f"  - Random seed: {args.seed}\n")
        f.write(f"  - Class C Hotpot ratio: {args.c_split_hotpot_ratio}\n")
        f.write(f"  - Output file: {args.output}\n\n")
        
        f.write(f"Total questions: {len(evaluation_dataset)}\n\n")
        
        f.write("By class:\n")
        for label in ['A', 'B', 'C']:
            count = class_counts.get(label, 0)
            pct = (count / len(evaluation_dataset)) * 100
            f.write(f"  Class {label}: {count} ({pct:.1f}%)\n")
        f.write("\n")
        
        f.write("By source dataset:\n")
        for source, count in source_counts.items():
            pct = (count / len(evaluation_dataset)) * 100
            f.write(f"  {source}: {count} ({pct:.1f}%)\n")
        f.write("\n")
        
        f.write("Class breakdown by source:\n")
        for label in ['A', 'B', 'C']:
            if label in breakdown.index:
                f.write(f"  Class {label}:\n")
                for source, count in breakdown.loc[label].items():
                    if count > 0:
                        f.write(f"    - {source}: {count}\n")
    
    print()
    print("=" * 80)
    print("Dataset generation complete!")
    print("=" * 80)
    print()
    print("Next steps:")
    print("1. Review the generated CSV file")
    print("2. Share with human evaluators")
    print("3. Use human-evaluation/labeling_evaluation_guide.md as instructions")


if __name__ == '__main__':
    main()
