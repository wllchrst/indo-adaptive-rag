import pandas as pd
import json
import ast
import traceback
import re
from typing import List, Dict, Any
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset


def init_translation_model():
    """Initialize the translation model for newer transformers versions."""
    model_name = "Helsinki-NLP/opus-mt-en-id"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading translation model on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    print(f"Model loaded successfully on {device}")
    
    return tokenizer, model, device


def translate_text(text: str, tokenizer, model, device: str, max_length: int = 200) -> str:
    """Translate text from English to Indonesian."""
    try:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_length=max_length,
                num_beams=4,
                early_stopping=True
            )
        
        translated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return translated
    except Exception as e:
        print(f"Error translating '{text[:50]}...': {e}")
        return text  # Return original if translation fails


def parse_contexts(contexts_str: str) -> List[Dict[str, Any]]:
    """Parse contexts string from CSV to list of dictionaries."""
    try:
        return json.loads(contexts_str)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(contexts_str)
        except (ValueError, SyntaxError):
            print(f"Error parsing contexts: {contexts_str[:100]}...")
            return []


def serialize_contexts(contexts: List[Dict[str, Any]]) -> str:
    """Serialize contexts list back to string format."""
    return json.dumps(contexts, ensure_ascii=False)


def load_original_hotpotqa() -> Dict[str, Any]:
    """
    Load the original HotpotQA validation dataset from Hugging Face.
    
    Returns:
        Dictionary mapping id to original data
    """
    print("Loading original HotpotQA validation dataset...")
    dataset = load_dataset('hotpot_qa', 'fullwiki', split='validation')
    id_to_original = {item['id']: item for item in dataset}
    print(f"Loaded {len(id_to_original)} items from original dataset")
    return id_to_original


def fix_title_translations(csv_path: str, backup: bool = True) -> bool:
    """
    Translate ALL titles from the original HotpotQA dataset for rows in validation.csv.
    Matches rows by id and replaces existing titles with fresh translations.
    
    Args:
        csv_path: Path to the CSV file
        backup: Whether to create a backup of original file
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Initialize translation model
        tokenizer, model, device = init_translation_model()
        
        # Load original HotpotQA dataset
        original_data = load_original_hotpotqa()
        
        print(f"\nReading data from {csv_path}...")
        df = pd.read_csv(csv_path)
        
        print(f"Loaded {len(df)} rows")
        print(f"Columns: {df.columns.tolist()}")
        
        # Create backup if requested
        if backup:
            backup_path = csv_path.replace('.csv', '_backup.csv')
            print(f"\nCreating backup at {backup_path}...")
            df.to_csv(backup_path, index=False)
            print("Backup created successfully")
        
        # Process each row
        total_rows = len(df)
        titles_translated = 0
        rows_skipped = 0
        rows_processed = 0
        
        for index, row in df.iterrows():
            row_id = row['id']
            print(f"\nProcessing row {index + 1}/{total_rows} (ID: {row_id})")
            
            try:
                # Check if ID exists in original dataset
                if row_id not in original_data:
                    print(f"  Warning: ID {row_id} not found in original HotpotQA dataset, skipping")
                    rows_skipped += 1
                    continue
                
                # Get original data
                original = original_data[row_id]
                original_context = original['context']
                
                rows_processed += 1
                
                # Parse existing contexts to preserve sentences
                contexts_str = row['contexts']
                if pd.isna(contexts_str):
                    print(f"  Skipping: no contexts data in validation file")
                    continue
                
                existing_contexts = parse_contexts(contexts_str)
                
                if not existing_contexts:
                    print(f"  Skipping: empty contexts in validation file")
                    continue
                
                # Translate all original titles
                updated_contexts = []
                row_translated = 0
                
                for i, ctx in enumerate(existing_contexts):
                    # Get original title from dataset
                    original_title = original_context['title'][i] if i < len(original_context['title']) else ''
                    
                    if original_title:
                        # Translate the original title
                        translated_title = translate_text(original_title, tokenizer, model, device)
                        print(f"    [TRANSLATE] '{original_title}' -> '{translated_title}'")
                        titles_translated += 1
                        row_translated += 1
                        
                        # Update context with new translation
                        updated_ctx = ctx.copy()
                        updated_ctx['title'] = translated_title
                        updated_contexts.append(updated_ctx)
                    else:
                        print(f"    [SKIP] No original title found for context {i}")
                        updated_contexts.append(ctx)
                
                # Update the row with new translations
                df.at[index, 'contexts'] = serialize_contexts(updated_contexts)
                print(f"  Row summary: {row_translated} titles translated")
                
            except Exception as e:
                print(f"  Error processing row {row_id}: {e}")
                traceback.print_exc()
                continue
        
        # Save the updated dataframe
        print(f"\n{'=' * 60}")
        print(f"Translation Summary:")
        print(f"  Total rows processed: {rows_processed}")
        print(f"  Rows skipped (ID not found): {rows_skipped}")
        print(f"  Total titles translated: {titles_translated}")
        print(f"\nSaving updated data to {csv_path}...")
        df.to_csv(csv_path, index=False)
        print("Successfully saved updated data")
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    csv_file = 'hotpot/validation.csv'
    
    print("=" * 60)
    print("HotpotQA Title Retranslation Script")
    print("=" * 60)
    print("This script will translate ALL titles from the original HotpotQA dataset.")
    print("Existing titles will be replaced with fresh translations.")
    print()
    
    success = fix_title_translations(csv_file, backup=True)
    
    print()
    print("=" * 60)
    if success:
        print("✓ Title retranslation completed successfully!")
    else:
        print("✗ Title retranslation failed. Check error messages above.")
    print("=" * 60)
