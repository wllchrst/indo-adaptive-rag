import torch
import re
from transformers import pipeline, AutoTokenizer

MAX_TOKEN = 250 
tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-id")

def init_pipline():
    device_number = 0 if torch.cuda.is_available() else -1
    init_pipeline = pipeline("translation", model="Helsinki-NLP/opus-mt-en-id", max_length=2000, device=device_number)
    print(f"Init translation pipeline on device: {device_number}")

    return init_pipeline

pipe = init_pipline()

def split_long_text(text: str, max_length: int = MAX_TOKEN):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current = ""

    for s in sentences:
        tokenized = tokenizer(current + " " + s if current else s, return_tensors="pt", truncation=False)
        token_count = tokenized.input_ids.shape[1]

        if token_count <= max_length:
            current += " " + s if current else s
        else:
            if current:
                chunks.append(current.strip())
                current = ""
            # Check if individual sentence is too long
            sentence_tokens = tokenizer(s, return_tensors="pt", truncation=False).input_ids[0]
            for i in range(0, len(sentence_tokens), max_length):
                part = tokenizer.decode(sentence_tokens[i:i + max_length], skip_special_tokens=True)
                chunks.append(part.strip())

    if current:
        chunks.append(current.strip())

    return chunks

def split_without_token(text: str, max_length: int = MAX_TOKEN):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current = ""

    for s in sentences:
        # Handle very long individual sentences
        if len(s) > max_length:
            if current:
                chunks.append(current.strip())
                current = ""

            # Split long sentence into smaller parts
            for i in range(0, len(s), max_length):
                chunks.append(s[i:i + max_length].strip())
            continue

        if len(current) + len(s) + 1 <= max_length:
            current += " " + s if current else s
        else:
            chunks.append(current.strip())
            current = s

    if current:
        chunks.append(current.strip())

    return chunks

def translate_safe(text: str, by_token: bool = False) -> str:
    try:
        tokens = tokenizer(text, return_tensors="pt", truncation=False).input_ids.shape[1]
        if tokens <= MAX_TOKEN and by_token:
            return pipe(text)[0]['translation_text']
        elif len(text) < MAX_TOKEN and not by_token:
            return pipe(text)[0]['translation_text']
        else:
            print("[Splitting triggered] Token count:", tokens)
            chunks = split_long_text(text) if by_token else split_without_token(text)
            return " ".join(pipe(chunk)[0]['translation_text'] for chunk in chunks)
    except Exception as e:
        print(f"[Translation skipped] Error: {e} \nText: {text}\nToken Count: {tokens}")
        raise e

