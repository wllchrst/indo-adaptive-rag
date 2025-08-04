import re
import string
import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt_tab')

def read_words_from_file(filepath: str):
        with open(filepath, 'r', encoding='utf-8') as f:
            words = f.read().splitlines()
        return words

stop_words = read_words_from_file('stop_words.txt')
class WordHelper:
    @staticmethod
    def remove_non_alphabetic(text: str) -> str:
        return re.sub(r'[^A-Za-z\s]+', '', text)

    @staticmethod
    def remove_stop_words(text: str) -> str:
        words = word_tokenize(text)
        words = [word for word in words if word.lower() not in stop_words]

        return ' '.join(words)
    
    @staticmethod
    def clean_sentence(text: str) -> str:
        # Remove markdown bold (**text**)
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
        # Remove newline escape characters (\n, \r)
        text = text.replace('\\n', '').replace('\\r', '')
        return text.strip()
    
    @staticmethod
    def get_tokens(s):
        if not s:
            return []
        return WordHelper.normalize_text(s).split()
    
    @staticmethod
    def normalize_text(text: str):
        def remove_articles(text):
            regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
            return re.sub(regex, " ", text)

        def white_space_fix(text):
            return " ".join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(text))))
    
    @staticmethod
    def contains(text: str, target: str):
        """Check if text is contained in target string.

        Args:
            text (str): text to check for presence in target
            target (str): target string to check against
        """        
        text = WordHelper.normalize_text(text)
        target = WordHelper.normalize_text(target)
        return text in target