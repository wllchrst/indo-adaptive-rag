import re
from nltk.tokenize import word_tokenize

def read_words_from_file(self, filepath: str):
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