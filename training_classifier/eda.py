import nltk
import nltk.corpus
import random
import re
import pandas as pd
from nltk import word_tokenize
from googletrans import Translator


# cleaning up text
def get_only_chars(line: str) -> str:
    """Clean the sentence from each line

    Args:
        line (str): string that is going to be processed

    Returns:
        str: return the processed string
    """

    clean_line = ""

    line = line.replace("’", "")
    line = line.replace("'", "")
    line = line.replace("-", " ")  # replace hyphens with spaces
    line = line.replace("\t", " ")
    line = line.replace("\n", " ")
    line = line.lower()

    for char in line:
        if char in 'qwertyuiopasdfghjklzxcvbnm ':
            clean_line += char
        else:
            clean_line += ' '

    clean_line = re.sub(' +', ' ', clean_line)  # delete extra spaces
    try:
        if clean_line[0] == ' ':
            clean_line = clean_line[1:]
    except IndexError as e:
        print(f'{line},{e}')
        raise e
    return clean_line


def load_synonyms(file_path):
    """Load synonyms from a .txt file into a dictionary."""
    synonym_dict = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            words = [word.strip() for word in line.split(",")]
            for word in words:
                synonym_dict[word] = [w for w in words if w != word]  # Exclude itself
    return synonym_dict


def load_stopwords(stopwords_path):
    """Load stopwords from a file into a set."""
    with open(stopwords_path, 'r', encoding='utf-8') as file:
        return set(word.strip() for word in file)


def get_synonyms(word: str) -> list[str]:
    """Get a list synonym from a word

    Args:
        word (str): word that will be target to finding the synonyms

    Returns:
        list[str]: return list of the word synonym
    """
    if word in synonyms_dict:
        return synonyms_dict[word]
    return []


stop_words = load_stopwords("important_files/id.stopwords.02.01.2016.txt")
synonyms_dict = load_synonyms("important_files/Tesaurus-BahasaIndonesia.txt")
translator = Translator()


def synonym_replacement(words: str, n: int) -> list[str]:
    """Synonym replacement for text augmenation

    Args:
        words (str): list of words that is going to be replaced by the synonym
        n (int): amount of changes

    Returns:
        list[str]: list of words has been replace with synonym
    """

    new_words = words.copy()
    random_word_list = list(set([word for word in words if word not in stop_words]))
    random.shuffle(random_word_list)
    num_replaced = 0
    for random_word in random_word_list:
        synonyms = get_synonyms(random_word)
        if len(synonyms) >= 1:
            synonym = random.choice(list(synonyms))
            new_words = [synonym if word == random_word else word for word in new_words]
            # print("replaced", random_word, "with", synonym)
            num_replaced += 1
        if num_replaced >= n:  # only replace up to n words
            break

    # this is stupid but we need it, trust me
    sentence = ' '.join(new_words)
    new_words = sentence.split(' ')

    return new_words


async def back_translation(sentence: str, src_lang='id', dest_lang='en') -> str:
    """Back Transalation

    Args:
        sentence (str): Sentence that is going to be augmented
        src_lang (str, optional): Orginal Language of the sentence. Defaults to 'id'.
        dest_lang (str, optional): Target Language. Defaults to 'en'.

    Returns:
        str: strin that been augmented
    """
    english_version = await translator.translate(sentence, src=src_lang, dest=dest_lang)
    back_translated = await translator.translate(english_version.text, src=dest_lang, dest=src_lang)
    return back_translated.text


async def indonesia_eda(sentence: str,
                        with_bt: bool = False) -> list[str]:
    """Augments an Indonesian sentence into three variations using Synonym Replacement and Back Translation."""

    print(sentence)
    try:
        sentence = get_only_chars(sentence)
        words = sentence.split(' ')
        words = [word for word in words if word != '']

        augmented_sentences = []

        a_words = synonym_replacement(words, 2)

        # Wrap async calls with retry mechanism
        async def safe_translate(text, lang="en"):
            try:
                return await back_translation(text, dest_lang=lang)
            except Exception as e:
                print(f"Translation failed ({lang}): {e}")
                return text  # Fallback to original text

        if with_bt:
            bt_argentina = await safe_translate(sentence, "ar")
            bt_english = await safe_translate(sentence)
            augmented_sentences.append(bt_argentina)
            augmented_sentences.append(bt_english)

        augmented_sentences.append(" ".join(a_words))

        return list(set(augmented_sentences))

    except Exception as e:
        print(f'Error Indonesia EDA: {e}')
        return []


def remove_indonesian_stopwords(words: list[str]) -> list[str]:
    """Removes Indonesian stopwords from a list of words."""
    return [word for word in words if word.lower() not in stop_words]


def keep_alphabet(words: list[str]) -> list[str]:
    """Keeps only words containing alphabetic characters."""
    return [word for word in words if word.isalpha()]


def split_words(sentence: str) -> list[str]:
    """Tokenizes a sentence into words."""
    return word_tokenize(sentence)
