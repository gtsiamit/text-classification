import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from string import punctuation
from wordcloud import WordCloud
from collections import Counter


def get_stopwords() -> list:
    """
    Download and return the list of English stopwords from NLTK.
    Returns:
        list: A list of stopwords.
    """
    nltk.download("stopwords")
    return list(stopwords.words("english"))


def get_punctuation() -> list:
    """
    Return a list of punctuation characters.
    Returns:
        list: A list of punctuation characters.
    """

    return list(punctuation)


def get_tokens(text: str) -> list:
    """
    Tokenize the input text into words using NLTK.

    Args:
        text (str): The input text to be tokenized.

    Returns:
        list: A list of tokens (words) extracted from the input text.
    """

    tokenizer = RegexpTokenizer(r"\w+")
    tokens = tokenizer.tokenize(text)

    return tokens


def get_top_n_words(input_: pd.Series, n: int = 20) -> pd.DataFrame:
    """
    Get the top n most frequent words from a pandas Series after removing stop words and punctuation.

    Args:
        input_ (pd.Series): The input data, which is a pandas Series.
        n (int, optional): The number of top words to return. Defaults to 20.

    Returns:
        pd.DataFrame: A DataFrame containing the top n words and their counts.
    """

    # Get stop words
    stop_words = get_stopwords()

    # Merge all input into a single string and convert to lowercase.
    words_in_data = " ".join(input_).lower()

    # Get tokens from the data. This will also remove punctuation.
    tokens_in_data = get_tokens(words_in_data)

    # Remove stop words from the data
    filtered_data = list(filter(lambda x: x not in stop_words, tokens_in_data))

    # Count the frequency of each word
    top_n_words = Counter(filtered_data).most_common(n)
    top_n_words = pd.DataFrame(top_n_words, columns=["word", "count"])

    return top_n_words


def get_word_cloud(input_: pd.Series) -> WordCloud:
    """
    Generate a word cloud from the input data.

    Args:
        input_ (pd.Series): The input data, which is a pandas Series.

    Returns:
        WordCloud: A WordCloud object representing the generated word cloud.
    """

    # Get stop words
    stop_words = get_stopwords()

    # Merge all input into a single string and convert to lowercase.
    joined_data = " ".join(input_).lower()

    # Get tokens from the data. This will also remove punctuation.
    # Concatenate all the tokens into a single string.
    tokens = get_tokens(text=joined_data)
    joined_data = " ".join(tokens)

    # Generate the word cloud.
    wordcloud = WordCloud(stopwords=stop_words, width=1000, height=500).generate(
        joined_data
    )

    return wordcloud
