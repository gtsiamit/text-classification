import pandas as pd
import json
from typing import Any
import pickle


def load_data(file_path):
    """
    Load data from a CSV file and return a DataFrame.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: DataFrame containing the loaded data.
    """
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def filter_df_by_label(df, label):
    """
    Filter the DataFrame by a specific label.

    Args:
        df (pd.DataFrame): DataFrame to filter.
        label (str): Label to filter by.

    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    if df is not None:
        return df[df["labels"] == label]
    else:
        print("DataFrame is None.")
        return None


def store_df(df: pd.DataFrame, filepath: str) -> None:
    """
    Save a pandas DataFrame to a csv file.

    Args:
        df (pd.DataFrame): The DataFrame to save.
        filepath (str): The path to save the file.
    """

    df.to_csv(filepath, sep=",", index=False, header=True)


def store_json(filepath: str, data: dict) -> None:
    """
    Save a dictionary to a JSON file.

    Args:
        filepath (str): The path to save the file.
        data (dict): The dictionary to save.
    """

    with open(filepath, "w") as f:
        json.dump(data, f)


def store_pickle(filepath: str, data: Any) -> None:
    """
    Stores a Python object to a file using pickle serialization.

    Args:
        filepath (str): Path to the file where the data will be stored.
        data (Any): The Python object to serialize and store.
    """

    with open(filepath, "wb") as f:
        pickle.dump(data, f)


def load_pickle(filepath: str) -> Any:
    """
    Loads a Python object from a pickle file.

    Args:
        filepath (str): Path to the pickle file to load.

    Returns:
        Any: The deserialized Python object.
    """
    with open(filepath, "rb") as f:
        return pickle.load(f)


def load_json(filepath: str) -> Any:
    """
    Loads and returns the contents of a JSON file.

    Args:
        filepath (str): The path to the JSON file.

    Returns:
        Any: The parsed contents of the JSON file as a Python object
             (usually a dict or list).
    """
    with open(filepath, "r") as f:
        return json.load(f)
