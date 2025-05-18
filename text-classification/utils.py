import pandas as pd
import json


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
