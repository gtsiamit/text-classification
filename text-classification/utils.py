import pandas as pd


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
