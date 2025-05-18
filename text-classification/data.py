from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
from typing import Tuple


def split_data(
    X: np.ndarray, y: np.ndarray, test_size: float = 0.2, val_size: float = 0.1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Splits the data into train, validation and test sets.

    Args:
        X (np.ndarray): The X data to be split.
        y (np.ndarray): The y data to be split.
        test_size (float, optional): The test size. Defaults to 0.2.
        val_size (float, optional): The validation size. Defaults to 0.1.

    Returns:
        Tuple:
            - X_train (np.ndarray): Training feature set.
            - X_val (np.ndarray): Validation feature set.
            - X_test (np.ndarray): Test feature set.
            - y_train (np.ndarray): Training labels.
            - y_val (np.ndarray): Validation labels.
            - y_test (np.ndarray): Test labels.
    """

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=2025, stratify=y
    )

    # Set the validation size according to the training set size in order to maintain the same ratio of the full dataset
    val_size_adjusted_to_train = val_size / (1 - test_size)

    # Split the training set into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=val_size_adjusted_to_train,
        random_state=2025,
        stratify=y_train,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def encode_labels(
    y: np.ndarray, encoder: LabelEncoder = None
) -> Tuple[np.ndarray, LabelEncoder]:
    """
    Encodes target labels using a scikit-learn LabelEncoder.

    If no encoder is provided, a new one is fitted on the input labels. If an encoder is given,
    it is used to transform the labels directly.

    Args:
        y (np.ndarray): Array of raw labels to encode.
        encoder (LabelEncoder, optional): A pre-fitted LabelEncoder to use. If None, a new one is created and fitted.

    Returns:
        Tuple[np.ndarray, LabelEncoder]:
            - y_encoded (np.ndarray): The label-encoded array.
            - encoder (LabelEncoder): The fitted LabelEncoder instance."""

    # Check if the encoder is provided
    if not encoder:
        le = LabelEncoder()
        le.fit(y)
    else:
        le = encoder

    # Transform the labels using the fitted encoder
    y = le.transform(y)

    # If no encoder was provided, return the encoder along with the transformed labels
    # If an encoder was provided, return only the transformed labels
    if not encoder:
        return y, le
    else:
        return y
