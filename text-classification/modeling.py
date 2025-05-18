import torch
from transformers import AutoTokenizer
import numpy as np
from transformers import AutoModelForSequenceClassification
from transformers import DataCollatorWithPadding
from datasets import Dataset
import transformers


def set_device():
    """
    Set the device to GPU if available, otherwise use CPU or MPS.
    """

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print("Device:", device)


def define_tokenizer() -> AutoTokenizer:
    """
    Defines the BERT tokenizer.

    Returns:
        AutoTokenizer: The BERT tokenizer.
    """

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    return tokenizer


def tokenize_data(
    data: np.ndarray, tokenizer: AutoTokenizer
) -> transformers.tokenization_utils_base.BatchEncoding:
    """
    Tokenizes the input data using the BERT tokenizer.

    Args:
        data (np.ndarray): The data to be tokenized.
        tokenizer (AutoTokenizer): The BERT tokenizer.

    Returns:
        transformers.tokenization_utils_base.BatchEncoding: The tokenized data.
    """

    tokenized_data = tokenizer(
        data.tolist(), padding="max_length", truncation=True, max_length=256
    )

    return tokenized_data


def define_model(num_labels: int) -> AutoModelForSequenceClassification:
    """
    Defines the BERT model for sequence classification.

    Args:
        num_labels (int): The number of labels for classification.

    Returns:
        AutoModelForSequenceClassification: The BERT model.
    """

    # Load the BERT model for sequence classification
    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=num_labels
    )

    # Freeze all layers except the classifier and the last layer
    for name, param in model.bert.named_parameters():
        if name.startswith("encoder.layer.11"):
            param.requires_grad = True
        else:
            param.requires_grad = False

    # Keep the classification head trainable
    for param in model.classifier.parameters():
        param.requires_grad = True

    return model


def define_data_collator(tokenizer: AutoTokenizer) -> DataCollatorWithPadding:
    """
    Defines the data collator for padding the input data.

    Args:
        tokenizer (AutoTokenizer): The BERT tokenizer.

    Returns:
        DataCollatorWithPadding: The data collator for padding.
    """

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    return data_collator


def format_dataset(
    tokenized_x: transformers.tokenization_utils_base.BatchEncoding, y: np.ndarray
) -> Dataset:
    """
    Formats the tokenized data and labels into a Dataset object.

    Args:
        tokenized_x (transformers.tokenization_utils_base.BatchEncoding): The tokenized input x data.
        y (np.ndarray): The y data, labels.

    Returns:
        Dataset: The formatted dataset.
    """

    data_dict = {
        "input_ids": tokenized_x.get("input_ids"),
        "attention_mask": tokenized_x.get("attention_mask"),
        "labels": y.tolist(),
    }

    dataset = Dataset.from_dict(data_dict)
    dataset.set_format(type="torch")

    return dataset
