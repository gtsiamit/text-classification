import argparse
import pandas as pd
from utils import load_data, store_df, store_json
from transformers import AutoTokenizer
import numpy as np
from transformers import AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score
from datasets import Dataset
from typing import Tuple, Dict
from data import split_data, encode_labels
from modeling import (
    set_device,
    define_tokenizer,
    tokenize_data,
    define_model,
    format_dataset,
    define_data_collator,
)


def define_train_args(
    epochs: float = 3, learning_rate: float = 5e-5
) -> TrainingArguments:
    """
    Defines the training arguments for the Trainer.

    Args:
        epochs (int, optional): The number of epochs. Defaults to 3.
        learning_rate (_type_, optional): The learning rate. Defaults to 5e-5.

    Returns:
        TrainingArguments: The training arguments.
    """

    training_args = TrainingArguments(
        output_dir="./output_finetune/results",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=epochs,
        weight_decay=0.01,
        save_total_limit=2,
        load_best_model_at_end=True,
        logging_dir="./output_finetune/logs",
        logging_strategy="epoch",
        seed=2025,
    )

    return training_args


def compute_metrics(eval_pred: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
    """
    Computes evaluation metrics for model predictions.

    Args:
        eval_pred (Tuple[np.ndarray, np.ndarray]):
            A tuple containing:
                - logits (np.ndarray): Raw model outputs of shape (num_samples, num_classes).
                - labels (np.ndarray): Ground truth labels of shape (num_samples,).

    Returns:
        Dict[str, float]: A dictionary containing:
            - "accuracy": Classification accuracy.
            - "weighted_f1": Weighted-average F1 score.
            - "macro_f1": Macro-average F1 score.
    """
    logits, labels = eval_pred
    predictions = logits.argmax(axis=1)

    return {
        "accuracy": accuracy_score(labels, predictions),
        "weighted_f1": f1_score(labels, predictions, average="weighted"),
        "macro_f1": f1_score(labels, predictions, average="macro"),
    }


def define_trainer(
    model: AutoModelForSequenceClassification,
    training_args: TrainingArguments,
    train_dataset: Dataset,
    val_dataset: Dataset,
    tokenizer: AutoTokenizer,
) -> Trainer:
    """
    Creates a HuggingFace Trainer for model training and evaluation.

    Args:
        model (AutoModelForSequenceClassification): The model to be trained.
        training_args (TrainingArguments): Training arguments for the Trainer.
        train_dataset (Dataset): The training dataset.
        val_dataset (Dataset): The validation dataset.
        tokenizer (AutoTokenizer): The BERT tokenizer.

    Returns:
        Trainer: Configured HuggingFace Trainer instance.
    """

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=define_data_collator(tokenizer),
        compute_metrics=compute_metrics,
    )

    return trainer


def setup_and_perform_training():

    # Set device to perform training
    set_device()

    # Load dataset and split X and y
    data = load_data(DATASET_PATH)
    X, y = data.data.to_numpy(), data.labels.to_numpy()
    print("Data loaded")

    # Split data into train, val and test sets
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X=X, y=y)
    print(f"X train shape: {X_train.shape}, y train shape: {y_train.shape}")
    print(f"X val shape: {X_val.shape}, y val shape: {y_val.shape}")
    print(f"X test shape: {X_test.shape}, y test shape: {y_test.shape}")

    # Define tokenizer
    tokenizer = define_tokenizer()

    # Tokenize X data
    X_train = tokenize_data(X_train, tokenizer=tokenizer)
    X_val = tokenize_data(X_val, tokenizer=tokenizer)
    X_test = tokenize_data(X_test, tokenizer=tokenizer)
    print("X data tokenized")

    # Apply label encoding to y data
    y_train, le = encode_labels(y=y_train)
    num_classes = len(le.classes_)
    y_val = encode_labels(y=y_val, encoder=le)
    y_test = encode_labels(y=y_test, encoder=le)
    print("y data label encoded")

    # Format train, val, test sets in order to be used by HuggingFace Trainer
    train_dataset = format_dataset(X_train, y_train)
    val_dataset = format_dataset(X_val, y_val)
    test_dataset = format_dataset(X_test, y_test)
    print("Train, val and test datasets formatted")

    # Define model
    model = define_model(num_classes)

    # Define the training arguments
    training_args = define_train_args()

    # Setup the Trainer
    trainer = define_trainer(
        model=model,
        training_args=training_args,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        tokenizer=tokenizer,
    )

    # Trigger training
    trainer.train()

    # Evaluate the model on the validation set
    eval_results = trainer.evaluate()

    # Generate predictions on the test set and keep also the true labels
    predictions = trainer.predict(test_dataset)
    predictions = predictions.predictions.argmax(axis=-1)
    test_actual_labels = test_dataset["labels"].tolist()

    # Create a DataFrame to store the true and predicted labels
    df_test_results = pd.DataFrame(test_actual_labels, columns=["true"])
    df_test_results["predicted"] = predictions

    # Storing model and tokenizer
    trainer.save_model("./output_finetune/model_ft")
    tokenizer.save_pretrained("./output_finetune/tokenizer_ft")
    print("Model and tokenizer stored")

    # Storing evaluation results and preditions on test set
    store_df(df=df_test_results, filepath="./output_finetune/test_preds.csv")
    store_json(filepath="./output_finetune/validation_results.json", data=eval_results)
    print("Evaluation results and predictions on test set stored")


def main():

    # input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_path", type=str, required=True, help="Path to dataset"
    )
    args = parser.parse_args()

    # set dataset path
    global DATASET_PATH
    DATASET_PATH = args.dataset_path

    # setup and run train process
    setup_and_perform_training()


if __name__ == "__main__":
    main()
