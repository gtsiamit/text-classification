# text-classification
Text classification with BERT

# Introduction
In this project a BERT model is fine-tuned for a text classification task using the Hugging Face Transformers library.

The dataset used is the BBC Full Text Document Classification, which can be found [here](https://www.kaggle.com/datasets/alfathterry/bbc-full-text-document-classification). This dataset contains full-length BBC news articles, with each article assigned to one of five distinct categories: business, entertainment, politics, sport, or tech.

# Repo structure
```
text-classification
├─ .gitignore
├─ README.md
└─ text-classification
   ├─ eda.ipynb    # The exploratory data analysis Jupyter Python notebook
   ├─ helpers
   │  ├─ data.py    # Contains functions related to data processing
   │  ├─ modeling.py    # Contains functions for modeling
   │  ├─ text.py    # Contains functions for text processing
   │  ├─ utils.py    # Includes utility functions for general purpose use
   │  └─ visualization.py    # Contains functions for visualizations and plots
   ├─ requirements.txt    # The Python packages dependencies for installing the project
   ├─ results.ipynb    # The results Jupyter Python notebook
   └─ train.py    # The main scipt that handles fine-tuning of BERT model
```

# Installation
In a virtual environment with `Python 3.13`, the required dependencies can be installed with the following commands:
```bash
cd text-classification
pip install -r requirements.txt
```

# Exploratory Data Analysis
The exploratory data analysis (EDA) conducted prior to fine-tuning BERT, used to better understand the dataset, is available in the [eda.ipynb](text-classification/eda.ipynb) Jupyter notebook.

# Execution
The BERT fine-tuning process can be executed with the following commands:
```bash
cd text-classification
python3 train.py --dataset_path /path/to/data/dataset.csv
```

The `--dataset_path` argument should be set to the exact path of the CSV dataset.

After executing `train.py`, the following outputs are saved in the `text-classification/output_finetune` directory:
- Trainer results and logs exported by the Hugging Face Trainer API.
- Fine-tuned BERT model, stored in the `model_ft` folder.
- Tokenizer used during the fine-tuning process, saved in the `tokenizer_ft` folder.
- LabelEncoder used to encode the target classes, saved as `le.pkl`.
- Validation metrics, stored in `validation_results.json`.
- Predictions on the test set, saved in `test_preds.csv`.

# Results extraction
The evaluation results on the test set (i.e., unseen data) can be explored in the [results.ipynb](text-classification/results.ipynb) Jupyter notebook.