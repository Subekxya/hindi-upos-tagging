# Hindi UPOS Tagging with Multilingual DistilBERT

A research-oriented NLP project exploring Hindi Part-of-Speech (POS) tagging using a pretrained multilingual Transformer. This project investigates how fine-tuning strategies affect token-classification performance when training data is limited.

## Project Overview

**Research question:** How does partially freezing a pretrained multilingual Transformer affect Hindi UPOS tagging performance compared with full fine-tuning?

The project uses the Hindi Universal Dependencies Hindi-HDTB treebank and the pretrained `distilbert/distilbert-base-multilingual-cased` model.

Two training strategies are compared:

- **Baseline:** Fine-tune the Transformer model and its token-classification head.
- **Partial freezing:** Freeze the first three of the model's six Transformer layers while training the remaining layers and classification head.

Both models are trained for two epochs and evaluated on the same validation and held-out test sets.

## Objectives

- Explore and prepare a Hindi linguistic dataset in CoNLL-U format.
- Understand the UPOS tag inventory and token-level labeling task.
- Implement subword tokenization and align word-level labels with Transformer tokens.
- Fine-tune a multilingual pretrained language model for sequence labeling.
- Compare full fine-tuning with partial layer freezing.
- Evaluate models using accuracy, precision, recall, and F1-score.
- Save evaluation metrics for reproducibility and comparison.

## Dataset

The project uses **Hindi UD-HDTB**, a Hindi treebank in the Universal Dependencies framework.

Dataset repository: [Universal Dependencies — Hindi-HDTB](https://github.com/UniversalDependencies/UD_Hindi-HDTB)

The experiments use the following splits:

| Split | Number of sentences | Purpose |
|---|---:|---|
| Training subset | 2,500 | Model training |
| Development | 1,659 | Validation during training |
| Test | 1,684 | Final held-out evaluation |

The training subset is stored in `data/processed/hi_hdtb_train_2500_seed42.json`. It contains 2,500 sentences sampled using seed 42.

The development and test datasets are loaded from CoNLL-U files.

### UPOS Labels

The task predicts one of 16 Universal Part-of-Speech tags:

`ADJ`, `ADP`, `ADV`, `AUX`, `CCONJ`, `DET`, `INTJ`, `NOUN`, `NUM`, `PART`, `PRON`, `PROPN`, `PUNCT`, `SCONJ`, `VERB`, and `X`.

## Methodology

### 1. Data exploration

The first notebook, `01_hindi_hdtb_data_exploration.ipynb`, explores the Hindi treebank and prepares the training subset used in the experiments.

### 2. Tokenization and label alignment

The second notebook, `02_baseline_finetuning.ipynb`, uses the fast tokenizer from the multilingual DistilBERT checkpoint.

Because the tokenizer can split a Hindi word into multiple subword tokens, word-level UPOS labels must be aligned with the resulting token sequence.

The implementation assigns each word's UPOS label to its first subtoken. Special tokens, such as `[CLS]` and `[SEP]`, and continuation subtokens receive the label `-100`, which excludes these positions from the token-classification loss.

### 3. Baseline fine-tuning

The pretrained model is loaded with a token-classification head containing 16 output labels. The model is fine-tuned for two epochs using a training batch size of 8 and a validation batch size of 8.

### 4. Partial layer freezing

A second model is initialized from the same pretrained checkpoint. The first three of its six Transformer layers are frozen, while the remaining layers and classification head remain trainable.

In this experiment, 84.22% of the model parameters remain trainable after freezing the first three layers.

### 5. Evaluation

Both models are evaluated on the development and held-out test sets using:

- Accuracy
- Precision
- Recall
- F1-score

The evaluation metrics are saved as JSON files in the `outputs/` directory.

## Results

### Held-out test set

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Baseline: full fine-tuning | **94.67%** | **88.38%** | **88.02%** | **88.19%** |
| Partial freezing | 93.20% | 86.89% | 85.63% | 86.18% |

### Validation set

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Baseline: full fine-tuning | **94.65%** | **88.08%** | **87.56%** | **87.80%** |
| Partial freezing | 93.11% | 86.38% | 84.87% | 85.51% |

### Interpretation

The baseline model achieved a test F1-score of **88.19%**, approximately 2.01 percentage points higher than the partially frozen model's 86.18%.

The baseline also achieved higher accuracy, precision, and recall. Under this experimental setup, allowing all Transformer layers to adapt to the Hindi UPOS tagging task produced better results than freezing the first half of the encoder.

This is a preliminary comparison using a limited training subset and two training epochs. It does not establish that full fine-tuning will outperform partial freezing in every setting. Further experiments with different freezing strategies, training sizes, and random seeds would be needed to draw broader conclusions.

## Technologies

- **Python 3.11**
- **PyTorch** — model training and tensor computation
- **Hugging Face Transformers** — pretrained model, tokenizer, and Trainer API
- **Hugging Face Datasets** — dataset preparation
- **NumPy and Pandas** — numerical computation and data exploration
- **Matplotlib and Seaborn** — data visualization
- **Jupyter Notebook** — experimentation and documentation
- **Apple Metal Performance Shaders (MPS)** — hardware acceleration on Apple Silicon

## Project Structure

The following structure reflects the paths referenced by the notebooks. Adjust it to match the actual files in your repository.

```text
MLP/
├── data/
│   ├── hi_hdtb-ud-train.conllu
│   ├── hi_hdtb-ud-dev.conllu
│   ├── hi_hdtb-ud-test.conllu
│   └── processed/
│       └── hi_hdtb_train_2500_seed42.json
├── notebooks/
│   ├── 01_hindi_hdtb_data_exploration.ipynb
│   └── 02_baseline_finetuning.ipynb
├── outputs/
│   ├── baseline_metrics.json
│   ├── partial_freezing_metrics.json
│   └── test_metrics.json
├── src/
│   ├── __init__.py
│   ├── conllu_utils.py
│   └── evaluation.py
├── .gitignore
├── README.md
└── requirements.txt
```

The development and test CoNLL-U files must also be available at the paths expected by the notebook. The raw dataset and generated outputs may be omitted from version control when appropriate.

## Installation

Clone your repository and enter its directory:

```bash
git clone <your-repository-url>
cd hindi-upos-tagging
```

Create and activate a Python virtual environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

On Windows, activate the environment with:

```powershell
.venv\Scripts\activate
```

Install the dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For Apple Silicon, install a PyTorch version compatible with MPS. For other platforms, follow the official [PyTorch installation guide](https://pytorch.org/get-started/locally/).

Launch JupyterLab:

```bash
jupyter lab
```

Run the notebooks in this order:

1. `01_hindi_hdtb_data_exploration.ipynb`
2. `02_baseline_finetuning.ipynb`

The fine-tuning notebook expects the processed training subset, development and test CoNLL-U files, and the helper modules under `src/` to be present at the configured paths.

## Limitations and Future Work

The current experiment has several limitations:

- Training uses only 2,500 sentences rather than the entire training treebank.
- Each configuration is trained for only two epochs.
- Only one partial-freezing strategy is evaluated.
- The experiments do not include repeated runs across multiple random seeds.
- Aggregate metrics do not show which UPOS categories are most difficult to predict.
- The results do not establish generalization to other Hindi domains or other languages.

Potential next steps include:

1. Train on larger portions of the treebank.
2. Compare freezing different numbers of Transformer layers.
3. Run multiple seeds and report variability.
4. Evaluate per-tag precision, recall, and F1-score.
5. Inspect confusion matrices and common tagging errors.
6. Compare against a simple baseline such as a majority-tag predictor.
7. Extend the study to cross-lingual transfer between Hindi and Nepali.

## References

- **Dataset:** [Universal Dependencies — Hindi-HDTB](https://github.com/UniversalDependencies/UD_Hindi-HDTB)
- **Model:** [DistilBERT multilingual cased](https://huggingface.co/distilbert/distilbert-base-multilingual-cased)
- **Transformers:** [Hugging Face documentation](https://huggingface.co/docs/transformers)
- **Datasets:** [Hugging Face Datasets documentation](https://huggingface.co/docs/datasets)
- **PyTorch:** [Official documentation](https://pytorch.org/docs/stable/index.html)

---

*This project is an ongoing exploration of multilingual NLP, Transformer fine-tuning, and experimental evaluation for Hindi sequence labeling.*
