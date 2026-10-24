# Group 12 — Probabilistic Latent Semantic Analysis for Information Retrieval

An experimental information retrieval project exploring **Probabilistic Latent Semantic Analysis (PLSA)** and comparing topic-based document retrieval against **TF-IDF / Vector Space Model (VSM)** and **Latent Semantic Indexing (LSI)** on the **20 Newsgroups** dataset.

> **Project status:** The uploaded `Group_12_PLSA_Baseline.ipynb` is the confirmed implementation described below. Additional experiments, such as merging 20 categories into six supercategories, query-likelihood ranking, and validation-based model selection, should be documented as implemented only after their notebooks are added and successfully verified. The baseline notebook itself does not implement those additions.

## 1. Objectives

- Implement PLSA from first principles using the Expectation–Maximization (EM) algorithm.
- Explore latent topic distributions and document-topic mixtures.
- Study whether topic representations support information retrieval beyond direct keyword matching.
- Compare PLSA retrieval with TF-IDF/VSM and LSI under a common evaluation setup.
- Investigate sensitivity to the number of latent topics and EM initialization.
- Analyze controlled mixtures of documents from different subject categories.

## 2. Dataset

The baseline uses the **20 Newsgroups** dataset through `sklearn.datasets.fetch_20newsgroups(subset="all")`.

- **Corpus:** all 20 original newsgroup categories, rather than a six-category subset.
- **Text cleanup:** dataset headers, footers and quoted passages are removed during loading.
- **Preprocessing:** spaCy-based normalization/lemmatization and stop-word filtering.
- **Features:** a training-fitted `CountVectorizer` with up to 3,000 vocabulary terms.
- **Data split:** stratified training/test split; vocabulary and retrieval models are fitted on training documents only.

Ground-truth newsgroup labels are used for stratification, relevance proxies and the controlled mixture experiment. **PLSA itself is trained without class labels.**

## 3. Methods

### PLSA

PLSA models the probability of a word occurring in a document as a mixture of latent topics:

\[
P(w\mid d)=\sum_{z=1}^{K} P(w\mid z)P(z\mid d)
\]

where `K` is the number of latent topics, `P(w|z)` is the topic–word distribution, and `P(z|d)` is the document–topic distribution.

The notebook implements the EM algorithm using NumPy and sparse document-term counts. It supports multiple random initializations, tracks training log-likelihood, inspects high-probability topic words and folds unseen documents/queries into the trained topic space.

### Retrieval baselines

- **TF-IDF/VSM:** lexical term-weighting and cosine-similarity retrieval.
- **LSI:** truncated SVD of the TF-IDF representation, followed by similarity-based retrieval.
- **PLSA baseline:** topic-space representation and similarity-based retrieval.

The baseline evaluates the same held-out documents and category-derived query relevance labels for each method.

### Controlled topic-mixture experiment

The notebook constructs synthetic documents by mixing words from baseball and politics-related source documents. It examines how the inferred PLSA topic mixture changes with the known source-word ratio. This is a diagnostic of learned topic behavior, **not** a guarantee of exact mixture-proportion recovery.

### Model-order and tempered-EM exploration

The baseline studies topic counts `K = 2, 3, 5, 8, 10` and includes an exploratory tempered E-step experiment using several temperature (`beta`) settings. These analyses are exploratory and should not be interpreted as validated retrieval hyperparameter selection.

## 4. Evaluation

The uploaded baseline reports the following retrieval metrics:

- **Precision@k:** fraction of top-k retrieved documents judged relevant.
- **Recall@k:** fraction of relevant test documents appearing in the top-k.
- **F1@k:** harmonic mean of Precision@k and Recall@k.

Relevance is approximated using each document's original newsgroup category. These labels are useful for a controlled comparison but are **not human relevance judgments** for real search queries.

**Results:** Run the notebook and record the actual outputs before adding numerical scores here. Do not claim that PLSA outperforms TF-IDF or LSI unless the measured results support that conclusion.

## 5. Repository structure

The confirmed uploaded notebook is:

```text
Group_12_PLSA_Baseline.ipynb
```

If the team stores notebooks under `notebooks/`, adjust the path accordingly. Additional experiment notebooks and shared modules may be added as the team merges and verifies them.

Suggested organization as the project develops:

```text
.
├── README.md
├── notebooks/
│   └── Group_12_PLSA_Baseline.ipynb
├── src/                  # Optional reusable Python modules
│   ├── plsa_model.py
│   ├── preprocessing.py
│   └── retrieval.py
└── results/              # Verified experiment summaries/figures
```

The `src/` and `results/` directories above are **proposed organization**, not a claim that those files are already present in the repository.

## 6. Run in Google Colab

1. Open [Google Colab](https://colab.research.google.com/).
2. Upload `Group_12_PLSA_Baseline.ipynb` or open it from GitHub.
3. Install dependencies in a Colab code cell if needed:

   ```python
   %pip install -q numpy scipy scikit-learn pandas matplotlib spacy
   !python -m spacy download en_core_web_sm
   ```

4. Run the notebook cells in order (**Runtime → Run all**).
5. Inspect the topic-word distributions, likelihood plots, mixture experiment, retrieval comparisons and model-order study.

> **Runtime note:** Multiple PLSA fits on the full corpus can take significant time and memory. The notebook has not been rerun as part of preparing this README.

### Run locally

Install Python 3 and Jupyter, then install the same dependencies and spaCy model:

```bash
python -m pip install numpy scipy scikit-learn pandas matplotlib spacy notebook
python -m spacy download en_core_web_sm
jupyter notebook
```

Open the `.ipynb` file in the Jupyter interface and run its cells in order.

## 7. Planned / separate experiments

These are separate from the **confirmed uploaded baseline** and should be marked complete only after the corresponding notebooks are committed and executed:

- **20-to-6 label aggregation:** retain all 20 source categories while mapping their labels into six broader supercategories. This is a *manual label grouping*, not PLSA discovering six classes.
- **Query-likelihood retrieval:** rank documents by the probability of query terms under the fitted PLSA topic mixture.
- **Validation-based selection:** choose topic count and retrieval scoring on a validation set, then report results on a held-out test set.
- **Expanded evaluation:** MAP@10, MRR@10, query-family breakdowns, and runtime/memory profiling.
- **Shared implementation:** extract reusable preprocessing, PLSA training and retrieval functions into `src/` so experiments use a consistent implementation.

For fair comparisons, all methods within an experiment should use the same corpus split, candidate documents, queries and relevance definition. Scores from different splits or different label mappings should not be compared as though they came from the same benchmark.

## 8. Team Git workflow

Work on separate branches and merge through pull requests:

```bash
git clone https://github.com/OWNER/REPOSITORY.git
cd REPOSITORY
git switch main
git pull origin main
git switch -c feature/my-experiment
# Edit your assigned files
git add .
git commit -m "Describe the experiment or fix"
git push -u origin feature/my-experiment
```

Open a pull request on GitHub, review the changes and merge into `main` after verification. Replace `OWNER/REPOSITORY` with the actual repository address.

**Notebook collaboration tip:** Avoid having two people edit the same `.ipynb` simultaneously; Jupyter notebooks are JSON files and can generate difficult merge conflicts. Keep major experiments in separate notebooks or shared `.py` modules.

## 9. Limitations

- PLSA maximizes data likelihood; it does not directly optimize retrieval metrics.
- Learned topic IDs do not necessarily correspond to human newsgroup labels.
- Topic count, initialization, preprocessing and query wording affect performance.
- TF-IDF can be stronger for exact lexical matches; LSI can also be competitive.
- Category membership is only a proxy for document relevance.
- The baseline does not use a separate validation split to select retrieval hyperparameters.
- Improvements should be demonstrated through held-out measurements rather than assumed.

## 10. Reference

Thomas Hofmann, **“Probabilistic Latent Semantic Analysis,”** Proceedings of the Fifteenth Conference on Uncertainty in Artificial Intelligence (UAI), 1999.

---

**Course project:** Group 12 — Information Retrieval  
**Maintainers:** Group 12 team members (add names and roles after confirming the final assignment).
