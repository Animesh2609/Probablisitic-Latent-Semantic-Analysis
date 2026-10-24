# PLSA Experiments: Six-Supergroup Retrieval and Bagging

**Prepared for transfer to the next team member.** This handoff records the completed experiments, the reasons for their design decisions, and the results in the notebooks. The notebooks have been rerun, and the saved results and notebook explanations have been reviewed.

## Project question and scope

The project compares PLSA retrieval with TF-IDF/VSM and LSI on the 20 Newsgroups corpus. The objective is to identify the conditions under which PLSA performs well and where it does not. We did not set out to prove that PLSA is universally better.

PLSA learns from document-word co-occurrence and does not receive category labels during fitting. Newsgroups labels are used here as proxy relevance judgments for retrieval evaluation. Scores therefore depend on the grouping, query, split, candidate set, and metric. Results below belong to the six-supergroup, six-query, 80/20 protocol and must not be compared as if they shared a benchmark with studies using different labels or query sets.

## Why we tested six supergroups

The initial concern was that the original 20 Newsgroups labels may be a poor clustering of the documents: closely related subjects are split among different labels, while several related categories may be better treated as one retrieval relevance family. We therefore defined a human-readable mapping from all 20 original labels into six supergroups and evaluated retrieval against those broader groups.

This changes the evaluation relevance definition; it does not supervise PLSA or guarantee one latent topic per supergroup. K=6 was used in the primary PLSA run as a deliberate alignment question—whether six latent topics would work well when relevance is judged over six groups—not as a claim that the corpus inherently contains exactly six topics.

We retained the natural document distribution instead of downsampling large groups to match small ones. Downsampling would discard training examples and potentially exclude document variation. All 18,846 documents were kept in the corpus; a stratified 80/20 split assigned 15,076 documents to training and 3,770 to the held-out test set. Supergroup labels were used for stratification and relevance only. The vocabulary was fit on training text to avoid test-vocabulary leakage.

The six-group mapping is recorded in `Six_Category_Experiment.ipynb`. It assigns every source category once:

- **Computing:** `comp.graphics`, `comp.os.ms-windows.misc`, `comp.sys.ibm.pc.hardware`, `comp.sys.mac.hardware`, `comp.windows.x`
- **Digital technology and engineering:** `sci.crypt`, `sci.electronics`
- **Health and space science:** `sci.med`, `sci.space`
- **Recreation and consumer exchange:** `misc.forsale`, `rec.autos`, `rec.motorcycles`, `rec.sport.baseball`, `rec.sport.hockey`
- **Religion and worldview:** `alt.atheism`, `soc.religion.christian`, `talk.religion.misc`
- **Politics and international affairs:** `talk.politics.guns`, `talk.politics.mideast`, `talk.politics.misc`

These are an explicit working taxonomy, not an objectively established clustering. The saved category-centroid diagnostics show that the groups differ in cohesion; this mapping is one hypothesis to evaluate, not ground truth.

## Experiment 1: Six-supergroup retrieval

### Protocol and decisions

`Six_Category_Experiment.ipynb` loads all 20 categories (18,846 documents), applies the mapping above, and performs a stratified 80/20 split. Preprocessing builds a 3,000-term vocabulary from the training split. The saved run reports 15,076 training rows and 3,770 test rows; 446 training and 121 test documents become empty after preprocessing. Retrieval excludes empty test vectors, leaving 3,649 non-empty test candidates.

The primary PLSA model uses K=6, five random initializations, up to 100 EM iterations, tolerance 1e-4, and beta=1.0. The best run is selected by training log-likelihood, not by test retrieval performance. The saved best training log-likelihood is -6,273,009.364 after 44 iterations. TF-IDF/VSM and 50-dimensional LSI use the same train/test documents and candidate set.

There is one query per supergroup, with relevance defined as membership in that query’s supergroup. Evaluation reports Precision@10, Recall@10, MAP@10, and NDCG@10. A macro average over six queries is presented. Since the number of relevant documents differs substantially by group, Recall@10 is naturally small; it should be read with the other metrics, not as an isolated failure signal.

The notebook also inspects the learned topics, runs a controlled sports/politics mixture probe, calculates held-out topic/group NMI and ARI, and includes an exploratory K sweep. These are diagnostics, not additional retrieval benchmarks. For K values 2, 5, 6, 8, 10, and 20, the saved highest NMI is 0.363 at K=8 and highest ARI is 0.343 at K=8; K=6 is not the best alignment setting in this sweep. The sweep does not replace held-out retrieval comparison or establish a universally optimal K.

### Saved retrieval results

Macro averages over the six queries:

| Method | Precision@10 | Recall@10 | MAP@10 | NDCG@10 |
|---|---:|---:|---:|---:|
| TF-IDF/VSM | 0.8833 | 0.0166 | 0.8189 | 0.8982 |
| LSI | 0.8333 | 0.0167 | 0.7155 | 0.8077 |
| PLSA cosine, best restart | 0.4333 | 0.0068 | 0.2858 | 0.4248 |

Under this primary six-supergroup setup, single-run PLSA is substantially behind TF-IDF/VSM and LSI on the four retrieval metrics. This is the clean primary comparison before the ensemble investigation. It shows that changing to broader relevance groups alone did not make this single PLSA run competitive.

The saved held-out topic/group alignment for the K=6 model is NMI=0.306 and ARI=0.231. In the mixture probe, estimated mean sports-topic weight rises as the actual sports-word ratio increases, but it underestimates the strongest mixtures and varies considerably across sampled mixtures. This is evidence of some directional signal in this controlled probe, not evidence that the topics recover the six groups cleanly.

## Experiment 2: Bagging and restart aggregation

### Why this experiment was added

The initial comparison gave PLSA a single best restart, while concerns arose that stochastic fitting and sample variation could affect its rankings. We added ensembles to test whether aggregation improves retrieval and to treat all methods consistently: TF-IDF and LSI also receive bootstrap ensembles. This remains within the same six-supergroup task and does not change the underlying evaluation question.

`Bagging_Experiment.ipynb` uses the same corpus mapping, 80/20 split, training vocabulary, six queries, and test candidates as the six-supergroup notebook. It compares full-training-data reference models with five-member bootstrap ensembles. Each bootstrap sample has the training-set size, sampled with replacement. The same five saved bootstrap index sets are reused across PLSA, TF-IDF, and LSI. Ensemble rankings use mean rank aggregation. The notebook also compares PLSA cosine and query-likelihood scoring, and includes a five-restart PLSA rank ensemble trained on the full training set. This restart ensemble helps distinguish gains from combining initializations from gains involving bootstrap resampling.

Because the saved comparison includes several PLSA scoring/aggregation variants, the “bagged PLSA” rows should be interpreted alongside the full-data best-restart and full-data five-restart ensemble controls. Do not attribute every improvement to bootstrap sampling alone.

### Saved macro retrieval results

Macro averages over the same six queries:

| Method | Precision@10 | Recall@10 | MAP@10 | NDCG@10 |
|---|---:|---:|---:|---:|
| TF-IDF/VSM (full) | 0.8833 | 0.0166 | 0.8189 | 0.8982 |
| TF-IDF/VSM (bagged) | 0.8833 | 0.0166 | 0.8220 | 0.8990 |
| LSI (full) | 0.8333 | 0.0167 | 0.7155 | 0.8077 |
| LSI (bagged) | 0.8167 | 0.0163 | 0.6869 | 0.7988 |
| PLSA cosine (best restart) | 0.4333 | 0.0068 | 0.2858 | 0.4248 |
| PLSA cosine (5 full-data restarts) | 0.9167 | 0.0170 | 0.8845 | 0.9356 |
| PLSA cosine (bagged) | 0.9500 | 0.0185 | 0.9196 | 0.9505 |
| PLSA query likelihood (best restart) | 0.4833 | 0.0079 | 0.3567 | 0.4892 |
| PLSA query likelihood (5 full-data restarts) | 0.9000 | 0.0168 | 0.8260 | 0.8937 |
| PLSA query likelihood (bagged) | 0.9333 | 0.0181 | 0.9138 | 0.9491 |

In this saved run, TF-IDF bagging changes its MAP and NDCG only slightly; LSI bagging is slightly lower than its full-data reference. PLSA aggregation changes results substantially: its cosine five-restart full-data ensemble already exceeds the TF-IDF full-data reference on these six-query macro averages, and cosine bagging is higher again. Query-likelihood PLSA shows the same broad pattern, with the bagged ensemble above its own full-data and restart-ensemble controls.

These are promising results for PLSA under this particular grouping, query set, split, preprocessing, K, random seeds, candidate set, and rank aggregation rule. They do not establish that PLSA generally beats TF-IDF or LSI. The comparison has only six fixed queries, and the notebook should be rerun from a clean kernel before these outputs are treated as final submission evidence. Preserve per-query results when reporting, since macro averages can hide which supergroups drive the advantage.

## Files and handoff

The experiment notebooks are named `Six_Category_Experiment.ipynb` and `Bagging_Experiment.ipynb`. This `HANDOFF.md` replaces the previous project handoff and accompanies those notebooks for the next team member. The experiments use a shared six-supergroup, six-query, stratified 80/20 evaluation setup; their detailed decisions and saved macro results are documented above and in the notebooks.
