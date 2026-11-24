# Bank term deposit prediction

Predicting which bank clients will subscribe to a term deposit after a telemarketing call, with nested validation, a custom stacking ensemble and an honest evaluation of when it helps.

## The question

A bank's marketing team can only call so many clients. Which ones should it call first? The goal is a model that ranks clients by their likelihood of subscribing, using only information available *before* the call, and an evaluation that says how well that ranking will work on clients the model has never seen.

## Data

- **Source:** UCI Machine Learning Repository, *Bank Marketing* (Portuguese bank telemarketing campaigns). The course provided a balanced subsample: 11,000 clients, 16 features, 47.5% subscribers.
- **Included in this repository** (`banks/`): the training set and the course's competition set.
- **Citation:** S. Moro, P. Cortez, P. Rita, "A data-driven approach to predict the success of bank telemarketing", *Decision Support Systems*, 62, 22–31, 2014.

## Method

- **Leakage control:** `duration` (call length) is removed. It is strongly predictive but only known after the call ends.
- **Feature engineering:** a custom `PdaysTransformer` separates "never contacted before" (`pdays = -1`) from the actual number of days since the last contact.
- **Evaluation design:** stratified holdout (2/3 train, 1/3 test) for the final estimate; stratified 5-fold cross-validation on the training set for every model choice.
- **Models:** KNN and decision trees (tuned with successive halving), logistic regression and RBF SVM (tuned with Optuna), compared against trivial baselines.
- **Stacking ensemble (gating network):** six experts (logistic regression, random forest and SVM, each tuned once for ROC-AUC and once for F2) feed their out-of-fold probabilities, together with the client's features, to a "director" random forest that learns which expert to trust for each kind of client. The decision threshold is chosen with Youden's J on out-of-fold predictions, never on the test set.
- **Deployment:** a Streamlit app (`mystreamlit.py`) serving the section 6 model.

## Results

All figures are on the held-out test set (3,667 clients the models never saw).

| Model | ROC-AUC | F2 |
|---|---|---|
| SVM (RBF), course's final model (section 6) | 0.663 | 0.700 |
| Best single expert: random forest (section 7) | **0.800** | 0.689 |
| Gating network (section 7) | 0.799 | 0.657 |
| Baseline: call every client | 0.500 | 0.819 |

**Ranking improved from 0.66 to 0.80 ROC-AUC.** Section 7's models are much better at ordering clients from most to least likely to subscribe, which is what a call list needs.

**The ensemble matched the best expert but did not beat it.** The gating network reached the same ROC-AUC as the best of its six experts without having to choose that expert in advance, but the experts' errors overlap too much for the routing to add value.

**F2 is misleading on this data.** The project originally optimised F2 (recall weighted twice as much as precision). With 47.5% subscribers, simply calling everyone scores F2 = 0.82, higher than any model. ROC-AUC is therefore the main comparison metric.

**Honest validation matters.** A first version scored the ensemble on its own training data and reported ROC-AUC 0.91. Evaluated on held-out data it scores 0.80. Its out-of-fold ROC-AUC on the training set (0.799) matches the test result, confirming the out-of-fold estimate is reliable.

## My contribution

Team of 2 (Matías Djukic and me) for the Machine Learning course (*Aprendizaje Automático*) at Universidad Carlos III de Madrid, 2026.

The project was built together. Matías built the first version of the EDA, the Streamlit deployment and the first version of the linear and SVM models (section 5), and reviewed the wording of the notebook's explanations and code comments. I made the project's main design decisions and:

- **Analysis and evaluation design:** extended the EDA into an in-depth analysis (multicollinearity, bivariate statistical tests, the `pdays` analysis, the `duration` leakage study) and designed the evaluation methodology (nested validation, asymmetric-cost metric, successive halving).
- **Restructured the pipeline** end to end and built the final model selection.
- **Designed and implemented the stacking ensemble** (section 7): the expert pool with dual-metric tuning, the gating network and its out-of-fold training.
- **Revision for publication (2026):** found that the ensemble had been evaluated on its training data, moved threshold selection to out-of-fold predictions and all metrics to the held-out set, added the always-yes baseline that exposes the F2 problem, and made the results reproducible.

## How to run

```bash
pip install -r requirements.txt
jupyter notebook eda_y_entrenamiento.ipynb     # full analysis and training
streamlit run mystreamlit.py                    # interactive prediction app
```

Tested with Python 3.13. The notebooks are written in Spanish. A full run is slow (section 7 trains several dozen models). The exported ensemble (`modelo_gating_network.joblib`, ~180 MB) is not included: running the notebook regenerates it.

## Repository structure

| Path | Contents |
|---|---|
| `eda_y_entrenamiento.ipynb` | EDA, evaluation design, models, final selection and the stacking ensemble |
| `predicciones_competicion.ipynb` | Loads the final model and generates the competition predictions |
| `mystreamlit.py` | Streamlit app for single-client predictions |
| `modelo_final.joblib` | Trained section 6 pipeline (used by the app) |
| `banks/` | Training and competition data |
| `predicciones*.csv`, `umbral_gating_network.joblib` | Competition predictions and the ensemble's decision threshold |

## License

MIT, see [LICENSE](LICENSE).