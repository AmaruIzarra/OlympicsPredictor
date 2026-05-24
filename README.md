# OlympicsPredictor

A group data science project analyzing what predicts Olympic medal outcomes, using 120 years of athlete data. Five independent analyses tackle the problem from different angles — predictive modelling, hypothesis testing, and statistical inference.

**Dataset:** [Global Olympic Athletes Performance Dataset](https://www.kaggle.com/datasets/eshummalik/global-olympic-athletes-performance-dataset) (Kaggle)

---

## Research Questions

| Analysis | Question |
|---|---|
| Multi-feature ML | Can an athlete's age, physical attributes, country, sport, and event predict medal outcome? |
| Logistic Regression & Naive Bayes | How do simpler models compare against Random Forest? |
| BMI alone | Is BMI on its own a useful predictor? |
| Home advantage | Does hosting the Olympics give a country a medal edge? |
| Age | Does age affect the probability of winning a medal? |
| Country effects | Are medal outcomes equal across countries, even after adjusting for participation? |

---

## Methods & Results

### Random Forest (Amaru)
Built a full ML pipeline with thoughtful feature engineering. High-cardinality categorical columns (Sport, NOC, Event) were encoded as smoothed historical medal rates rather than one-hot encoded, avoiding sparsity and data leakage. Age was mean-imputed; Height and Weight were imputed using sport- and sex-specific group means to preserve physical patterns.

**ROC-AUC: 0.832** — the model correctly ranks a medal winner above a non-winner 83% of the time. Recall for medal winners was 73.8%, with lower precision (32%), a deliberate trade-off from using `class_weight='balanced'` to handle the heavily imbalanced 86/14 class split.

### Logistic Regression & Naive Bayes (Anant)
Ran a grid search over Logistic Regression hyperparameters (regularization strength, penalty type, solver) with 5-fold cross-validation. Compared against Gaussian Naive Bayes. Logistic Regression outperformed Naive Bayes on CV ROC-AUC, likely because features like Height and Weight are correlated — violating Naive Bayes' independence assumption.

### BMI as a Standalone Predictor (Chris)
Tested whether BMI alone could predict medal outcomes. Neither Logistic Regression nor Random Forest trained on BMI alone could identify medal winners — precision and recall for the medal class were near zero. Conclusion: BMI carries almost no predictive signal on its own, since different sports demand very different body types.

### Home Country Advantage (Jereton)
Mapped each Olympic year to its host country (1896–2016) and tested whether host-country athletes win medals at a higher rate. A two-sample t-test rejected the null hypothesis (p < 0.05), confirming a statistically significant home advantage. Visualizations showed host-country medal rates consistently above the global average.

### Age & Medal Probability (Paresh)
Examined how age relates to medal outcomes, split by individual vs. team sports. T-tests showed a statistically significant age difference between medalists and non-medalists overall and within each sport type. A probability curve by age bin showed medal likelihood peaking in the mid-20s. A separate analysis found experienced athletes (returning Olympians) win medals at older ages than first-time participants.

### Country-Level Medal Inequality (Patrick)
Used chi-square tests and negative binomial regression to analyze whether some countries win disproportionately more medals. Even after adjusting for delegation size, medal outcomes are highly unequal across countries. The regression showed a super-linear relationship — a 1% increase in athletes sent is associated with a ~1.47% increase in medals — suggesting larger delegations convert participation into medals more efficiently.

---

## Setup

```bash
git clone https://github.com/AmaruIzarra/OlympicsPredictor.git
cd OlympicsPredictor
pip install kagglehub pandas scikit-learn matplotlib seaborn scipy statsmodels
jupyter notebook analysis.ipynb
```

The dataset downloads automatically via `kagglehub` when the first cell is run.

---

## Key Takeaways

- Physical attributes (age, height, weight, BMI) alone are weak predictors. Context — what sport, what event, what country — matters far more.
- Country and event historical medal rates are the strongest features in the ML model.
- Host countries have a measurable advantage, and larger delegations consistently outperform smaller ones even after normalizing by participation.
- A ROC-AUC of 0.832 is a reasonable ceiling for this dataset; medal outcomes depend on factors (form on the day, injuries, head-to-head matchups) that no static dataset can capture.

---

## Contributors

| Name | Analysis |
|---|---|
| Amaru Izarra-Jacome | Random Forest, feature engineering, data pipeline |
| Anant | Logistic Regression, Naive Bayes, model comparison |
| Chris | BMI analysis |
| Jereton | Home country advantage, hypothesis testing |
| Paresh | Age and medal probability |
| Patrick | Country-level medal inequality, negative binomial regression |
