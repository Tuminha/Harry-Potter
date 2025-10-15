# 🪄 Harry Potter Fan Fiction Logistic Regression Analysis

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![StatsModels](https://img.shields.io/badge/StatsModels-0.13%2B-orange.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0%2B-green.svg)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)
![Status](https://img.shields.io/badge/Status-Active-green.svg)

**Mastering Logistic Regression with Interaction and Polynomial Terms on Harry Potter Fan Fiction Data**

[🎯 Overview](#-project-overview) • [📊 Results](#-results) • [🚀 Quick-Start](#-quick-start) • [📦 Learning Journey](#-learning-journey)

</div>

> First comprehensive analysis: Exploring complex relationships in fan fiction popularity — building advanced logistic regression models with interaction and polynomial terms. Next up: regularization and feature selection techniques.

---

## 👨‍💻 Author
<div align="center">

**Francisco Teixeira Barbosa**

[![GitHub](https://img.shields.io/badge/GitHub-Tuminha-black?style=flat&logo=github)](https://github.com/Tuminha)
[![Kaggle](https://img.shields.io/badge/Kaggle-Profile-20BEFF?style=flat&logo=kaggle&logoColor=white)](https://www.kaggle.com/franciscotbarbosa)
[![Email](https://img.shields.io/badge/Email-cisco%40periospot.com-blue?style=flat&logo=gmail)](mailto:cisco@periospot.com)
[![Twitter](https://img.shields.io/badge/Twitter-cisco__research-1DA1F2?style=flat&logo=twitter)](https://twitter.com/cisco_research)

*Learning Machine Learning through CodeCademy • Building AI solutions step by step*

</div>

---

## 🎯 Project Overview
- **What:** Advanced logistic regression analysis of Harry Potter fan fiction popularity using interaction and polynomial terms
- **Why:** Master complex feature engineering techniques and understand how variables interact in real-world data
- **Expected Outcome:** Comprehensive understanding of logistic regression with advanced terms, model interpretation skills, and insights into fan fiction popularity factors

### 🎓 Learning Objectives
- Master interaction terms (Binary×Quantitative, Quantitative×Quantitative, Binary×Binary)
- Understand polynomial (quadratic) terms for non-linear relationships
- Compare StatsModels vs scikit-learn approaches to logistic regression
- Develop model interpretation and evaluation skills
- Practice feature engineering and data visualization techniques

### 🏆 Key Achievements
- [x] Comprehensive EDA with advanced visualizations
- [x] Target variable creation from engagement metrics
- [x] Feature engineering with all interaction term types
- [x] Polynomial term implementation for non-linear relationships
- [x] Dual modeling approach (StatsModels + scikit-learn)
- [x] Model comparison and performance evaluation
- [x] Advanced result visualizations (ROC, feature importance)
- [ ] Regularization techniques (L1/L2)
- [ ] Cross-validation implementation
- [ ] Feature selection optimization

---

## 📊 Dataset / Domain
- **Source:** [Kaggle: Harry Potter Fanfiction Data](https://www.kaggle.com/datasets/nehatiwari03/harry-potter-fanfiction-data) (scraped from fanfiction.net)
- **Size:** ~10,000+ fan fiction stories
- **Target:** Binary popularity indicator based on engagement metrics (favorites, follows, reviews)
- **Inspiration Questions:** Most popular pairings, language trends, post-movie/book publication trends

**Variables:**
- **Quantitative:** words, reviews, favorites, follows
- **Binary:** harry, hermione, multiple, english, humor

---

## 🚀 Quick Start
### Prerequisites
```bash
pip install -r requirements.txt
# or
pip install pandas numpy matplotlib seaborn statsmodels scikit-learn jupyter
```

### Setup
```bash
git clone [repository_url]
cd harry_potter_logistic_regression
jupyter notebook notebooks/harry_potter_logistic_regression.ipynb
```

---

## 📈 Project Phases
### Phase 1: Introduction & Setup ✅
<details>
<summary><strong>Details</strong></summary>

- [x] Import all necessary libraries (pandas, numpy, matplotlib, seaborn, statsmodels, sklearn)
- [x] Understand learning objectives and dataset structure
- [x] Set up analysis environment

</details>

### Phase 2: Data Loading & Initial Exploration ✅
<details>
<summary><strong>Details</strong></summary>

- [x] Load hp.csv dataset using pandas
- [x] Display first rows and dataset shape
- [x] Check data types and missing values
- [x] Generate descriptive statistics

</details>

### Phase 3: Exploratory Data Analysis (EDA) ✅
<details>
<summary><strong>Details</strong></summary>

- [x] Distribution plots for quantitative variables
- [x] Count plots for binary categorical variables
- [x] Scatter plots for key relationships
- [x] Correlation heatmap analysis
- [x] Pattern identification and reflection

</details>

### Phase 4: Target Variable Creation ✅
<details>
<summary><strong>Details</strong></summary>

- [x] Explore engagement metrics distributions
- [x] Create binary popularity indicator
- [x] Check target variable balance
- [x] Define popularity threshold strategy

</details>

### Phase 5: Feature Engineering - Interaction Terms ✅
<details>
<summary><strong>Details</strong></summary>

- [x] Binary × Quantitative interactions (harry×words, hermione×reviews, etc.)
- [x] Quantitative × Quantitative interactions (words×reviews, favorites×follows, etc.)
- [x] Binary × Binary interactions (harry×hermione, multiple×english, etc.)
- [x] Interaction term interpretation and significance

</details>

### Phase 6: Feature Engineering - Polynomial Terms ✅
<details>
<summary><strong>Details</strong></summary>

- [x] Quadratic terms for all quantitative variables
- [x] Non-linear relationship visualization
- [x] Polynomial term significance analysis

</details>

### Phase 7: Model Building with StatsModels ✅
<details>
<summary><strong>Details</strong></summary>

- [x] Feature matrix preparation
- [x] Constant term addition
- [x] Logistic regression model fitting
- [x] Detailed statistical output interpretation

</details>

### Phase 8: Model Building with scikit-learn ✅
<details>
<summary><strong>Details</strong></summary>

- [x] Train/test data splitting
- [x] LogisticRegression model implementation
- [x] Prediction generation and evaluation
- [x] Performance metrics calculation

</details>

### Phase 9: Model Comparison & Interpretation ✅
<details>
<summary><strong>Details</strong></summary>

- [x] Baseline model (original features only)
- [x] Full model performance comparison
- [x] Most significant interaction/polynomial terms identification
- [x] Feature importance analysis

</details>

### Phase 10: Visualization of Results ✅
<details>
<summary><strong>Details</strong></summary>

- [x] ROC curve and AUC calculation
- [x] Predicted probabilities distribution
- [x] Feature importance visualization
- [x] Confusion matrix heatmap

</details>

### Phase 11: Conclusion & Reflection ✅
- Summary: Comprehensive analysis of fan fiction popularity factors using advanced logistic regression techniques

---

## 🏆 Results
Final Model Performance (Threshold: 0.5):
├── Accuracy: [To be filled during analysis]
├── ROC-AUC: [To be filled during analysis]
├── Precision: [To be filled during analysis]
├── Recall: [To be filled during analysis]
├── F1-Score: [To be filled during analysis]
└── Confusion Matrix: [To be filled during analysis]

### 📌 Business Interpretation
- **Most Important Features:** [To be identified during analysis]
- **Key Interactions:** [To be discovered during analysis]
- **Popularity Drivers:** [To be determined during analysis]

### 🖼 Visuals
<div align="center">

*Visualizations will be generated during the analysis and stored in the `images/` directory*

</div>

---

## 🛠 Technical Stack
| Component | Technology | Purpose |
|-----------|------------|---------|
| Data Processing | Pandas, NumPy | ETL & feature engineering |
| Visualization | Matplotlib, Seaborn | EDA & result visualization |
| Statistical Modeling | StatsModels | Detailed regression analysis |
| Machine Learning | Scikit-learn | Model training & evaluation |
| Development | Jupyter Notebook | Interactive analysis environment |
| Version Control | Git/GitHub | Project management |

---

## 📦 Learning Journey
- **Statistical Modeling** • **Feature Engineering** • **Model Interpretation** • **Data Visualization** • **Machine Learning Workflows**

---

## 🚀 Next Steps
- [ ] Implement regularization techniques (L1/L2) to prevent overfitting
- [ ] Add cross-validation for more robust performance estimates
- [ ] Explore feature selection methods (Recursive Feature Elimination)
- [ ] Compare with other algorithms (Random Forest, XGBoost)
- [ ] Deploy model as a web application for story popularity prediction

---

## 📄 License
MIT License (see [LICENSE](LICENSE))

<div align="center">

**⭐ Star this repo if you found it helpful! ⭐**  
*Building AI solutions one dataset at a time* 🚀

</div>
