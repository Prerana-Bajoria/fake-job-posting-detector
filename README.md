🕵️ Fake Job Posting Detector
A machine learning system to detect fraudulent job postings using NLP and structured features, helping job seekers avoid scams.
Problem Statement
Online job platforms are increasingly targeted by fraudulent postings that deceive job seekers. This project builds a binary classifier to distinguish real job postings from fake ones using text analysis and structural signals.
Dataset

Source: Fake and Real Job Postings Dataset - Kaggle
Size: 17,880 job postings (17,014 real, 866 fake)
Features: Job title, description, requirements, company profile, benefits, and structural metadata

Approach
1. Exploratory Data Analysis

Identified severe class imbalance (95% real, 5% fake)
Discovered key fraud signals:

Jobs with no company logo had 16% fraud rate vs 2% with logo
Fake postings were on average 25% shorter than real ones
Jobs with no screening questions had 2x higher fraud rate
Part-time and executive-level postings showed higher fraud rates



2. Text Preprocessing

Combined title, description, requirements, and company profile into a single text field
Cleaned text: lowercased, removed punctuation, stripped stopwords using NLTK
Vectorized using TF-IDF (5,000 features)

3. Feature Engineering

Combined TF-IDF text features with numerical features: has_company_logo, has_questions, telecommuting, text_length
Applied SMOTE on training data to handle class imbalance (balanced to 13,611 each)

4. Modeling
Trained and compared three classifiers:
ModelPrecision (Fraud)Recall (Fraud)F1 (Fraud)Logistic Regression0.610.870.72Random Forest0.990.650.79XGBoost0.930.720.81
XGBoost selected as final model - best balance of precision and recall.
5. Explainability (SHAP)
Applied SHAP TreeExplainer to interpret model predictions:

has_company_logo - strongest predictor; absence strongly signals fraud
text_length - shorter postings are more likely fraudulent
has_questions - no screening questions increases fraud likelihood
Buzzwords like "exciting", "successful", "duties" frequently appear in fake postings

Key Insight

Fake job postings tend to have no company logo, shorter descriptions, no screening questions, and rely on vague buzzwords - signals that are both statistically significant and intuitively explainable.

## Dataset Setup
The dataset is not included in this repository due to size constraints.

To download it:
1. Go to [Kaggle Dataset](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction)
2. Download `fake_job_postings.csv`
3. Place it in the root directory of this project

Tech Stack

Language: Python
Libraries: Pandas, Scikit-learn, XGBoost, SHAP, NLTK, Imbalanced-learn, Matplotlib, SciPy

How to Run
bash git clone https://github.com/Prerana-Bajoria/fake-job-posting-detector
cd fake-job-detector
pip install -r requirements.txt
jupyter notebook fake_job_detection.ipynb
Results
XGBoost achieved 98% overall accuracy with an F1 score of 0.81 for fraud detection on an imbalanced real-world dataset.
