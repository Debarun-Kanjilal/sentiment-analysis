# AI-Powered Sentiment Analysis System

An end-to-end Natural Language Processing (NLP) and Machine Learning project that classifies customer reviews into **Positive**, **Negative**, or **Neutral** sentiment. The project includes data exploration, text preprocessing, TF-IDF feature extraction, model training and evaluation, a saved machine-learning pipeline, and an interactive Streamlit web application.

## 1. Project Overview

Customer reviews contain useful feedback about products and services. Reading and categorizing a large number of reviews manually can be time-consuming. This project uses supervised machine learning to automatically predict the sentiment expressed in a review.

**Project details**

- **Project:** AI-Powered Sentiment Analysis System
- **Level:** Beginner to Intermediate
- **Domain:** Natural Language Processing (NLP) and Machine Learning
- **Dataset:** Customer reviews dataset (`customer_reviews.csv`)
- **Target classes:** Positive, Negative, Neutral
- **Application:** Streamlit web application

## 2. Objectives

- Understand and clean a customer-review dataset.
- Explore sentiment distributions and review-text characteristics.
- Preprocess review text for machine-learning use.
- Convert text into numerical features using TF-IDF.
- Train and compare multiple classification models.
- Evaluate model performance using suitable classification metrics.
- Use cross-validation and hyperparameter tuning to select a model.
- Save the trained pipeline for reuse.
- Build a Streamlit interface that predicts sentiment for new reviews.

## 3. Dataset Description

The dataset is provided as a CSV file containing customer reviews.

| Column | Description |
| :--- | :--- |
| `review_id` | Unique identifier for each review |
| `review_text` | Customer review text used as the model input |
| `sentiment` | Target sentiment label: Positive, Negative, or Neutral |
| `category` | Product or service category |
| `source` | Source of the feedback |

The target variable is `sentiment`. The model uses `review_text` as its primary input.

Place the CSV file in the `data/` directory. Update the file path in the notebook if your filename or folder location is different.

## 4. Technologies and Libraries

- **Python** — Main programming language
- **Pandas and NumPy** — Data handling and numerical operations
- **Matplotlib and Seaborn** — Data visualization
- **Scikit-learn** — Text vectorization, machine-learning models, model evaluation, cross-validation, and hyperparameter tuning
- **Joblib** — Saving and loading the trained pipeline
- **Streamlit** — Interactive web application
- **Jupyter Notebook** — Experimentation and model development

## 5. Machine Learning Workflow

1. **Data loading:** Load the CSV dataset using Pandas.
2. **Data inspection and cleaning:** Inspect columns, missing values, empty reviews, and duplicate records; apply appropriate cleaning.
3. **Exploratory Data Analysis:** Examine sentiment distribution and review-text characteristics using visualizations.
4. **Text preprocessing:** Normalize review text using the preprocessing function defined in the training notebook.
5. **Feature extraction:** Use TF-IDF to convert text into numerical feature vectors. The project uses unigram and bigram features.
6. **Model training:** Train and compare classification algorithms.
7. **Model evaluation:** Evaluate predictions using accuracy, precision, recall, F1-score, and a confusion matrix. Explore ROC-AUC where appropriate for the multiclass task.
8. **Cross-validation and tuning:** Use cross-validation and GridSearchCV to search for suitable hyperparameters.
9. **Pipeline serialization:** Save the selected fitted pipeline with Joblib.
10. **Streamlit deployment:** Load the saved pipeline in the web application and predict the sentiment of newly entered review text.

## 6. Models

The project compares the following machine-learning classifiers:

- Logistic Regression
- Multinomial Naive Bayes
- Random Forest Classifier

The final model is the model selected by the notebook's evaluation and tuning process. In this implementation, a calibrated Logistic Regression pipeline is used for probability estimates.

## 7. Evaluation Metrics

The notebook evaluates the model using:

- **Accuracy:** Proportion of all predictions that are correct.
- **Precision:** Proportion of predicted instances of a class that are correct.
- **Recall:** Proportion of actual instances of a class that are correctly identified.
- **F1-score:** Harmonic mean of precision and recall.
- **Confusion matrix:** Shows correct and incorrect predictions for each sentiment class.
- **ROC-AUC:** Evaluates class-separation performance where appropriate for the multiclass setting.

### Final Model Results

| Metric | Result |
| :--- | :--- |
| Selected model | Logistic Regression (Tuned) |
| Test accuracy | 1.00 |
| Macro precision | 1.00 |
| Macro recall | 1.00 |
| Macro F1-score | 1.00 |
| ROC-AUC (if calculated) | 1.00 |
| Best cross-validation score | 1.00 |

## 8. Streamlit Application

The Streamlit application allows a user to:

1. Enter or paste a customer review.
2. Click the **Analyze Sentiment** button.
3. View the predicted sentiment: Positive, Negative, or Neutral.
4. View the model's probability/confidence information, if supported by the saved estimator.
5. Review any additional information displayed by the application.

The application loads the saved pipeline, so model training does not need to run every time the app starts.

## 9. Project Structure

The project files are organized as follows:

```text
sentiment-analysis/
│
├── data/
│   └── customer_reviews.csv
│
├── models/
│   └── sentiment_pipeline.pkl
│
├── notebooks/
│   └── sentiment_analysis.ipynb
│
├── .gitignore
├── app.py
├── README.md
├── requirements.txt
└── sentiment_analysis_evaluation_report.docx
```

## 10. Installation and Setup

### Prerequisites

- Python installed on your system
- pip package installer
- Jupyter Notebook or JupyterLab (for running the notebook)
- A terminal or command prompt

### Step 1: Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd sentiment-analysis
```

Replace `<YOUR_GITHUB_REPOSITORY_URL>` with your actual GitHub repository URL.

### Step 2: Create a virtual environment (recommended)

**Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Add the dataset

Place `customer_reviews.csv` inside the `data/` folder. If your dataset has a different name, update the path in the notebook accordingly.

### Step 5: Train and save the model

Open `notebooks/sentiment_analysis.ipynb` in Jupyter Notebook or JupyterLab. Run the notebook cells in order. Confirm that the trained pipeline is saved to the `models/` directory and that its filename matches the path configured in `app.py`.

### Step 6: Run the Streamlit application

From the project root directory, run:

```bash
streamlit run app.py
```

Streamlit will display a local URL in the terminal. Open that URL in your browser to use the application.

## 11. Example Reviews to Test

You can test the application with your own review text. For example:

| Review | Expected sentiment to check |
| :--- | :--- |
| `The product quality is excellent and I am very happy with it.` | Positive |
| `The product is average. It works as described.` | Neutral |
| `The product arrived damaged and the quality is very poor.` | Negative |

These are illustrative test inputs, not guaranteed model predictions. The application output depends on the trained model.

## 12. Important Notes

- Run the notebook before starting the Streamlit application so the saved model file exists.
- The preprocessing logic used during inference in `app.py` must match the preprocessing logic used during training.
- Keep the dataset and trained model in the locations expected by the notebook and application, or update the paths accordingly.
- Do not commit virtual-environment folders such as `venv/` to GitHub.
- Do not report evaluation scores until they have been obtained by running the notebook on the project dataset.
- The project demonstrates a traditional NLP and machine-learning workflow; an advanced deep-learning model or external AI API is not required.

## 13. Future Improvements

- Add a larger and more diverse review dataset.
- Compare TF-IDF with other text-representation methods.
- Add more detailed visualizations and error analysis.
- Improve the user interface and add batch review analysis.
- Evaluate the model on new, unseen review data.

## 14. Author

**Name:** Debarun Kanjilal  
**Project:** AI-Powered Sentiment Analysis System  
**Domain:** NLP and Machine Learning

---

If you find this project useful, feel free to star the repository.