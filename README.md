# 📰 News Classification using spaCy Word Vectors & Machine Learning

A complete end-to-end machine learning project that classifies news articles as **Fake** or **Real** using **spaCy word vectors** and **scikit-learn** models. The system transforms raw news text into semantic embeddings and evaluates multiple classifiers, providing clear visualizations and comparison metrics.

---

## 🚀 Project Overview

Fake news detection has become a crucial topic due to the rapid spread of misinformation on digital platforms.
This project aims to build a baseline machine learning pipeline using:

* **spaCy (`en_core_web_lg`) word vectors** for text embeddings
* **Multinomial Naive Bayes**, **K-Nearest Neighbors**, and **Random Forest** for classification
* **Evaluation visualizations** such as Confusion Matrices, F1 Score Bar Chart, and ROC Curve

This workflow demonstrates how classical ML models combined with modern NLP embeddings can effectively classify short and long text documents.

---

## 📂 Dataset

The dataset used for this project includes:

* `text` — the news article content
* `category` — label indicating whether the news is *fake* or *real*

You must include your dataset file (e.g., `fake_real_news.csv`) in the repository or specify its location.

---

## 🧠 Features of This Project

* Converts raw news text into 300-dimensional spaCy vector embeddings
* Supports multiple ML models for comparison
* Scales embeddings for compatibility with Naive Bayes
* Generates:

  * Confusion Matrix heatmaps
  * F1 Score comparison chart
  * ROC Curve for model performance
* Clean, modular code suitable for learning and portfolio showcasing

---

## 🛠️ Technologies Used

* **Python 3**
* **spaCy** (`en_core_web_lg`)
* **scikit-learn**
* **NumPy & Pandas**
* **Matplotlib & Seaborn**
* **Joblib** (optional for saving models)

---

## 📦 Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/your-username/news-classification.git
cd news-classification
```

### 2. Install required libraries

```bash
pip install -r requirements.txt
```

### 3. Download spaCy large model

```bash
python -m spacy download en_core_web_lg
```

### 4. Run the script

```bash
python news_classification.py --data fake_real_news.csv
```

---

## 📊 Output & Visualizations

The project generates:

* **Classification Reports**
* **Confusion Matrices** (for NB, KNN, RF)
* **F1 Score Comparison Graph**
* **ROC Curve Comparison**

These visualizations help analyze which model performs best on the fake news detection task.

---

## 🏁 Conclusion

This project showcases how combining NLP word embeddings with classical machine learning algorithms can create an effective baseline for **fake news classification**. The workflow demonstrates text preprocessing, vectorization, model training, evaluation, and visualization — making it a strong foundation for more advanced transformer-based NLP systems.

---

## 📌 Future Improvements

* Add hyperparameter tuning (GridSearchCV / RandomSearchCV)
* Experiment with TF-IDF + Logistic Regression
* Integrate BERT or transformer-based embeddings
* Deploy using Flask / FastAPI

---

## 📄 License

This project is open-source and free to use for educational purposes. Add your preferred license here (MIT recommended).

---

## 👤 Author

**Your Name**
Add your GitHub profile link, email, or portfolio here.
