# Twitter Sentiment Analysis

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/YOUR_REPO/blob/main/Twitter_Sentiment_Analysis.ipynb)

A machine learning project for analyzing sentiment in tweets using Natural Language Processing (NLP) techniques and classification algorithms.

## 📋 Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Models](#models)
- [Results](#results)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project implements a sentiment analysis system for Twitter data, classifying tweets as either positive (non-hate speech) or negative (hate speech). The system uses various machine learning algorithms including Logistic Regression, Support Vector Machines (SVM), and Naive Bayes to achieve high accuracy in sentiment classification.

## 📊 Dataset

The project uses two datasets:
- **Training Set**: 31,962 tweets (after removing duplicates: 29,530)
- **Test Set**: 17,197 tweets (after removing duplicates: 16,130)

Each tweet is labeled as:
- `0`: Not hate speech
- `1`: Hate speech

### Data Distribution
- Class 0 (Not hate speech): ~93.2%
- Class 1 (Hate speech): ~6.8%

## ✨ Features

### Data Preprocessing
1. **Duplicate Removal**: Eliminates repeated tweets to avoid bias
2. **Text Cleaning**:
   - Lowercase conversion
   - URL removal
   - Mention (@username) removal
   - Hashtag symbol removal
   - Punctuation and special character removal
   - Extra whitespace removal
3. **Stopword Removal**: Filters common English words
4. **Tokenization**: Splits text into individual words
5. **Stemming**: Reduces words to their root form using Porter Stemmer
6. **TF-IDF Vectorization**: Converts text to numerical features (5000 features, unigrams + bigrams)
7. **Class Imbalance Handling**: Uses RandomOverSampler to balance the dataset

### Machine Learning Models
- **Logistic Regression**: Baseline model with liblinear solver
- **Linear SVM (LinearSVC)**: Support Vector Machine with linear kernel
- **Multinomial Naive Bayes**: Probabilistic classifier

## 🚀 Installation

### Prerequisites
```bash
Python 3.7+
```

### Required Packages

```bash
# Core Data Science Libraries
pandas>=1.3.0
numpy>=1.21.0

# Machine Learning
scikit-learn>=0.24.0
imbalanced-learn>=0.8.0

# Natural Language Processing
nltk>=3.6.0

# Visualization
matplotlib>=3.4.0
missingno>=0.5.0
```

### Install Dependencies

```bash
pip install pandas numpy scikit-learn imbalanced-learn nltk matplotlib missingno
```

### Download NLTK Data

```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```

## 💻 Usage

### Running the Notebook

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO
```

2. Launch Jupyter Notebook:
```bash
jupyter notebook Twitter_Sentiment_Analysis.ipynb
```

3. Or open directly in Google Colab using the badge above

### Quick Start Example

```python
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load data
df_train = pd.read_csv('data/train.csv')

# Preprocess and train (simplified example)
vectorizer = TfidfVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(df_train['clean_tweet'])
y_train = df_train['label']

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)
```

## 📁 Project Structure

```
Twitter_Sentiment_Analysis/
│
├── data/
│   ├── train.csv          # Training dataset
│   └── test.csv           # Test dataset
│
├── Twitter_Sentiment_Analysis.ipynb  # Main notebook
└── README.md              # Project documentation
```

## 🤖 Models

### 1. Logistic Regression
- **Solver**: liblinear
- **Max Iterations**: 1000
- **Evaluation Accuracy**: ~94.9%
- **AUC-ROC**: High performance on evaluation set

### 2. Linear SVM (LinearSVC)
- **Regularization (C)**: 1.0
- **Max Iterations**: 5000
- **Performance**: Competitive with Logistic Regression

### 3. Multinomial Naive Bayes
- **Type**: Probabilistic classifier
- **Best for**: Text classification tasks

## 📈 Results

### Model Performance (Evaluation Set)

**Logistic Regression:**
- Accuracy: 94.9%
- Precision (Class 0): 0.97
- Recall (Class 0): 0.93
- F1-Score (Class 0): 0.95
- Precision (Class 1): 0.93
- Recall (Class 1): 0.97
- F1-Score (Class 1): 0.95

### Confusion Matrix
```
[[5111  393]
 [ 164 5339]]
```

### Key Insights
- The model performs well on both classes despite initial class imbalance
- RandomOverSampler effectively balanced the training data
- TF-IDF with bigrams captures important contextual information
- ROC-AUC curves show excellent discrimination capability

## 🛠️ Technologies Used

### Programming Language
- Python 3.x

### Libraries & Frameworks
- **Data Manipulation**: pandas, numpy
- **Machine Learning**: scikit-learn
- **Imbalanced Learning**: imbalanced-learn
- **NLP**: NLTK (Natural Language Toolkit)
- **Visualization**: matplotlib, missingno
- **Text Processing**: TfidfVectorizer, PorterStemmer

### Development Environment
- Jupyter Notebook
- Google Colab (optional)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 👤 Author

**Your Name**
- GitHub: [@YOUR_USERNAME](https://github.com/YOUR_USERNAME)

## 🙏 Acknowledgments

- Dataset source: [Add dataset source if applicable]
- Inspired by various NLP and sentiment analysis research papers
- Thanks to the scikit-learn and NLTK communities

---

**Note**: Remember to replace `YOUR_USERNAME` and `YOUR_REPO` with your actual GitHub username and repository name in the Colab badge and links.
