# TopicLens — LDA Topic Modelling Explorer

## Overview

TopicLens is a web-based Natural Language Processing application that explores latent themes in a collection of text documents using Latent Dirichlet Allocation (LDA).

The application compares Variational Bayes inference with Collapsed Gibbs Sampling and presents topic-word probabilities, document-topic distributions, and an interpretation report.

## Features

- Text preprocessing and cleaning
- Document-term matrix generation using CountVectorizer
- Variational Bayes LDA using scikit-learn
- Collapsed Gibbs Sampling with 100 iterations
- Topic-word probability visualisation
- Document-topic distribution comparison
- Interactive dashboard built with Flask

## Technologies Used

Python, Flask, scikit-learn, NumPy, HTML, CSS, JavaScript, and Jinja2.

## Project Structure

```text
TopicLens/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── templates/
│   └── index.html
├── static/
│   └── style.css
└── screenshots/
```

## Installation and Execution

1. Install Python 3.10 or a compatible version.
2. Clone the repository.
3. Install the dependencies:

```bash
pip install -r requirements.txt
```

4. Start the application:

```bash
python app.py
```

5. Open `http://127.0.0.1:5000/` in your browser.

## Screenshots

Screenshots of the dashboard, preprocessing, topic distributions, and interpretation report are available in the `screenshots/` folder.

## Academic Context

Developed as a demonstration project for Advanced Natural Language Processing (AML23702), focusing on probabilistic topic modelling and sampling-based inference.

## Limitations

The demonstration uses a small, predefined corpus and five topics. Topic labels are heuristic, and topic quality should be evaluated further on a larger dataset.
