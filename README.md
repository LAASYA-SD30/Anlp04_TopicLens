# TopicLens — LDA Topic Modelling Web App

Implements preprocessing, document-term matrix creation, LDA training, topic-word distributions, document-topic distributions, and a topic interpretation report.

## Run on Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5000.

The built-in sample corpus contains 29 short documents covering technology, business, sports, health, and environment. Replace or extend `DOCUMENTS` in `app.py` if your instructor provided a specific corpus. The matrix display shows the first 12 columns for readability, while the full matrix is used to train LDA.
