# 📰 Fake News Detection API (Binary Classifier - Real vs Fake)

Initially, i experimented with Decision Tree and Random Forest classifiers. However, due to the imbalanced nature of the LIAR dataset, these models failed to achieve satisfactory accuracy and F1 scores.

To address this, we switched to XGBoost, which:

Handles class imbalance more effectively through scale_pos_weight and sample_weight

Supports early stopping and regularization

Performs better in high-dimensional sparse data (like TF-IDF)

As a result, XGBoost achieved better generalization and classification performance, especially on minority class predictions.


This project implements a **Fake News Detection system** using **XGBoost**, **TF-IDF**, and **Truncated SVD**, wrapped inside a **Flask API** for real-time prediction.

The model classifies a political statement as either **Real** or **Fake** based on various features such as:
- The statement text
- Context
- Speaker information
- Party affiliation
- Historical credibility counts

---

1. Clone this repository or download the code:

git clone https://github.com/yourusername/fake-news-detection-api.git
cd fake-news-detection-api

2. Install dependencies:


pip install -r requirements.txt

3. Train the model and save all artifacts:
python main.py

4. Start the Flask API:

python app.py

5. Endpoint: POST /predict


Send a JSON payload with the following fields:

{
  "statement": "The earth revolves around the sun",
  "context": "science",
  "job_title": "astronomer",
  "speaker": "Galileo",
  "state_info": "Italy",
  "party_affiliation": "Independent"
}