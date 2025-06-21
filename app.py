from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("models/xgb_model.pkl")
tfidf = joblib.load("models/tfidf_vectorizer.pkl")
svd = joblib.load("models/svd_transformer.pkl")
label_encoders = joblib.load("models/label_encoders.pkl")

def safe_encode(label, encoder, col_name):
    if label not in encoder.classes_:
        if 'unknown' not in encoder.classes_:
            encoder.classes_ = np.append(encoder.classes_, 'unknown')
        return encoder.transform(['unknown'])[0]
    return encoder.transform([label])[0]

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    try:
        statement = data.get("statement", "")
        context = data.get("context", "")
        job_title = data.get("job_title", "")
        speaker = data.get("speaker", "")
        state_info = data.get("state_info", "")
        party_affiliation = data.get("party_affiliation", "")

        full_text = f"{statement} {context} {job_title}"
        X_text = tfidf.transform([full_text])
        X_text_svd = svd.transform(X_text)

        speaker_enc = safe_encode(speaker, label_encoders["speaker"], "speaker")
        state_enc = safe_encode(state_info, label_encoders["state_info"], "state_info")
        party_enc = safe_encode(party_affiliation, label_encoders["party_affiliation"], "party_affiliation")

        count_feats = np.array([[0, 0, 0, 0, 0]])

        cat_feats = np.array([[speaker_enc, state_enc, party_enc]])
        X_input = np.hstack([X_text_svd, count_feats, cat_feats])


        pred = model.predict(X_input)[0]
        label = "Real" if pred == 1 else "Fake"

        return jsonify({"prediction": int(pred), "label": label})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
