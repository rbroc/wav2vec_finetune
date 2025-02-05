"""evaluate model performance"""
import torch
import torch.nn.functional as F
import torchaudio

from transformers import AutoConfig, Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification

import numpy as np
import pandas as pd

import os

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

def speech_file_to_array_fn(path, sampling_rate):
    speech_array, _sampling_rate = torchaudio.load(path)
    resampler = torchaudio.transforms.Resample(_sampling_rate, sampling_rate)
    speech = resampler(speech_array).squeeze().numpy()
    return speech


def predict(path, sampling_rate):
    speech = speech_file_to_array_fn(path, sampling_rate)
    features = processor(speech, sampling_rate=sampling_rate, return_tensors="pt", padding=True)

    input_values = features.input_values.to(device)
    attention_mask = features.attention_mask.to(device)

    with torch.no_grad():
        logits = model(input_values, attention_mask=attention_mask).logits

    scores = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
    pred = config.id2label[np.argmax(scores)]
    confidence = scores[np.argmax(scores)]
    return pred, confidence


def add_predicted_and_confidence(df):
    pred, confidence = predict(df["file"], target_sampling_rate)
    df["pred"] = pred
    df["confidence"] = confidence
    return df

# setup model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = os.path.join("model", "xlsr_gender_recognition", "checkpoint-100")
config = AutoConfig.from_pretrained(model_path)
processor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)
target_sampling_rate = processor.sampling_rate
model = Wav2Vec2ForSequenceClassification.from_pretrained(model_path).to(device)

# load test data
test = pd.read_csv(os.path.join("preproc_data", "test_data.csv"), sep="\t")

# apply predictions
test = test.apply(add_predicted_and_confidence, axis=1)

print(confusion_matrix(test["label"], test["pred"]))
print(classification_report(test["label"], test["pred"]))
acc = accuracy_score(test["label"], test["pred"])
print(f"accuracy: {acc}")