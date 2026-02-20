import config
import torch
import time
from fastapi import FastAPI
from pydantic import BaseModel
from model import BERTBaseUncased

app = FastAPI()

DEVICE = config.DEVICE

MODEL = BERTBaseUncased()
MODEL.load_state_dict(torch.load(config.MODEL_PATH, map_location=DEVICE))
MODEL.to(DEVICE)
MODEL.eval()


class TextRequest(BaseModel):
    sentence: str


def sentence_prediction(sentence):
    tokenizer = config.TOKENIZER
    max_len = config.MAX_LEN

    review = str(sentence)
    review = " ".join(review.split())

    inputs = tokenizer.encode_plus(
        review,
        None,
        add_special_tokens=True,
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_token_type_ids=True,
    )

    ids = torch.tensor(inputs["input_ids"], dtype=torch.long).unsqueeze(0)
    mask = torch.tensor(inputs["attention_mask"], dtype=torch.long).unsqueeze(0)
    token_type_ids = torch.tensor(inputs["token_type_ids"], dtype=torch.long).unsqueeze(0)

    ids = ids.to(DEVICE)
    mask = mask.to(DEVICE)
    token_type_ids = token_type_ids.to(DEVICE)

    with torch.no_grad():
        outputs = MODEL(ids=ids, mask=mask, token_type_ids=token_type_ids)
        outputs = torch.sigmoid(outputs).cpu().numpy()

    return outputs[0][0]


@app.post("/predict")
def predict(request: TextRequest):
    start_time = time.time()

    positive_prediction = sentence_prediction(request.sentence)
    negative_prediction = 1 - positive_prediction

    return {
        "positive": float(positive_prediction),
        "negative": float(negative_prediction),
        "sentence": request.sentence,
        "time_taken": round(time.time() - start_time, 4),
    }

@app.get("/")
def root():
    return {"message": "Sentiment API is running"}