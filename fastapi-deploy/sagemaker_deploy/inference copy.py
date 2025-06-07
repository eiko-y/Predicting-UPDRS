import joblib
import os
import numpy as np
import json
from typing import Any

def model_fn(model_dir: str) -> Any:
    model_path = os.path.join(model_dir, "model.pkl")
    model = joblib.load(model_path)
    return model

def input_fn(request_body: str, content_type: str) -> Any:
    if content_type == "application/json":
        input_data = json.loads(request_body)
        return input_data
    else:
        raise ValueError(f"Unsupported content type: {content_type}")

def predict_fn(input_data: Any, model: Any) -> Any:
    features = np.array([list(input_data.values())])
    prediction = model.predict(features)
    return prediction.tolist()

def output_fn(prediction: Any, content_type: str) -> str:
    if content_type == "application/json":
        return json.dumps({"predicted_total_UPDRS": prediction[0]})
    else:
        raise ValueError(f"Unsupported content type: {content_type}")
