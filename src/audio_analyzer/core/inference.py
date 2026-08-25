import torch
import numpy as np
import structlog
from transformers import Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)
from torch import nn
from src.audio_analyzer.config import settings

log = structlog.get_logger()

MODEL_NAME = "audeering/wav2vec2-large-robust-24-ft-age-gender"

_processor = None
_model = None


class ModelHead(nn.Module):
    def __init__(self, config, num_labels):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, num_labels)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class AgeGenderModel(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.wav2vec2 = Wav2Vec2Model(config)
        self.age = ModelHead(config, 1)
        self.gender = ModelHead(config, 3)
        self.init_weights()
    
    @property
    def all_tied_weights_keys(self):
        return []

    def forward(self, input_values):
        outputs = self.wav2vec2(input_values)
        hidden_states = outputs.last_hidden_state
        hidden_states = torch.mean(hidden_states, dim=1)
        age = self.age(hidden_states)
        gender = self.gender(hidden_states)
        return age, gender


def load_model():
    global _processor, _model
    if _processor is None:
        log.info("loading_model", model=MODEL_NAME)
        _processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
        _model = AgeGenderModel.from_pretrained(MODEL_NAME, ignore_mismatched_sizes=True)
        _model.eval()
        log.info("model_loaded", model=MODEL_NAME)


def predict(audio: np.ndarray) -> dict:
    load_model()

    max_samples = int(settings.sample_rate * 3.0)
    if len(audio) > max_samples:
        audio = audio[:max_samples]

    inputs = _processor(
        audio,
        sampling_rate=settings.sample_rate,
        return_tensors="pt",
        padding=True
    )

    with torch.no_grad():
        age_logits, gender_logits = _model(inputs["input_values"])

    age_val = age_logits[0, 0].item()
    gender_probs = torch.softmax(gender_logits[0], dim=-1)
    female_conf = gender_probs[0].item()
    male_conf = gender_probs[1].item()

    top_conf = max(female_conf, male_conf)
    if top_conf < 0.55:
        gender_pred = "unknown"
        gender_conf = 0.0
    else:
        gender_pred = "female" if female_conf > male_conf else "male"
        gender_conf = round(top_conf, 4)

    age_bracket, age_conf = _age_to_bracket(age_val)

    return {
        "gender": {"prediction": gender_pred, "confidence": gender_conf},
        "age_bracket": {"prediction": age_bracket, "confidence": age_conf}
    }


def _age_to_bracket(age_val: float) -> tuple[str, float]:
    age_years = age_val * 100
    brackets = [
        ("18-30", 18, 30),
        ("31-45", 31, 45),
        ("46-60", 46, 60),
        ("60+", 61, 100),
    ]
    for label, low, high in brackets:
        if low <= age_years <= high:
            center = (low + high) / 2
            distance = abs(age_years - center) / (high - low)
            confidence = round(1.0 - distance, 4)
            return label, confidence
    return "unknown", 0.0