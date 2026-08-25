from pydantic import BaseModel
from typing import Literal

class GenderResult(BaseModel):
    prediction: Literal["male", "female", "unknown"]
    confidence: float

class AgeBracketResult(BaseModel):
    prediction: Literal["18-30", "31-45", "46-60", "60+", "unknown"]
    confidence: float

class AnalyzeResponse(BaseModel):
    contact_id: str
    gender: GenderResult
    age_bracket: AgeBracketResult
    processing_ms: float
    audio_quality: Literal["good", "degraded", "insufficient"]