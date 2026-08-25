from pydantic import BaseModel
from typing import Literal, Optional

class GenderResult(BaseModel):
    prediction: Literal["male", "female", "unknown"]
    confidence: float

class AgeBracketResult(BaseModel):
    prediction: Literal["18-30", "31-45", "46-60", "60+", "unknown"]
    confidence: float

class LanguageResult(BaseModel):
    prediction: str
    confidence: float

class AnalyzeResponse(BaseModel):
    contact_id: str
    gender: GenderResult
    age_bracket: AgeBracketResult
    language: Optional[LanguageResult] = None
    processing_ms: float
    audio_quality: Literal["good", "degraded", "insufficient"]