"""Pydantic schemas for SafetyExtractor v2 node output."""

from enum import Enum

from pydantic import BaseModel, Field, field_validator


class SafetyCategory(str, Enum):
    LOTO = "LOTO"
    PPE = "PPE"
    HAZARD_WARNING = "HAZARD_WARNING"
    ELECTRICAL = "ELECTRICAL"
    CHEMICAL = "CHEMICAL"
    MECHANICAL = "MECHANICAL"
    THERMAL = "THERMAL"
    PRESSURE = "PRESSURE"
    GENERAL = "GENERAL"


class SafetySeverity(str, Enum):
    DANGER = "DANGER"
    WARNING = "WARNING"
    CAUTION = "CAUTION"
    NOTICE = "NOTICE"


class ExtractionConfidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ExtractionStatus(str, Enum):
    SUCCESS = "success"
    EXTRACTION_FAILED = "extraction_failed"
    SKIPPED = "skipped"


class SafetyProtocol(BaseModel):
    id: str = Field(default="")
    category: SafetyCategory = Field(default=SafetyCategory.GENERAL)
    severity: SafetySeverity = Field(default=SafetySeverity.WARNING)
    instruction: str = ""
    source_chunk_ids: list[str] = Field(default_factory=list)
    source_text_excerpt: str = Field(
        default="", description="<=100 char excerpt from source for citation"
    )
    applies_before_step: int | None = None
    applies_during_step: int | None = None
    applies_throughout: bool = False

    @field_validator("category", mode="before")
    @classmethod
    def _coerce_category(cls, v):
        if isinstance(v, SafetyCategory):
            return v
        try:
            return SafetyCategory(v)
        except (ValueError, KeyError):
            return SafetyCategory.GENERAL

    @field_validator("severity", mode="before")
    @classmethod
    def _coerce_severity(cls, v):
        if isinstance(v, SafetySeverity):
            return v
        try:
            return SafetySeverity(v)
        except (ValueError, KeyError):
            return SafetySeverity.WARNING


class SafetyExtractionMetadata(BaseModel):
    total_chunks_scanned: int = 0
    chunks_with_safety_content: int = 0
    signal_words_found: dict[str, int] = Field(
        default_factory=lambda: {
            "DANGER": 0,
            "WARNING": 0,
            "CAUTION": 0,
            "NOTICE": 0,
        }
    )
    extraction_confidence: ExtractionConfidence = ExtractionConfidence.LOW
    no_safety_content_found: bool = True
    status: ExtractionStatus = ExtractionStatus.SUCCESS


# Gemini structured output schema for the extraction call.
# Mirrors SafetyProtocol + SafetyExtractionMetadata but as a single
# response object the LLM returns.
class SafetyExtractionResponse(BaseModel):
    safety_protocols: list[SafetyProtocol] = Field(default_factory=list)
    extraction_metadata: SafetyExtractionMetadata = Field(
        default_factory=SafetyExtractionMetadata
    )
