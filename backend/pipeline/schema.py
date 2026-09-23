"""schema.py — typed structures shared across the pipeline."""
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class Word:
    text: str
    start: float
    end: float


@dataclass
class LinguisticMetrics:
    duration_sec: float
    speaking_time_sec: float
    word_count: int
    wpm: float
    filler_count: int
    filler_rate_per_min: float
    filler_breakdown: dict
    hedge_count: int
    hedge_rate_per_min: float
    hedge_breakdown: dict
    avg_sentence_len_words: float
    long_sentence_count: int
    repeated_phrases: list
    trailing_off_count: int


@dataclass
class PauseMetrics:
    pause_count: int
    long_pause_count: int
    total_pause_sec: float
    avg_pause_sec: float
    longest_pause_sec: float


@dataclass
class ProsodyMetrics:
    pitch_mean_hz: Optional[float]
    pitch_stdev_hz: Optional[float]
    pitch_cv: Optional[float]
    intensity_mean_db: Optional[float]
    intensity_stdev_db: Optional[float]
    voiced_fraction: Optional[float]


@dataclass
class Finding:
    severity: str  # "info" | "watch" | "flag"
    category: str
    message: str
    evidence: str


@dataclass
class Report:
    transcript: str
    linguistics: LinguisticMetrics
    pauses: PauseMetrics
    prosody: ProsodyMetrics
    findings: list
    llm_summary: Optional[str] = None
    conversation: Optional[dict] = None  # set only when diarization was requested and succeeded

    def to_dict(self):
        return asdict(self)


@dataclass
class AnswerGrading:
    available: bool
    correctness_score: Optional[int] = None
    summary: Optional[str] = None
    strengths: list = field(default_factory=list)
    gaps: list = field(default_factory=list)
    unavailable_reason: Optional[str] = None


@dataclass
class DeliveryScore:
    score: int
    breakdown: dict  # category -> {penalty: int, note: str}


@dataclass
class PracticeReport:
    question: str
    transcript: str
    duration_sec: float
    linguistics: LinguisticMetrics
    pauses: PauseMetrics
    prosody: ProsodyMetrics
    findings: list
    delivery_score: DeliveryScore
    answer_grading: AnswerGrading

    def to_dict(self):
        return asdict(self)
