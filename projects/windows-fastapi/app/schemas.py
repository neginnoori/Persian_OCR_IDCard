from typing import Literal
from pydantic import BaseModel, Field


class FieldValue(BaseModel):
    label: str = ''
    value: str | None = None
    text_raw: str = ''
    box: tuple[int, int, int, int] | None = None
    source_region_ids: list[int] = Field(default_factory=list)
    status: Literal['candidate', 'missing', 'invalid_format'] = 'missing'
    review_required: bool = True
    assignment_method: str | None = None


class CardFields(BaseModel):
    national_id: FieldValue = Field(default_factory=FieldValue)
    first_name: FieldValue = Field(default_factory=FieldValue)
    last_name: FieldValue = Field(default_factory=FieldValue)
    date_of_birth: FieldValue = Field(default_factory=FieldValue)
    father_name: FieldValue = Field(default_factory=FieldValue)
    expiry_date: FieldValue = Field(default_factory=FieldValue)


class OCRResponse(BaseModel):
    request_id: str
    status: Literal['ocr_candidate', 'no_text', 'needs_review', 'not_found']
    fields: CardFields
    text_raw: str
    text_normalized: str
    warnings: list[str]
    review_required: bool = True
    preprocessing: str
    coordinate_space: str | None
    processed_image_size: tuple[int, int] | None
    processing_ms: float


class ErrorDetail(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    request_id: str
    error: ErrorDetail
