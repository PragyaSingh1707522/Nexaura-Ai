from pydantic import BaseModel, Field

class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    strategy: str | None = None
    document_ids: list[int] | None = None

class VendorRequest(BaseModel):
    vendor_name: str = Field(min_length=1, max_length=200)
    industry: str = "Other"
    data_sensitivity: int = Field(ge=1, le=5)
    access_level: int = Field(ge=1, le=5)
    control_maturity: int = Field(ge=1, le=5)
    incident_history: int = Field(ge=0, le=5)
    criticality: int = Field(ge=1, le=5)

class ReviewRequest(BaseModel):
    status: str = Field(pattern="^(approved|rejected|pending)$")
    reviewer: str = "reviewer"
    note: str = ""
