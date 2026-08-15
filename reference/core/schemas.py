from typing import List, Literal, Optional
from pydantic import BaseModel, Field, ConfigDict, AliasChoices

class VulnerabilityFinding(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str = Field(min_length=1, max_length=256, description="A short, descriptive title of the vulnerability")
    severity: Literal["Low", "Medium", "High", "Critical"] = Field(
        description="The severity of the vulnerability (Low, Medium, High, Critical)"
    )
    description: str = Field(min_length=1, max_length=16384, description="A detailed description of the flaw and its potential impact")
    line_numbers: Optional[List[int]] = Field(default=None, description="The specific line numbers where the vulnerability occurs")
    remediation: str = Field(default="", max_length=16384, description="Steps or code suggestions to fix the vulnerability")
    status: Optional[str] = Field(default=None, max_length=32, description="Finding status in lifecycle")

class VulnerabilityReport(BaseModel):
    model_config = ConfigDict(extra="forbid")
    findings: List[VulnerabilityFinding] = Field(
        default_factory=list,
        validation_alias=AliasChoices("findings", "vulnerabilities"),
        description="A list of all vulnerabilities found in the file. Empty if none found."
    )

