"""User input contracts shared by HTTP and future interfaces."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Payload(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ProblemInput(Payload):
    question: str = Field(min_length=5, max_length=4000)
    dataset: str = "general"


class ChatInput(Payload):
    message: str = Field(min_length=1, max_length=6000)
    research: bool = False
    problem_id: str | None = None


class RunInput(Payload):
    kind: Literal["fixture", "p1", "research"]
    dataset_id: str | None = None
    live: bool = False


class ReferenceInput(Payload):
    id: str


class SolveInput(ReferenceInput):
    depth: Literal["focused", "balanced", "thorough"] = "balanced"


class RevisionInput(Payload):
    id: str
    assumption: str = Field(min_length=3, max_length=2000)


class ContextInput(Payload):
    problem_id: str
    title: str = Field(min_length=1, max_length=200)
    content: str = Field(min_length=1, max_length=1000000)
    format: Literal["text", "csv", "json", "geojson"] = "text"
    basis: Literal["user_context", "assumption", "observed_data"] = "user_context"


class WhatIfInput(Payload):
    model_id: str
    node_id: str
    value: float = Field(ge=0, le=1)


class UploadInput(Payload):
    problem_id: str
    filename: str = Field(min_length=1, max_length=200)
    content_base64: str = Field(max_length=2800000)


class ComputeInput(ReferenceInput):
    instruction: str = Field(default="", max_length=2000)
    predecessor_package_id: str | None = None


class SteeringInput(Payload):
    problem_id: str
    instruction: str = Field(min_length=1, max_length=4000)
    kind: Literal["context", "judgment", "constraint", "direction"] = "direction"


class SceneInput(ReferenceInput):
    blob_id: str
    time_column: str
    x_column: str
    y_column: str
    z_column: str
    group_column: str | None = None
    filters: dict[str, str] = Field(default_factory=dict, max_length=12)
