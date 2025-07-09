from typing import Any, Optional, List
from uuid import UUID
from pydantic import BaseModel


class PipelineRunInfo(BaseModel):
    status: str
    pipeline_run_id: UUID
    dataset_id: UUID
    dataset_name: str
    payload: Optional[Any] = None

    model_config = {
        "arbitrary_types_allowed": True,
    }


class PipelineRunStarted(PipelineRunInfo):
    status: str = "PipelineRunStarted"
    pass


class PipelineRunYield(PipelineRunInfo):
    status: str = "PipelineRunYield"
    pass


class PipelineRunCompleted(PipelineRunInfo):
    status: str = "PipelineRunCompleted"
    processed_file_ids: Optional[List[UUID]] = None  # Files that were successfully processed
    failed_file_ids: Optional[List[UUID]] = None     # Files that failed during processing


class PipelineRunErrored(PipelineRunInfo):
    status: str = "PipelineRunErrored"
    processed_file_ids: Optional[List[UUID]] = None  # Files that were successfully processed before error
    failed_file_ids: Optional[List[UUID]] = None     # Files that failed during processing
    unprocessed_file_ids: Optional[List[UUID]] = None  # Files that were not processed
