import json
from typing import Any, List, Set
from uuid import UUID, uuid4

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.shared.logging_utils import get_logger
from cognee.modules.users.methods import get_default_user
from cognee.modules.pipelines.utils import generate_pipeline_id
from cognee.modules.pipelines.models.PipelineRunInfo import (
    PipelineRunCompleted,
    PipelineRunErrored,
    PipelineRunStarted,
    PipelineRunYield,
)

from cognee.modules.pipelines.operations import (
    log_pipeline_run_start,
    log_pipeline_run_complete,
    log_pipeline_run_error,
)
from cognee.modules.settings import get_current_settings
from cognee.modules.users.models import User
from cognee.shared.utils import send_telemetry

from .run_tasks_base import run_tasks_base
from ..tasks.task import Task

logger = get_logger("run_tasks(tasks: [Task], data)")


async def run_tasks_with_telemetry(
    tasks: list[Task], data, user: User, pipeline_name: str, context: dict = None
):
    config = get_current_settings()

    logger.debug("\nRunning pipeline with configuration:\n%s\n", json.dumps(config, indent=1))

    try:
        logger.info("Pipeline run started: `%s`", pipeline_name)
        send_telemetry(
            "Pipeline Run Started",
            user.id,
            additional_properties={
                "pipeline_name": str(pipeline_name),
            }
            | config,
        )

        async for result in run_tasks_base(tasks, data, user, context):
            yield result

        logger.info("Pipeline run completed: `%s`", pipeline_name)
        send_telemetry(
            "Pipeline Run Completed",
            user.id,
            additional_properties={
                "pipeline_name": str(pipeline_name),
            },
        )
    except Exception as error:
        logger.error(
            "Pipeline run errored: `%s`\n%s\n",
            pipeline_name,
            str(error),
            exc_info=True,
        )
        send_telemetry(
            "Pipeline Run Errored",
            user.id,
            additional_properties={
                "pipeline_name": str(pipeline_name),
            }
            | config,
        )

        raise error


async def run_tasks(
    tasks: list[Task],
    dataset_id: UUID,
    data: Any = None,
    user: User = None,
    pipeline_name: str = "unknown_pipeline",
    context: dict = None,
):
    if not user:
        user = get_default_user()

    # Get Dataset object
    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        from cognee.modules.data.models import Dataset

        dataset = await session.get(Dataset, dataset_id)

    pipeline_id = generate_pipeline_id(user.id, dataset.id, pipeline_name)

    pipeline_run = await log_pipeline_run_start(pipeline_id, pipeline_name, dataset_id, data)

    pipeline_run_id = pipeline_run.pipeline_run_id

    # Track file IDs and their explicit status
    initial_file_ids = set()
    processed_file_ids = set()
    failed_file_ids = set()
    
    if isinstance(data, list):
        for item in data:
            if hasattr(item, 'id'):
                initial_file_ids.add(item.id)
    elif hasattr(data, 'id'):
        initial_file_ids.add(data.id)

    logger.info(f"Pipeline tracking {len(initial_file_ids)} files: {list(initial_file_ids)}")

    yield PipelineRunStarted(
        pipeline_run_id=pipeline_run_id,
        dataset_id=dataset.id,
        dataset_name=dataset.name,
        payload=data,
    )

    try:
        last_result = None
        
        async for result in run_tasks_with_telemetry(
            tasks=tasks,
            data=data,
            user=user,
            pipeline_name=pipeline_id,
            context=context,
        ):
            last_result = result
            
            # Track only explicitly processed/failed files
            if hasattr(result, 'processed_files'):
                processed_file_ids.update(result.processed_files)
            if hasattr(result, 'failed_files'):
                failed_file_ids.update(result.failed_files)
            
            yield PipelineRunYield(
                pipeline_run_id=pipeline_run_id,
                dataset_id=dataset.id,
                dataset_name=dataset.name,
                payload=result,
            )

        # Calculate unprocessed files (not explicitly processed or failed)
        unprocessed_file_ids = initial_file_ids - processed_file_ids - failed_file_ids

        logger.info(f"Pipeline completed:")
        logger.info(f"- Processed: {len(processed_file_ids)} files")
        logger.info(f"- Failed: {len(failed_file_ids)} files")
        logger.info(f"- Remaining unprocessed: {len(unprocessed_file_ids)} files")

        if processed_file_ids:
            logger.info(f"Successfully processed file IDs: {list(processed_file_ids)}")
        if failed_file_ids:
            logger.warning(f"Failed file IDs: {list(failed_file_ids)}")
        if unprocessed_file_ids:
            logger.info(f"Unprocessed file IDs: {list(unprocessed_file_ids)}")

        await log_pipeline_run_complete(
            pipeline_run_id, pipeline_id, pipeline_name, dataset_id, last_result
        )

        yield PipelineRunCompleted(
            pipeline_run_id=pipeline_run_id,
            dataset_id=dataset.id,
            dataset_name=dataset.name,
            processed_file_ids=list(processed_file_ids) if processed_file_ids else None,
            failed_file_ids=list(failed_file_ids) if failed_file_ids else None,
        )

    except Exception as error:
        # On pipeline error, track unprocessed files separately from failed files
        unprocessed_file_ids = initial_file_ids - processed_file_ids - failed_file_ids

        logger.error(f"Pipeline errored:")
        logger.error(f"- Processed: {len(processed_file_ids)} files")
        logger.error(f"- Failed: {len(failed_file_ids)} files")
        logger.error(f"- Remaining unprocessed: {len(unprocessed_file_ids)} files")

        await log_pipeline_run_error(
            pipeline_run_id, pipeline_id, pipeline_name, dataset_id, data, error
        )

        yield PipelineRunErrored(
            pipeline_run_id=pipeline_run_id,
            payload=error,
            dataset_id=dataset.id,
            dataset_name=dataset.name,
            processed_file_ids=list(processed_file_ids) if processed_file_ids else None,
            failed_file_ids=list(failed_file_ids) if failed_file_ids else None,
            unprocessed_file_ids=list(unprocessed_file_ids) if unprocessed_file_ids else None,
        )
