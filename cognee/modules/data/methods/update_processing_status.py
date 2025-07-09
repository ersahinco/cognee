from typing import List, Optional
from uuid import UUID
from sqlalchemy import update
from sqlalchemy.exc import SQLAlchemyError
from cognee.modules.data.models import Data, FileProcessingStatus
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.shared.logging_utils import get_logger
from cognee.modules.data.methods import get_datasets_by_name, get_dataset_data

logger = get_logger("file_processing_status")

async def prepare_files_for_tracking(datasets: List[str], user_id: str) -> List[Data]:
    """Prepare and return file data for status tracking."""
    file_data_items = []
    for dataset_name in datasets:
        dataset_results = await get_datasets_by_name([dataset_name], user_id)
        if dataset_results:
            dataset = dataset_results[0]
            dataset_data = await get_dataset_data(dataset.id)
            file_data_items.extend(dataset_data)
    return file_data_items

async def update_processing_status_batch(
    file_ids: List[UUID], 
    status: FileProcessingStatus,
    description: Optional[str] = None
) -> int:
    """Update file processing status in batches with optional description."""
    if not file_ids:
        return 0
    
    if len(file_ids) > 1000:
        raise ValueError("Cannot update more than 1000 files at once")
    
    logger.debug(f"Updating {len(file_ids)} files to {status.value}")
    
    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        try:
            update_values = {"processing_status": status}
            if description:
                update_values["status_description"] = description
                
            result = await session.execute(
                update(Data)
                .where(Data.id.in_(file_ids))
                .values(**update_values)
            )
            await session.commit()
            updated_count = result.rowcount
            
            if updated_count == 0:
                logger.warning(f"No files updated - IDs may not exist")
            elif updated_count != len(file_ids):
                logger.warning(f"Only {updated_count}/{len(file_ids)} files updated")
            
            return updated_count
            
        except SQLAlchemyError as e:
            await session.rollback()
            logger.error(f"Database error: {e}")
            raise
        except Exception as e:
            await session.rollback()
            logger.error(f"Unexpected error: {e}")
            raise 