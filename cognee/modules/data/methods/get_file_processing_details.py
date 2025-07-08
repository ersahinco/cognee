from typing import List, Optional, Union, Tuple
from uuid import UUID
from sqlalchemy import select, and_, func
from cognee.modules.data.models import Data, FileProcessingStatus, ProcessingMetrics
from cognee.infrastructure.databases.relational import get_relational_engine


async def get_file_processing_details(
    file_id: UUID, 
    dataset_id: Optional[UUID] = None,
    status_only: bool = False
) -> Union[Optional[Data], FileProcessingStatus]:
    """
    Get file processing details or just status.
    
    Args:
        file_id: The ID of the file
        dataset_id: Optional dataset ID to verify file belongs to dataset
        status_only: If True, returns only the status instead of full file data
        
    Returns:
        If status_only=True: FileProcessingStatus
        If status_only=False: Optional[Data] (full file data or None if not found)
    """
    db_engine = get_relational_engine()
    
    async with db_engine.get_async_session() as session:
        # Build base query
        if status_only:
            query = select(Data.processing_status)
        else:
            query = select(Data)
            
        # Add file ID filter
        query = query.where(Data.id == file_id)
        
        # Add dataset filter if provided
        if dataset_id:
            query = query.join(Data.datasets).where(Data.datasets.any(id=dataset_id))
        
        result = await session.execute(query)
        
        if status_only:
            status = result.scalar_one_or_none()
            return status if status is not None else FileProcessingStatus.UNPROCESSED
        else:
            return result.scalar_one_or_none()


async def get_dataset_files_processing_details(
    dataset_id: UUID,
    status_filter: Optional[FileProcessingStatus] = None,
    with_metrics: bool = False,
    limit: Optional[int] = None,
    offset: int = 0
) -> Union[List[Data], Tuple[List[Data], ProcessingMetrics]]:
    """
    Get files in a dataset with their processing status and optionally get metrics.
    
    Args:
        dataset_id: The dataset ID
        status_filter: Optional status to filter by
        with_metrics: If True, also returns metrics along with files
        limit: Optional limit for pagination
        offset: Offset for pagination
        
    Returns:
        If with_metrics=False: List[Data]
        If with_metrics=True: Tuple[List[Data], ProcessingMetrics]
    """
    db_engine = get_relational_engine()
    
    async with db_engine.get_async_session() as session:
        # Base query for files
        files_query = (
            select(Data)
            .join(Data.datasets)
            .where(Data.datasets.any(id=dataset_id))
            .offset(offset)
        )
        
        # Add status filter if provided
        if status_filter:
            files_query = files_query.where(Data.processing_status == status_filter)
        
        # Add limit if provided
        if limit:
            files_query = files_query.limit(limit)
        
        # Execute files query
        files_result = await session.execute(files_query)
        files = files_result.scalars().all()
        
        # If metrics not needed, return just files
        if not with_metrics:
            return files
            
        # Get metrics if requested
        metrics_query = (
            select(
                Data.processing_status,
                func.count(Data.id).label('count')
            )
            .join(Data.datasets)
            .where(Data.datasets.any(id=dataset_id))
            .group_by(Data.processing_status)
        )
        
        metrics_result = await session.execute(metrics_query)
        status_counts = dict(metrics_result.all())
        
        # Calculate metrics
        metrics = ProcessingMetrics(
            total_files=sum(status_counts.values()),
            processed_files=status_counts.get(FileProcessingStatus.PROCESSED, 0),
            failed_files=status_counts.get(FileProcessingStatus.ERROR, 0),
            processing_files=status_counts.get(FileProcessingStatus.PROCESSING, 0)
        )
        
        return files, metrics 