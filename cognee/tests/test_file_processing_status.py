#!/usr/bin/env python3
"""
Test suite for file processing status tracking feature.

Core Requirements:
1. New files have default status of UNPROCESSED
2. File status updates during cognify process  
3. Status can be queried via API
4. Files can be filtered by status
5. Partial processing works correctly (only process files with specific status)
"""

import asyncio
import sys
import os
import tempfile
import pathlib
import uuid
from typing import List, Dict, Any

# Add the current directory to Python path so we can import cognee
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import cognee
from cognee.modules.data.models import FileProcessingStatus
from cognee.modules.data.methods import (
    get_file_processing_details,
    get_dataset_files_processing_details,
    update_processing_status_batch,
    get_datasets_by_name,
    get_dataset_data,
)
from cognee.modules.users.methods import get_default_user


async def setup_test_environment():
    """Setup clean test environment for testing."""
    print("🔧 Setting up test environment...")
    
    # Setup isolated test directories
    data_directory_path = str(pathlib.Path(os.path.join(pathlib.Path(__file__).parent, ".data_storage/test_file_processing")).resolve())
    cognee.config.data_root_directory(data_directory_path)
    
    cognee_directory_path = str(pathlib.Path(os.path.join(pathlib.Path(__file__).parent, ".cognee_system/test_file_processing")).resolve())
    cognee.config.system_root_directory(cognee_directory_path)

    # Clean up all test data and system files
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    print("   🧹 Cleaned up previous test data")
    

async def create_test_files(count: int = 6) -> List[str]:
    """Create a set of test files with unique content."""
    test_files = []
    test_dir = tempfile.mkdtemp(prefix="cognee_test_")
    
    for i in range(count):
        filename = f"test_file_{i+1}_{uuid.uuid4().hex[:8]}.txt"
        filepath = os.path.join(test_dir, filename)
        
        content = f"""Test File {i+1}
UUID: {uuid.uuid4()}
Content: This is test file number {i+1}.
Timestamp: {uuid.uuid4()}
Random: {uuid.uuid4().hex}

This file is created for testing the file processing status tracking feature.
It contains unique content to ensure proper processing and tracking.
"""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        test_files.append(filepath)
    
    return test_files, test_dir


async def test_default_status():
    """Test that new files have default status of UNPROCESSED."""
    print("\n🧪 Testing default file status")
    
    # Create test dataset with unique name
    dataset_name = f"default_status_test_{uuid.uuid4().hex[:8]}"
    test_files, test_dir = await create_test_files(count=3)
    
    try:
        # Add files to dataset
        for file_path in test_files:
            await cognee.add(file_path, dataset_name)
        
        # Get dataset and verify file status
        user = await get_default_user()
        datasets = await get_datasets_by_name(dataset_name, user.id)
        dataset = datasets[0]
        
        # Check all files are UNPROCESSED
        files = await get_dataset_files_processing_details(dataset.id, status_filter=FileProcessingStatus.UNPROCESSED)
        assert len(files) == len(test_files), f"Expected {len(test_files)} unprocessed files, got {len(files)}"
        
        print(f"   ✅ All {len(files)} files have default UNPROCESSED status")
        return True
        
    finally:
        # Cleanup
        for file_path in test_files:
            if os.path.exists(file_path):
                os.unlink(file_path)
        if os.path.exists(test_dir):
            os.rmdir(test_dir)


async def test_partial_processing():
    """Test that cognify only processes files with specific status."""
    print("\n🧪 Testing partial processing")
    
    # Create isolated test dataset
    dataset_name = f"partial_processing_test_{uuid.uuid4().hex[:8]}"
    test_files, test_dir = await create_test_files(count=6)
    
    try:
        # Add files to dataset
        for file_path in test_files:
            await cognee.add(file_path, dataset_name)
        
        # Get dataset
        user = await get_default_user()
        datasets = await get_datasets_by_name(dataset_name, user.id)
        dataset = datasets[0]
        
        # Get all files
        all_files = await get_dataset_data(dataset.id)
        assert len(all_files) == 6, f"Expected 6 files, got {len(all_files)}"
        
        # Mark first 2 files as PROCESSED
        processed_file_ids = [f.id for f in all_files[:2]]
        await update_processing_status_batch(processed_file_ids, FileProcessingStatus.PROCESSED)
        
        # Mark next 2 files as ERROR
        error_file_ids = [f.id for f in all_files[2:4]]
        await update_processing_status_batch(error_file_ids, FileProcessingStatus.ERROR)
        
        # Last 2 files remain UNPROCESSED
        unprocessed_file_ids = [f.id for f in all_files[4:]]
        
        print(f"   📊 Initial state:")
        print(f"      - PROCESSED: 2 files")
        print(f"      - ERROR: 2 files")
        print(f"      - UNPROCESSED: 2 files")
        
        # Run cognify - should only process UNPROCESSED files
        print(f"   🚀 Running cognify...")
        result = await cognee.cognify([dataset_name], user=user)
        
        # Verify final state
        final_files = await get_dataset_data(dataset.id)
        
        # Check previously processed files stayed processed
        still_processed = [f for f in final_files if f.id in processed_file_ids and f.processing_status == FileProcessingStatus.PROCESSED]
        assert len(still_processed) == 2, f"Expected 2 files to remain processed, got {len(still_processed)}"
        
        # Check error files stayed in error
        still_error = [f for f in final_files if f.id in error_file_ids and f.processing_status == FileProcessingStatus.ERROR]
        assert len(still_error) == 2, f"Expected 2 files to remain in error, got {len(still_error)}"
        
        # Check unprocessed files were processed
        newly_processed = [f for f in final_files if f.id in unprocessed_file_ids and f.processing_status == FileProcessingStatus.PROCESSED]
        newly_error = [f for f in final_files if f.id in unprocessed_file_ids and f.processing_status == FileProcessingStatus.ERROR]
        assert len(newly_processed) + len(newly_error) == 2, "Not all unprocessed files were handled"
        
        print(f"   📊 Final state verified:")
        print(f"      - Previously PROCESSED files: {len(still_processed)} (stayed processed)")
        print(f"      - Previously ERROR files: {len(still_error)} (stayed in error)")
        print(f"      - Previously UNPROCESSED files: {len(newly_processed)} processed, {len(newly_error)} errored")
        
        print("   ✅ Partial processing works correctly")
        return True
        
    finally:
        # Cleanup
        for file_path in test_files:
            if os.path.exists(file_path):
                os.unlink(file_path)
        if os.path.exists(test_dir):
            os.rmdir(test_dir)


async def test_status_updates():
    """Test that file status updates correctly during processing."""
    print("\n🧪 Testing status updates during processing")
    
    # Create test dataset
    dataset_name = f"status_updates_test_{uuid.uuid4().hex[:8]}"
    test_files, test_dir = await create_test_files(count=3)
    
    try:
        # Add files to dataset
        for file_path in test_files:
            await cognee.add(file_path, dataset_name)
    
        # Get dataset
        user = await get_default_user()
        datasets = await get_datasets_by_name(dataset_name, user.id)
        dataset = datasets[0]
        
        # Get initial files
        initial_files = await get_dataset_data(dataset.id)
        assert len(initial_files) == 3, f"Expected 3 files, got {len(initial_files)}"
        
        # Run cognify
        print(f"   🚀 Running cognify...")
        result = await cognee.cognify([dataset_name], user=user)
        
        # Get final state
        final_files = await get_dataset_data(dataset.id)
        
        # Verify all files reached a final state
        for file in final_files:
            assert file.processing_status in [FileProcessingStatus.PROCESSED, FileProcessingStatus.ERROR], \
                f"File {file.name} in unexpected state: {file.processing_status}"
        
        processed = [f for f in final_files if f.processing_status == FileProcessingStatus.PROCESSED]
        errored = [f for f in final_files if f.processing_status == FileProcessingStatus.ERROR]
        
        print(f"   📊 Final state:")
        print(f"      - PROCESSED: {len(processed)} files")
        print(f"      - ERROR: {len(errored)} files")
        
        print("   ✅ Status updates work correctly")
        return True

    finally:
        # Cleanup
        for file_path in test_files:
            if os.path.exists(file_path):
                os.unlink(file_path)
        if os.path.exists(test_dir):
            os.rmdir(test_dir)


async def test_status_filtering():
    """Test that files can be filtered by status."""
    print("\n🧪 Testing status filtering")
    
    # Create test dataset
    dataset_name = f"status_filter_test_{uuid.uuid4().hex[:8]}"
    test_files, test_dir = await create_test_files(count=6)
    
    try:
        # Add files to dataset
        for file_path in test_files:
            await cognee.add(file_path, dataset_name)
        
        # Get dataset
        user = await get_default_user()
        datasets = await get_datasets_by_name(dataset_name, user.id)
        dataset = datasets[0]
        
        # Get all files
        all_files = await get_dataset_data(dataset.id)
        
        # Set up mixed states
        await update_processing_status_batch([f.id for f in all_files[:2]], FileProcessingStatus.PROCESSED)
        await update_processing_status_batch([f.id for f in all_files[2:4]], FileProcessingStatus.ERROR)
        # Last 2 remain UNPROCESSED
        
        # Test filtering for each status
        for status in FileProcessingStatus:
            filtered = await get_dataset_files_processing_details(dataset.id, status_filter=status)
            expected_count = 2 if status in [FileProcessingStatus.PROCESSED, FileProcessingStatus.ERROR, FileProcessingStatus.UNPROCESSED] else 0
            assert len(filtered) == expected_count, f"Expected {expected_count} {status.value} files, got {len(filtered)}"
            print(f"   ✓ Found {len(filtered)} {status.value} files")
        
        print("   ✅ Status filtering works correctly")
        return True
        
    finally:
        # Cleanup
        for file_path in test_files:
            if os.path.exists(file_path):
                os.unlink(file_path)
        if os.path.exists(test_dir):
            os.rmdir(test_dir)


async def test_configurable_processing():
    """Test that cognify can be configured to process files with specific status."""
    print("\n🧪 Testing configurable processing status")
    
    # Create isolated test dataset
    dataset_name = f"configurable_processing_test_{uuid.uuid4().hex[:8]}"
    test_files, test_dir = await create_test_files(count=9)  # 3 files for each status
    
    try:
        # Add files to dataset
        for file_path in test_files:
            await cognee.add(file_path, dataset_name)
        
        # Get dataset
        user = await get_default_user()
        datasets = await get_datasets_by_name(dataset_name, user.id)
        dataset = datasets[0]
        
        # Get all files
        all_files = await get_dataset_data(dataset.id)
        assert len(all_files) == 9, f"Expected 9 files, got {len(all_files)}"
        
        # Set up mixed states (3 files each)
        await update_processing_status_batch([f.id for f in all_files[:3]], FileProcessingStatus.PROCESSED)
        await update_processing_status_batch([f.id for f in all_files[3:6]], FileProcessingStatus.ERROR)
        # Last 3 remain UNPROCESSED
        
        print(f"   📊 Initial state:")
        print(f"      - PROCESSED: 3 files")
        print(f"      - ERROR: 3 files")
        print(f"      - UNPROCESSED: 3 files")
        
        # Test processing ERROR files
        print(f"   🚀 Running cognify targeting ERROR files...")
        result = await cognee.cognify(
            [dataset_name], 
            user=user,
            target_status=FileProcessingStatus.ERROR
        )
        
        # Verify only ERROR files were processed
        final_files = await get_dataset_data(dataset.id)
        
        # Check PROCESSED files stayed unchanged
        still_processed = [f for f in final_files if f.id in [f.id for f in all_files[:3]] 
                         and f.processing_status == FileProcessingStatus.PROCESSED]
        assert len(still_processed) == 3, "Previously PROCESSED files should not be affected"
        
        # Check ERROR files were processed
        reprocessed = [f for f in final_files if f.id in [f.id for f in all_files[3:6]] 
                      and f.processing_status in [FileProcessingStatus.PROCESSED, FileProcessingStatus.ERROR]]
        assert len(reprocessed) == 3, "All ERROR files should be reprocessed"
        
        # Check UNPROCESSED files stayed unchanged
        still_unprocessed = [f for f in final_files if f.id in [f.id for f in all_files[6:]] 
                           and f.processing_status == FileProcessingStatus.UNPROCESSED]
        assert len(still_unprocessed) == 3, "UNPROCESSED files should not be affected"
        
        print(f"   📊 Final state verified:")
        print(f"      - Previously PROCESSED files: {len(still_processed)} (stayed processed)")
        print(f"      - Previously ERROR files: {len(reprocessed)} (reprocessed)")
        print(f"      - UNPROCESSED files: {len(still_unprocessed)} (stayed unprocessed)")
        
        # Test processing all files regardless of status
        print(f"\n   🚀 Running cognify with no status filter...")
        result = await cognee.cognify(
            [dataset_name], 
            user=user,
            target_status=None  # Process all files
        )
        
        # Verify all files were processed
        final_files = await get_dataset_data(dataset.id)
        processed_files = [f for f in final_files if f.processing_status == FileProcessingStatus.PROCESSED]
        error_files = [f for f in final_files if f.processing_status == FileProcessingStatus.ERROR]
        
        assert len(processed_files) + len(error_files) == 9, "All files should be processed"
        assert not any(f.processing_status == FileProcessingStatus.UNPROCESSED for f in final_files), \
            "No files should remain UNPROCESSED"
        
        print(f"   📊 Final state after processing all:")
        print(f"      - PROCESSED: {len(processed_files)} files")
        print(f"      - ERROR: {len(error_files)} files")
        print(f"      - UNPROCESSED: 0 files")
        
        print("   ✅ Configurable processing works correctly")
        return True
        
    finally:
        # Cleanup
        for file_path in test_files:
            if os.path.exists(file_path):
                os.unlink(file_path)
        if os.path.exists(test_dir):
            os.rmdir(test_dir)


async def run_tests():
    """Run all tests in sequence."""
    print("File Processing Status Tracking - Tests")
    print("=" * 50)
    
    # Setup environment
    await setup_test_environment()
    
    # Run tests
    tests = [
        test_default_status,
        test_status_updates,
        test_status_filtering,
        test_partial_processing,
        test_configurable_processing,  # Add the new test
    ]
    
    passed = 0
    for test in tests:
        try:
            if await test():
                passed += 1
        except Exception as e:
            print(f"   ❌ Test failed: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 50)
    if passed == len(tests):
        print(f"✅ All {passed} tests passed!")
    else:
        print(f"❌ {len(tests) - passed} tests failed ({passed}/{len(tests)} passed)")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(run_tests()) 