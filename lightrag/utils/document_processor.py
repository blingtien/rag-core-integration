"""
Enhanced document processor with timeout and error recovery
"""

import asyncio
import logging
import time
import traceback
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass
from contextlib import asynccontextmanager

from .lock_manager import get_timeout_lock

logger = logging.getLogger(__name__)

@dataclass
class ProcessingResult:
    """Result of document processing"""
    success: bool
    doc_id: str
    file_path: str
    processing_time: float
    error: Optional[str] = None
    retry_count: int = 0

class DocumentProcessor:
    """Enhanced document processor with timeout and recovery"""

    def __init__(
        self,
        max_processing_time: float = 600.0,  # 10 minutes per document
        max_retries: int = 2,
        retry_delay: float = 5.0,
    ):
        self.max_processing_time = max_processing_time
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._processing_stats = {
            "total_processed": 0,
            "successful": 0,
            "failed": 0,
            "timeouts": 0,
            "retries": 0
        }

    @asynccontextmanager
    async def process_with_timeout(
        self,
        doc_id: str,
        file_path: str,
        process_func: Callable,
        context: Optional[Dict[str, Any]] = None
    ):
        """Process a document with timeout and retry logic"""
        start_time = time.time()
        retry_count = 0
        last_error = None

        while retry_count <= self.max_retries:
            try:
                logger.info(f"Processing document {doc_id} (attempt {retry_count + 1}/{self.max_retries + 1})")

                # Create processing context
                processing_context = {
                    "doc_id": doc_id,
                    "file_path": file_path,
                    "attempt": retry_count + 1,
                    "max_time": self.max_processing_time,
                    **(context or {})
                }

                # Use timeout lock for the entire processing
                async with get_timeout_lock(
                    f"doc_processing_{doc_id}",
                    timeout=self.max_processing_time,
                    context=processing_context
                ):
                    # Execute the processing function with timeout
                    result = await asyncio.wait_for(
                        process_func(),
                        timeout=self.max_processing_time
                    )

                    processing_time = time.time() - start_time
                    self._processing_stats["total_processed"] += 1
                    self._processing_stats["successful"] += 1

                    if retry_count > 0:
                        self._processing_stats["retries"] += retry_count

                    yield ProcessingResult(
                        success=True,
                        doc_id=doc_id,
                        file_path=file_path,
                        processing_time=processing_time,
                        retry_count=retry_count
                    )
                    return

            except asyncio.TimeoutError:
                processing_time = time.time() - start_time
                error_msg = f"Document processing timeout after {processing_time:.2f}s"
                logger.error(f"Timeout processing {doc_id}: {error_msg}")

                self._processing_stats["timeouts"] += 1
                last_error = error_msg

                if retry_count < self.max_retries:
                    logger.info(f"Retrying {doc_id} after timeout (attempt {retry_count + 2})")
                    await asyncio.sleep(self.retry_delay)
                    retry_count += 1
                    continue
                else:
                    break

            except Exception as e:
                processing_time = time.time() - start_time
                error_msg = f"Processing error: {str(e)}"
                logger.error(f"Error processing {doc_id}: {error_msg}")
                logger.error(traceback.format_exc())

                last_error = error_msg

                if retry_count < self.max_retries and self._is_retryable_error(e):
                    logger.info(f"Retrying {doc_id} after error (attempt {retry_count + 2})")
                    await asyncio.sleep(self.retry_delay)
                    retry_count += 1
                    continue
                else:
                    break

        # All retries exhausted
        processing_time = time.time() - start_time
        self._processing_stats["total_processed"] += 1
        self._processing_stats["failed"] += 1

        if retry_count > 0:
            self._processing_stats["retries"] += retry_count

        yield ProcessingResult(
            success=False,
            doc_id=doc_id,
            file_path=file_path,
            processing_time=processing_time,
            error=last_error,
            retry_count=retry_count
        )

    def _is_retryable_error(self, error: Exception) -> bool:
        """Determine if an error is worth retrying"""
        # Add logic to determine retryable errors
        retryable_errors = (
            ConnectionError,
            TimeoutError,
            OSError,
            # Add more retryable error types
        )

        # Don't retry certain errors
        non_retryable_keywords = [
            "permission denied",
            "file not found",
            "invalid format",
            "corrupted",
        ]

        error_str = str(error).lower()
        if any(keyword in error_str for keyword in non_retryable_keywords):
            return False

        return isinstance(error, retryable_errors) or "connection" in error_str

    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        total = self._processing_stats["total_processed"]
        return {
            **self._processing_stats,
            "success_rate": (self._processing_stats["successful"] / total * 100) if total > 0 else 0,
            "failure_rate": (self._processing_stats["failed"] / total * 100) if total > 0 else 0,
            "timeout_rate": (self._processing_stats["timeouts"] / total * 100) if total > 0 else 0,
        }

    def reset_stats(self):
        """Reset processing statistics"""
        self._processing_stats = {
            "total_processed": 0,
            "successful": 0,
            "failed": 0,
            "timeouts": 0,
            "retries": 0
        }


class BatchDocumentProcessor:
    """Process multiple documents with better resource management"""

    def __init__(
        self,
        max_concurrent: int = 2,
        single_doc_timeout: float = 600.0,
        batch_timeout: float = 3600.0,  # 1 hour for entire batch
    ):
        self.max_concurrent = max_concurrent
        self.single_doc_timeout = single_doc_timeout
        self.batch_timeout = batch_timeout
        self.processor = DocumentProcessor(max_processing_time=single_doc_timeout)

    async def process_batch(
        self,
        documents: List[Dict[str, Any]],
        process_func: Callable,
        progress_callback: Optional[Callable] = None
    ) -> List[ProcessingResult]:
        """Process a batch of documents with proper resource management"""

        results = []
        semaphore = asyncio.Semaphore(self.max_concurrent)
        start_time = time.time()

        async def process_single_doc(doc_info: Dict[str, Any]) -> ProcessingResult:
            async with semaphore:
                doc_id = doc_info.get("doc_id")
                file_path = doc_info.get("file_path", "unknown")

                try:
                    async with self.processor.process_with_timeout(
                        doc_id=doc_id,
                        file_path=file_path,
                        process_func=lambda: process_func(doc_info)
                    ) as result:
                        if progress_callback:
                            await progress_callback(result)
                        return result

                except Exception as e:
                    logger.error(f"Unexpected error processing {doc_id}: {e}")
                    return ProcessingResult(
                        success=False,
                        doc_id=doc_id,
                        file_path=file_path,
                        processing_time=time.time() - start_time,
                        error=str(e)
                    )

        try:
            # Process all documents with overall timeout
            tasks = [process_single_doc(doc) for doc in documents]
            results = await asyncio.wait_for(
                asyncio.gather(*tasks),
                timeout=self.batch_timeout
            )

        except asyncio.TimeoutError:
            logger.error(f"Batch processing timeout after {self.batch_timeout}s")
            # Handle partial results if needed

        return results

    def get_stats(self) -> Dict[str, Any]:
        """Get batch processing statistics"""
        return self.processor.get_stats()


# Global processor instance
_global_processor = DocumentProcessor()

async def process_document_with_recovery(
    doc_id: str,
    file_path: str,
    process_func: Callable,
    context: Optional[Dict[str, Any]] = None
) -> ProcessingResult:
    """Global function to process a document with recovery"""
    async with _global_processor.process_with_timeout(
        doc_id, file_path, process_func, context
    ) as result:
        return result