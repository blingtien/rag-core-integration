"""
Enhanced MinerU wrapper with timeout and error recovery
"""

import asyncio
import logging
import time
import signal
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass

from .document_processor import process_document_with_recovery
from .lock_manager import get_timeout_lock

logger = logging.getLogger(__name__)

@dataclass
class MineruConfig:
    """Configuration for MinerU processing"""
    timeout: float = 300.0  # 5 minutes per document
    max_retries: int = 2
    enable_formula: bool = True
    enable_table: bool = True
    device: str = "auto"
    backend: str = "pipeline"


class EnhancedMineruProcessor:
    """Enhanced MinerU processor with timeout and recovery"""

    def __init__(self, config: Optional[MineruConfig] = None):
        self.config = config or MineruConfig()
        self._active_processes: Dict[str, subprocess.Popen] = {}

    async def process_pdf_with_recovery(
        self,
        input_path: Union[str, Path],
        output_dir: Union[str, Path],
        doc_id: str,
        method: str = "auto",
        lang: Optional[str] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Process PDF with enhanced error recovery and timeout"""

        input_path = Path(input_path)
        output_dir = Path(output_dir)

        # Create processing function
        async def process_func():
            return await self._process_pdf_internal(
                input_path, output_dir, method, lang, **kwargs
            )

        # Use document processor for timeout and retry logic
        result = await process_document_with_recovery(
            doc_id=doc_id,
            file_path=str(input_path),
            process_func=process_func,
            context={
                "method": method,
                "lang": lang,
                "output_dir": str(output_dir),
                "kwargs": kwargs
            }
        )

        if not result.success:
            raise Exception(f"MinerU processing failed: {result.error}")

        return result

    async def _process_pdf_internal(
        self,
        input_path: Path,
        output_dir: Path,
        method: str,
        lang: Optional[str],
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Internal PDF processing with timeout control"""

        # Filter kwargs to supported parameters
        supported_params = {
            'backend', 'start_page', 'end_page', 'formula', 'table',
            'device', 'source', 'vlm_url'
        }
        mineru_kwargs = {k: v for k, v in kwargs.items() if k in supported_params}

        # Create unique process ID for tracking
        process_id = f"mineru_{int(time.time())}_{id(input_path)}"

        try:
            # Import and execute MinerU with timeout monitoring
            from raganything.parser import MineruParser
            parser = MineruParser()

            # Monitor the process execution
            start_time = time.time()

            # Use a lock to ensure only one MinerU process per document
            async with get_timeout_lock(
                f"mineru_processing_{input_path.name}",
                timeout=self.config.timeout,
                context={"input_path": str(input_path), "process_id": process_id}
            ):
                # Execute MinerU processing
                result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    self._run_mineru_sync,
                    parser,
                    input_path,
                    output_dir,
                    method,
                    lang,
                    mineru_kwargs,
                    process_id
                )

                processing_time = time.time() - start_time
                logger.info(f"MinerU processing completed in {processing_time:.2f}s for {input_path.name}")

                return result

        except asyncio.TimeoutError:
            logger.error(f"MinerU processing timeout for {input_path.name}")
            await self._cleanup_process(process_id)
            raise

        except Exception as e:
            logger.error(f"MinerU processing error for {input_path.name}: {e}")
            await self._cleanup_process(process_id)
            raise

        finally:
            # Ensure cleanup
            await self._cleanup_process(process_id)

    def _run_mineru_sync(
        self,
        parser,
        input_path: Path,
        output_dir: Path,
        method: str,
        lang: Optional[str],
        mineru_kwargs: Dict[str, Any],
        process_id: str
    ) -> List[Dict[str, Any]]:
        """Synchronous MinerU execution with process tracking"""

        try:
            # Store current process info for potential cleanup
            self._active_processes[process_id] = {
                "start_time": time.time(),
                "input_path": str(input_path),
                "pid": None  # Will be updated if we can get subprocess PID
            }

            # Call the original MinerU method
            parser._run_mineru_command(
                input_path=input_path,
                output_dir=output_dir,
                method=method,
                lang=lang,
                **mineru_kwargs
            )

            # Read the generated output files
            name_without_suff = input_path.stem
            content_list, _ = parser._read_output_files(
                output_dir, name_without_suff, method=method
            )

            return content_list

        except Exception as e:
            logger.error(f"Synchronous MinerU processing failed: {e}")
            raise

        finally:
            # Clean up process tracking
            self._active_processes.pop(process_id, None)

    async def _cleanup_process(self, process_id: str):
        """Clean up a specific process"""
        try:
            process_info = self._active_processes.get(process_id)
            if process_info:
                logger.warning(f"Cleaning up MinerU process {process_id}")

                # If we have subprocess info, try to terminate it
                if "pid" in process_info and process_info["pid"]:
                    try:
                        os.kill(process_info["pid"], signal.SIGTERM)
                        await asyncio.sleep(2)
                        os.kill(process_info["pid"], signal.SIGKILL)
                    except (ProcessLookupError, OSError):
                        pass  # Process already dead

                self._active_processes.pop(process_id, None)

        except Exception as e:
            logger.error(f"Error cleaning up process {process_id}: {e}")

    async def cleanup_all_processes(self):
        """Emergency cleanup of all active processes"""
        logger.warning("Emergency cleanup of all MinerU processes")

        for process_id in list(self._active_processes.keys()):
            await self._cleanup_process(process_id)

    def get_active_processes(self) -> Dict[str, Any]:
        """Get information about active processes"""
        current_time = time.time()
        return {
            process_id: {
                **info,
                "running_time": current_time - info["start_time"]
            }
            for process_id, info in self._active_processes.items()
        }


# Global enhanced processor
_enhanced_processor = EnhancedMineruProcessor()

async def process_with_enhanced_mineru(
    input_path: Union[str, Path],
    output_dir: Union[str, Path],
    doc_id: str,
    method: str = "auto",
    lang: Optional[str] = None,
    **kwargs
) -> List[Dict[str, Any]]:
    """Global function to process with enhanced MinerU"""
    return await _enhanced_processor.process_pdf_with_recovery(
        input_path, output_dir, doc_id, method, lang, **kwargs
    )

async def cleanup_all_mineru_processes():
    """Global cleanup function"""
    await _enhanced_processor.cleanup_all_processes()

def get_mineru_process_status():
    """Get status of all MinerU processes"""
    return _enhanced_processor.get_active_processes()