"""
Enhanced lock manager with timeout and recovery mechanisms
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional, Set
from dataclasses import dataclass, field
from threading import Lock as ThreadLock

logger = logging.getLogger(__name__)

@dataclass
class LockInfo:
    """Information about an active lock"""
    name: str
    acquired_at: float
    owner_task: Optional[str] = None
    owner_process: Optional[int] = None
    timeout: Optional[float] = None
    context: Dict[str, Any] = field(default_factory=dict)

class TimeoutLockManager:
    """Enhanced lock manager with timeout and recovery capabilities"""

    def __init__(self, default_timeout: float = 300.0):  # 5 minutes default
        self.default_timeout = default_timeout
        self._locks: Dict[str, asyncio.Lock] = {}
        self._lock_info: Dict[str, LockInfo] = {}
        self._cleanup_task: Optional[asyncio.Task] = None
        self._thread_lock = ThreadLock()  # For thread-safe access to _locks dict
        self._active_lock_names: Set[str] = set()

    async def start_cleanup_monitor(self):
        """Start the background cleanup task"""
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._cleanup_expired_locks())

    async def stop_cleanup_monitor(self):
        """Stop the background cleanup task"""
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

    @asynccontextmanager
    async def acquire_lock(
        self,
        lock_name: str,
        timeout: Optional[float] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        """
        Acquire a lock with timeout and automatic cleanup

        Args:
            lock_name: Name of the lock
            timeout: Lock timeout in seconds (None = no timeout)
            context: Additional context information for debugging
        """
        effective_timeout = timeout if timeout is not None else self.default_timeout
        start_time = time.time()

        # Get or create lock
        with self._thread_lock:
            if lock_name not in self._locks:
                self._locks[lock_name] = asyncio.Lock()
            lock = self._locks[lock_name]

        try:
            # Try to acquire with timeout
            if effective_timeout > 0:
                await asyncio.wait_for(lock.acquire(), timeout=effective_timeout)
            else:
                await lock.acquire()

            # Record lock info
            current_task = asyncio.current_task()
            task_name = current_task.get_name() if current_task else None

            lock_info = LockInfo(
                name=lock_name,
                acquired_at=time.time(),
                owner_task=task_name,
                owner_process=None,  # Could add process info if needed
                timeout=effective_timeout if effective_timeout > 0 else None,
                context=context or {}
            )

            with self._thread_lock:
                self._lock_info[lock_name] = lock_info
                self._active_lock_names.add(lock_name)

            logger.debug(f"Lock '{lock_name}' acquired by task '{task_name}' after {time.time() - start_time:.2f}s")

            yield lock_info

        except asyncio.TimeoutError:
            logger.error(f"Lock '{lock_name}' acquisition timeout after {effective_timeout}s")
            await self._handle_timeout(lock_name, effective_timeout)
            raise asyncio.TimeoutError(f"Failed to acquire lock '{lock_name}' within {effective_timeout}s")

        except Exception as e:
            logger.error(f"Error acquiring lock '{lock_name}': {e}")
            raise

        finally:
            # Always release the lock
            try:
                if lock.locked():
                    lock.release()
                    logger.debug(f"Lock '{lock_name}' released")
            except Exception as e:
                logger.error(f"Error releasing lock '{lock_name}': {e}")

            # Clean up lock info
            with self._thread_lock:
                self._lock_info.pop(lock_name, None)
                self._active_lock_names.discard(lock_name)

    async def _handle_timeout(self, lock_name: str, timeout: float):
        """Handle lock timeout - log and potentially force cleanup"""
        with self._thread_lock:
            lock_info = self._lock_info.get(lock_name)

        if lock_info:
            held_time = time.time() - lock_info.acquired_at
            logger.warning(
                f"Lock '{lock_name}' timeout - held for {held_time:.2f}s by task '{lock_info.owner_task}'"
            )

        # Could add forced cleanup logic here if needed
        await self._emergency_cleanup(lock_name)

    async def _emergency_cleanup(self, lock_name: str):
        """Emergency cleanup for stuck locks"""
        try:
            with self._thread_lock:
                if lock_name in self._locks:
                    lock = self._locks[lock_name]
                    # Force release if locked (dangerous but necessary for recovery)
                    if lock.locked():
                        logger.warning(f"Force releasing stuck lock '{lock_name}'")
                        lock.release()
        except Exception as e:
            logger.error(f"Emergency cleanup failed for lock '{lock_name}': {e}")

    async def _cleanup_expired_locks(self):
        """Background task to clean up expired locks"""
        while True:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                current_time = time.time()
                expired_locks = []

                with self._thread_lock:
                    for lock_name, lock_info in self._lock_info.items():
                        if lock_info.timeout and (current_time - lock_info.acquired_at) > lock_info.timeout:
                            expired_locks.append(lock_name)

                for lock_name in expired_locks:
                    logger.warning(f"Cleaning up expired lock: {lock_name}")
                    await self._emergency_cleanup(lock_name)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in lock cleanup task: {e}")

    def get_lock_status(self) -> Dict[str, Any]:
        """Get current lock status for monitoring"""
        with self._thread_lock:
            return {
                "total_locks": len(self._locks),
                "active_locks": len(self._active_lock_names),
                "lock_details": {
                    name: {
                        "acquired_at": info.acquired_at,
                        "held_time": time.time() - info.acquired_at,
                        "owner_task": info.owner_task,
                        "timeout": info.timeout,
                        "context": info.context
                    }
                    for name, info in self._lock_info.items()
                }
            }

    async def force_release_all(self):
        """Emergency: force release all locks"""
        logger.warning("Force releasing all locks - emergency recovery")

        with self._thread_lock:
            for lock_name, lock in self._locks.items():
                try:
                    if lock.locked():
                        lock.release()
                        logger.warning(f"Force released lock: {lock_name}")
                except Exception as e:
                    logger.error(f"Failed to force release lock '{lock_name}': {e}")

            self._lock_info.clear()
            self._active_lock_names.clear()

# Global instance
_lock_manager = TimeoutLockManager()

async def get_timeout_lock(
    name: str,
    timeout: float = 300.0,
    context: Optional[Dict[str, Any]] = None
):
    """Get a timeout-enabled lock"""
    return _lock_manager.acquire_lock(name, timeout, context)

async def start_lock_manager():
    """Start the global lock manager"""
    await _lock_manager.start_cleanup_monitor()

async def stop_lock_manager():
    """Stop the global lock manager"""
    await _lock_manager.stop_cleanup_monitor()

def get_lock_manager_status():
    """Get lock manager status for monitoring"""
    return _lock_manager.get_lock_status()

async def emergency_release_all_locks():
    """Emergency function to release all locks"""
    await _lock_manager.force_release_all()