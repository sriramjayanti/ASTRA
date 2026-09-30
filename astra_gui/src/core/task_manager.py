"""
ASTRA Background Task Manager.
Thread pool and runnable workers ensuring DSP and AI inference never freeze the GUI.
"""

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal, Slot
from typing import Callable, Any, Dict, Optional
import traceback


class WorkerSignals(QObject):
    """Signals emitted by background worker runnable."""
    started = Signal(str)
    progress = Signal(float, str)
    result = Signal(object)
    finished = Signal()
    error = Signal(str, str)  # (error_type, traceback_str)


class BackgroundWorker(QRunnable):
    """Generic cancellable QRunnable executing long-running pipeline tasks."""

    def __init__(self, task_name: str, fn: Callable, *args, **kwargs):
        super().__init__()
        self.task_name = task_name
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()
        self._is_cancelled = False

    def cancel(self):
        self._is_cancelled = True

    @property
    def is_cancelled(self) -> bool:
        return self._is_cancelled

    @Slot()
    def run(self):
        if self._is_cancelled:
            return

        try:
            self.signals.started.emit(self.task_name)
        except (RuntimeError, ReferenceError):
            return

        try:
            # Inject cancel checker into kwargs if function accepts it
            if "cancel_check" in self.fn.__code__.co_varnames:
                self.kwargs["cancel_check"] = lambda: self._is_cancelled

            res = self.fn(*self.args, **self.kwargs)
            if not self._is_cancelled:
                try:
                    self.signals.result.emit(res)
                except (RuntimeError, ReferenceError):
                    pass
        except Exception as e:
            tb = traceback.format_exc()
            try:
                self.signals.error.emit(str(e), tb)
            except (RuntimeError, ReferenceError):
                pass
        finally:
            try:
                self.signals.finished.emit()
            except (RuntimeError, ReferenceError):
                pass


class TaskManager:
    """Manages thread pool worker allocation and task lifecycles."""

    def __init__(self, max_threads: Optional[int] = None):
        self.thread_pool = QThreadPool.globalInstance()
        if max_threads:
            self.thread_pool.setMaxThreadCount(max_threads)
        self.active_workers: Dict[str, BackgroundWorker] = {}

    def start_task(
        self,
        task_name: str,
        fn: Callable,
        on_result: Optional[Callable[[Any], None]] = None,
        on_progress: Optional[Callable[[float, str], None]] = None,
        on_error: Optional[Callable[[str, str], None]] = None,
        *args,
        **kwargs
    ) -> BackgroundWorker:
        """Launches a callable in the global background thread pool."""
        worker = BackgroundWorker(task_name, fn, *args, **kwargs)

        if on_result:
            worker.signals.result.connect(on_result)
        if on_progress:
            worker.signals.progress.connect(on_progress)
        if on_error:
            worker.signals.error.connect(on_error)

        def cleanup():
            self.active_workers.pop(task_name, None)

        worker.signals.finished.connect(cleanup)
        self.active_workers[task_name] = worker
        self.thread_pool.start(worker)
        return worker

    def cancel_task(self, task_name: str) -> bool:
        if task_name in self.active_workers:
            self.active_workers[task_name].cancel()
            return True
        return False

    def cancel_all(self):
        for worker in self.active_workers.values():
            worker.cancel()
        self.active_workers.clear()
