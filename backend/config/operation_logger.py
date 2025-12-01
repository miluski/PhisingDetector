import logging
from datetime import datetime
from typing import Optional


class OperationLogger:
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.operation_id: Optional[str] = None
        self.start_time: Optional[datetime] = None

    def start_operation(self, operation_id: str, operation_name: str, **kwargs):
        self.operation_id = operation_id
        self.start_time = datetime.now()
        self._log_operation_header(operation_name, operation_id, kwargs)

    def _log_operation_header(self, name: str, op_id: str, params: dict):
        self.logger.info(f"{'='*60}")
        self.logger.info(f"🚀 STARTING: {name}")
        self.logger.info(f"   Operation ID: {op_id}")
        if params:
            details = " | ".join(f"{k}={v}" for k, v in params.items())
            self.logger.info(f"   Parameters: {details}")
        self.logger.info(f"{'='*60}")

    def step(self, step_name: str, **kwargs):
        details = " | ".join(f"{k}={v}" for k, v in kwargs.items()) if kwargs else ""
        msg = f"   ➜ {step_name}" + (f" ({details})" if details else "")
        self.logger.info(msg)

    def substep(self, message: str):
        self.logger.debug(f"      • {message}")

    def success(self, message: str, **kwargs):
        details = " | ".join(f"{k}={v}" for k, v in kwargs.items()) if kwargs else ""
        msg = f"   ✅ {message}" + (f" ({details})" if details else "")
        self.logger.info(msg)

    def warning(self, message: str):
        self.logger.warning(f"   ⚠️  {message}")

    def error(self, message: str, exception: Optional[Exception] = None):
        self.logger.error(f"   ❌ {message}")
        if exception:
            self.logger.error(f"      Exception: {type(exception).__name__}: {str(exception)}")

    def end_operation(self, success: bool = True, result_summary: str = ""):
        duration = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
        self._log_operation_footer(success, duration, result_summary)
        self._reset_state()

    def _log_operation_footer(self, success: bool, duration: float, summary: str):
        self.logger.info(f"{'='*60}")
        status = "✅ COMPLETED" if success else "❌ FAILED"
        self.logger.info(f"{status}: Operation {self.operation_id}")
        self.logger.info(f"   Duration: {duration:.2f}s")
        if summary:
            self.logger.info(f"   Result: {summary}")
        self.logger.info(f"{'='*60}\n")

    def _reset_state(self):
        self.operation_id = None
        self.start_time = None
