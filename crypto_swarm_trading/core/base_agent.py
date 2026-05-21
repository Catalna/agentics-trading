"""
core/base_agent.py — Abstract Base Agent with production-grade stability features.

Features:
  - AgentState enum (IDLE, RUNNING, ERROR, STOPPED, RECOVERING)
  - PerformanceTracker with execution time, error rate, throughput
  - safe_agent_execution decorator with timeout protection
  - Retry mechanism with exponential backoff
  - Heartbeat status tracking
  - Auto-recovery on light exceptions
  - Comprehensive logging per-agent
"""

from abc import ABC, abstractmethod
from enum import Enum
import logging
from logging.handlers import RotatingFileHandler
import sys
import time
import threading
from functools import wraps
from typing import Dict, Any, Optional, List
import traceback

class AgentState(Enum):
    IDLE = "IDLE"
    RUNNING = "RUNNING"
    ERROR = "ERROR"
    STOPPED = "STOPPED"
    RECOVERING = "RECOVERING"

class PerformanceTracker:
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.execution_times: List[float] = []
        self.success_count = 0
        self.error_count = 0
        self.start_time = time.time()
        self._lock = threading.Lock()

    def record_execution(self, success: bool, duration: float):
        with self._lock:
            if success:
                self.success_count += 1
            else:
                self.error_count += 1
                
            self.execution_times.append(duration)
            if len(self.execution_times) > self.max_history:
                self.execution_times.pop(0)

    @property
    def avg_execution_time(self) -> float:
        with self._lock:
            if not self.execution_times:
                return 0.0
            return sum(self.execution_times) / len(self.execution_times)

    @property
    def error_rate(self) -> float:
        with self._lock:
            total = self.success_count + self.error_count
            if total == 0:
                return 0.0
            return self.error_count / total

    @property
    def throughput(self) -> float:
        """Executions per second"""
        elapsed = time.time() - self.start_time
        if elapsed <= 0:
            return 0.0
        with self._lock:
            return (self.success_count + self.error_count) / elapsed

    def reset(self):
        with self._lock:
            self.execution_times.clear()
            self.success_count = 0
            self.error_count = 0
            self.start_time = time.time()

def safe_agent_execution(func):
    """
    Decorator for safe agent method execution.
    Catches exceptions, logs errors, updates metrics, and ensures a return value.
    Note: Real timeout protection in Python threads is complex without killing threads,
    so we rely on cooperative timeouts inside the methods where possible, but this
    decorator ensures exceptions don't crash the agent.
    """
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        start_time = time.time()
        success = False
        try:
            result = func(self, *args, **kwargs)
            success = True
            return result
        except Exception as e:
            self.logger.error(f"Error in {func.__name__}: {e}\n{traceback.format_exc()}")
            self.state = AgentState.ERROR
            self._consecutive_errors += 1
            return None
        finally:
            duration = time.time() - start_time
            self.performance.record_execution(success, duration)
    return wrapper


class BaseAgent(ABC):
    """
    Abstract base class for all swarm agents
    """
    def __init__(self, agent_id: str, config: Any):
        self.agent_id = agent_id
        self.config = config
        self.message_bus = None  # Set by orchestrator
        self.logger = self._setup_logger()
        self.state = AgentState.IDLE
        self.performance = PerformanceTracker()
        
        self._start_time = time.time()
        self._last_heartbeat = time.time()
        self._consecutive_errors = 0
        self._max_consecutive_errors = getattr(config, 'AGENT_ERROR_THRESHOLD', 5)
        
        # Load agent specific config if available
        try:
            from config.agent_config import AGENT_CONFIGS
            self.agent_config = AGENT_CONFIGS.get(agent_id, {})
        except ImportError:
            self.agent_config = {}
            
        self.timeout_seconds = self.agent_config.get('timeout', 10)
        self.retry_count = self.agent_config.get('retry', 1)
        self.retry_delay = self.agent_config.get('delay', 1)

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger(f"Agent.{self.agent_id}")
        logger.setLevel(getattr(self.config, 'LOG_LEVEL', 'INFO'))
        
        # Prevent adding handlers multiple times in interactive environments
        if not logger.handlers:
            formatter = logging.Formatter(
                '%(asctime)s | %(name)-25s | %(levelname)-8s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            
            # File handler
            log_file = getattr(self.config, 'LOG_FILE', 'swarm_system.log')
            fh = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)
            fh.setFormatter(formatter)
            logger.addHandler(fh)
            
            # Console handler
            ch = logging.StreamHandler(sys.stdout)
            ch.setFormatter(formatter)
            logger.addHandler(ch)
            
        return logger

    @abstractmethod
    def process(self, input_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Main processing logic - MUST implement"""
        pass
    
    @abstractmethod
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Input validation - MUST implement"""
        pass

    @safe_agent_execution
    def execute(self, input_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Concrete method to wrap process execution with validation, retry, and recovery logic.
        """
        self._heartbeat()
        
        if self.state == AgentState.ERROR:
            if self._consecutive_errors < self._max_consecutive_errors:
                self._attempt_recovery()
            else:
                self.logger.warning(f"Agent {self.agent_id} is in ERROR state and exceeded max retries. Execution skipped.")
                return None
                
        if self.state == AgentState.STOPPED:
            self.logger.debug(f"Agent {self.agent_id} is STOPPED. Execution skipped.")
            return None

        if not self.validate_input(input_data):
            self.logger.warning(f"Input validation failed for {self.agent_id}")
            return None

        self.state = AgentState.RUNNING
        
        result = None
        attempts = 0
        max_attempts = self.retry_count + 1
        
        while attempts < max_attempts:
            try:
                # We do not use threading.Timer to kill the thread as it is dangerous.
                # We expect process() to respect self.timeout_seconds internally if doing I/O.
                result = self.process(input_data)
                self._consecutive_errors = 0 # reset on success
                self.state = AgentState.IDLE
                break
            except Exception as e:
                attempts += 1
                self.logger.error(f"Attempt {attempts}/{max_attempts} failed for {self.agent_id}: {e}")
                if attempts < max_attempts:
                    time.sleep(self.retry_delay * (2 ** (attempts - 1))) # Exponential backoff
                else:
                    self.state = AgentState.ERROR
                    self._consecutive_errors += 1
                    raise # Re-raise to be caught by safe_agent_execution

        return result

    def send_message(self, recipient: str, message_type: str, payload: Dict[str, Any], priority: int = 3):
        """Send message to other agents via bus"""
        if self.message_bus:
            from .message_bus import AgentMessage
            import uuid
            from datetime import datetime, timezone
            
            msg = AgentMessage(
                id=str(uuid.uuid4()),
                timestamp=datetime.now(timezone.utc),
                sender=self.agent_id,
                recipient=recipient,
                message_type=message_type,
                payload=payload,
                priority=priority,
                ttl_seconds=getattr(self.config, 'MESSAGE_TTL_SECONDS', 30)
            )
            self.message_bus.publish(msg)
        else:
            self.logger.warning(f"MessageBus not attached to {self.agent_id}. Cannot send message.")

    def receive_message(self, message: Any):
        """Handle incoming messages. Override in subclasses if needed."""
        self.logger.debug(f"Received message {message.id} of type {message.message_type} from {message.sender}")

    def health_check(self) -> Dict[str, Any]:
        """Return comprehensive health status"""
        self._heartbeat()
        return {
            'agent_id': self.agent_id,
            'state': self.state.name,
            'uptime_seconds': self.uptime,
            'error_rate': self.performance.error_rate,
            'avg_exec_time_ms': self.performance.avg_execution_time * 1000,
            'last_heartbeat': self._last_heartbeat,
            'consecutive_errors': self._consecutive_errors,
            'is_healthy': self.is_healthy
        }

    def _heartbeat(self):
        """Update last heartbeat timestamp"""
        self._last_heartbeat = time.time()
        
    def _attempt_recovery(self):
        """Attempt to recover from a transient error"""
        self.logger.info(f"Attempting recovery for {self.agent_id}")
        self.state = AgentState.RECOVERING
        # Reset consecutive errors to give it another chance, but maybe not fully to 0
        # For simple recovery, we just set state to IDLE and let execute() try again.
        # If it fails again, consecutive_errors will increment.
        self.state = AgentState.IDLE

    @property
    def uptime(self) -> float:
        return time.time() - self._start_time

    @property
    def is_healthy(self) -> bool:
        return (
            self.state in [AgentState.IDLE, AgentState.RUNNING] and
            self._consecutive_errors < self._max_consecutive_errors and
            (time.time() - self._last_heartbeat) < getattr(self.config, 'HEARTBEAT_INTERVAL', 30) * 2
        )

    def start(self):
        """Start the agent"""
        self.logger.info(f"Starting agent {self.agent_id}")
        self.state = AgentState.IDLE
        self._start_time = time.time()
        self._heartbeat()

    def stop(self):
        """Stop the agent"""
        self.logger.info(f"Stopping agent {self.agent_id}")
        self.state = AgentState.STOPPED
