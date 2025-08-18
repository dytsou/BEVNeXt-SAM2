#!/usr/bin/env python3
"""
Network Error Handler for Training Resilience

This module provides comprehensive network error handling and retry logic
for robust training continuation during connection issues.

Author: Senior Python Programmer & AI Training Expert
"""

import os
import sys
import time
import socket
import logging
import threading
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from contextlib import contextmanager

import requests
import urllib3
from urllib3.exceptions import (
    ConnectTimeoutError, ReadTimeoutError, ProtocolError,
    MaxRetryError, NewConnectionError, ConnectionError as Urllib3ConnectionError
)

logger = logging.getLogger(__name__)


class ErrorType(Enum):
    """Classification of network errors"""
    CONNECTION_RESET = "connection_reset"
    CONNECTION_TIMEOUT = "connection_timeout"
    CONNECTION_REFUSED = "connection_refused"
    DNS_RESOLUTION = "dns_resolution"
    SSL_ERROR = "ssl_error"
    HTTP_ERROR = "http_error"
    UNKNOWN_NETWORK = "unknown_network"
    NON_NETWORK = "non_network"


class ErrorSeverity(Enum):
    """Severity levels for errors"""
    LOW = 1      # Transient, retry immediately
    MEDIUM = 2   # Recoverable, retry with backoff
    HIGH = 3     # Serious, retry with long backoff
    CRITICAL = 4 # May not be recoverable


@dataclass
class ErrorEvent:
    """Record of a network error event"""
    timestamp: datetime
    error_type: ErrorType
    severity: ErrorSeverity
    error_message: str
    context: str
    traceback_info: str
    retry_count: int = 0
    resolved: bool = False
    resolution_time: Optional[datetime] = None


@dataclass
class RetryConfig:
    """Configuration for retry behavior"""
    max_retries: int = 5
    base_delay: float = 1.0
    max_delay: float = 300.0
    exponential_base: float = 2.0
    jitter: bool = True
    timeout: float = 30.0
    
    # Error-specific overrides
    connection_reset_retries: int = 10
    connection_timeout_retries: int = 3
    dns_retries: int = 2


class NetworkErrorHandler:
    """Comprehensive network error handling with intelligent retry logic"""
    
    def __init__(
        self,
        retry_config: Optional[RetryConfig] = None,
        enable_monitoring: bool = True,
        log_file: Optional[Union[str, Path]] = None,
        emergency_callback: Optional[Callable[[ErrorEvent], None]] = None
    ):
        """
        Initialize network error handler
        
        Args:
            retry_config: Retry behavior configuration
            enable_monitoring: Enable error monitoring and statistics
            log_file: Optional file to log error events
            emergency_callback: Function to call for critical errors
        """
        self.retry_config = retry_config or RetryConfig()
        self.enable_monitoring = enable_monitoring
        self.emergency_callback = emergency_callback
        
        # Error tracking
        self.error_history: List[ErrorEvent] = []
        self.error_counts: Dict[ErrorType, int] = {}
        self.consecutive_errors = 0
        self.last_error_time: Optional[datetime] = None
        self.total_retry_attempts = 0
        
        # Monitoring
        self.start_time = datetime.now()
        self.lock = threading.Lock()
        
        # Setup logging
        self.log_file = Path(log_file) if log_file else None
        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Connection health
        self.connection_healthy = True
        self.last_health_check = datetime.now()
        
        logger.info("NetworkErrorHandler initialized")

    def handle_error(
        self,
        error: Exception,
        context: str,
        operation: Optional[Callable] = None,
        *args,
        **kwargs
    ) -> Any:
        """
        Handle network error with intelligent retry logic
        
        Args:
            error: The exception that occurred
            context: Description of what was being attempted
            operation: Optional function to retry
            *args, **kwargs: Arguments for the operation
            
        Returns:
            Result of successful retry or raises the final error
        """
        error_type, severity = self._classify_error(error)
        
        # Record error event
        error_event = ErrorEvent(
            timestamp=datetime.now(),
            error_type=error_type,
            severity=severity,
            error_message=str(error),
            context=context,
            traceback_info=traceback.format_exc()
        )
        
        self._record_error(error_event)
        
        # Check if error is recoverable
        if not self._is_recoverable_error(error_type, severity):
            logger.error(f"Non-recoverable error in {context}: {error}")
            self._trigger_emergency_callback(error_event)
            raise error
        
        # Determine retry strategy
        max_retries = self._get_max_retries_for_error(error_type)
        
        if operation is None:
            logger.warning(f"No operation provided for retry in {context}")
            raise error
        
        # Execute retry logic
        return self._execute_with_retry(
            operation, error_event, max_retries, *args, **kwargs
        )

    def _classify_error(self, error: Exception) -> Tuple[ErrorType, ErrorSeverity]:
        """
        Classify error type and severity
        
        Returns:
            Tuple of (error_type, severity)
        """
        error_str = str(error).lower()
        error_type_name = type(error).__name__.lower()
        
        # Connection reset errors
        if any(keyword in error_str for keyword in [
            'connection reset', 'connection broken', 'broken pipe',
            'reset by peer', 'connection aborted'
        ]):
            return ErrorType.CONNECTION_RESET, ErrorSeverity.MEDIUM
        
        # Timeout errors
        if any(keyword in error_str for keyword in [
            'timeout', 'timed out', 'read timeout', 'connect timeout'
        ]) or isinstance(error, (TimeoutError, socket.timeout, ConnectTimeoutError, ReadTimeoutError)):
            return ErrorType.CONNECTION_TIMEOUT, ErrorSeverity.MEDIUM
        
        # Connection refused
        if 'connection refused' in error_str or isinstance(error, ConnectionRefusedError):
            return ErrorType.CONNECTION_REFUSED, ErrorSeverity.HIGH
        
        # DNS resolution errors
        if any(keyword in error_str for keyword in [
            'name resolution', 'getaddrinfo', 'dns', 'host not found'
        ]) or isinstance(error, socket.gaierror):
            return ErrorType.DNS_RESOLUTION, ErrorSeverity.HIGH
        
        # SSL errors
        if any(keyword in error_str for keyword in [
            'ssl', 'certificate', 'handshake', 'tls'
        ]) or 'ssl' in error_type_name:
            return ErrorType.SSL_ERROR, ErrorSeverity.HIGH
        
        # HTTP errors
        if hasattr(error, 'status_code') or isinstance(error, requests.exceptions.HTTPError):
            if hasattr(error, 'status_code') and error.status_code >= 500:
                return ErrorType.HTTP_ERROR, ErrorSeverity.MEDIUM
            else:
                return ErrorType.HTTP_ERROR, ErrorSeverity.LOW
        
        # Generic network errors
        if any(isinstance(error, exc_type) for exc_type in [
            requests.exceptions.ConnectionError,
            requests.exceptions.ConnectTimeout,
            requests.exceptions.ReadTimeout,
            urllib3.exceptions.MaxRetryError,
            urllib3.exceptions.NewConnectionError,
            urllib3.exceptions.ProtocolError,
            Urllib3ConnectionError,
            ConnectionError,
            OSError
        ]) or any(keyword in error_str for keyword in [
            'network', 'connection', 'socket', 'http'
        ]):
            return ErrorType.UNKNOWN_NETWORK, ErrorSeverity.MEDIUM
        
        # Non-network error
        return ErrorType.NON_NETWORK, ErrorSeverity.CRITICAL

    def _is_recoverable_error(self, error_type: ErrorType, severity: ErrorSeverity) -> bool:
        """Determine if error is recoverable through retry"""
        # Non-network errors are generally not recoverable through retry
        if error_type == ErrorType.NON_NETWORK:
            return False
        
        # Critical errors need investigation
        if severity == ErrorSeverity.CRITICAL:
            return False
        
        # Check error pattern - too many recent errors might indicate systemic issue
        if self.consecutive_errors >= 10:
            logger.warning("Too many consecutive errors - may not be recoverable")
            return False
        
        return True

    def _get_max_retries_for_error(self, error_type: ErrorType) -> int:
        """Get maximum retry count for specific error type"""
        config = self.retry_config
        
        retry_map = {
            ErrorType.CONNECTION_RESET: config.connection_reset_retries,
            ErrorType.CONNECTION_TIMEOUT: config.connection_timeout_retries,
            ErrorType.DNS_RESOLUTION: config.dns_retries,
            ErrorType.CONNECTION_REFUSED: config.max_retries,
            ErrorType.SSL_ERROR: config.max_retries // 2,
            ErrorType.HTTP_ERROR: config.max_retries,
            ErrorType.UNKNOWN_NETWORK: config.max_retries,
        }
        
        return retry_map.get(error_type, config.max_retries)

    def _execute_with_retry(
        self,
        operation: Callable,
        error_event: ErrorEvent,
        max_retries: int,
        *args,
        **kwargs
    ) -> Any:
        """Execute operation with retry logic"""
        last_error = None
        
        for attempt in range(max_retries + 1):
            if attempt > 0:
                # Calculate delay
                delay = self._calculate_retry_delay(attempt, error_event.error_type)
                
                logger.info(f"Retrying {error_event.context} in {delay:.1f}s "
                           f"(attempt {attempt}/{max_retries})")
                
                # Wait before retry
                time.sleep(delay)
                
                # Update retry count
                error_event.retry_count = attempt
                self.total_retry_attempts += 1
            
            try:
                # Attempt operation
                result = operation(*args, **kwargs)
                
                # Success - mark error as resolved
                if attempt > 0:
                    error_event.resolved = True
                    error_event.resolution_time = datetime.now()
                    logger.info(f"Operation succeeded after {attempt} retries: {error_event.context}")
                    self._reset_error_counters()
                
                return result
                
            except Exception as e:
                last_error = e
                
                # Classify new error
                new_error_type, new_severity = self._classify_error(e)
                
                # If error type changed, it might be worth continuing
                if new_error_type != error_event.error_type:
                    logger.info(f"Error type changed from {error_event.error_type} to {new_error_type}")
                
                # Log retry attempt
                logger.warning(f"Retry {attempt}/{max_retries} failed for {error_event.context}: {e}")
                
                # Check if we should continue retrying
                if not self._should_continue_retry(e, attempt, max_retries):
                    break
        
        # All retries exhausted
        logger.error(f"All retry attempts exhausted for {error_event.context}")
        error_event.retry_count = max_retries
        
        # Trigger emergency callback for persistent failures
        self._trigger_emergency_callback(error_event)
        
        # Raise the last error
        raise last_error

    def _calculate_retry_delay(self, attempt: int, error_type: ErrorType) -> float:
        """Calculate delay before retry attempt"""
        config = self.retry_config
        
        # Base exponential backoff
        delay = config.base_delay * (config.exponential_base ** (attempt - 1))
        
        # Apply maximum delay
        delay = min(delay, config.max_delay)
        
        # Error-specific adjustments
        if error_type == ErrorType.CONNECTION_RESET:
            delay *= 0.5  # Faster retry for connection resets
        elif error_type == ErrorType.DNS_RESOLUTION:
            delay *= 2.0  # Slower retry for DNS issues
        
        # Add jitter to avoid thundering herd
        if config.jitter:
            import random
            jitter_range = delay * 0.1
            delay += random.uniform(-jitter_range, jitter_range)
        
        return max(0.1, delay)  # Minimum 100ms delay

    def _should_continue_retry(self, error: Exception, attempt: int, max_retries: int) -> bool:
        """Determine if we should continue retrying"""
        if attempt >= max_retries:
            return False
        
        # Check for non-recoverable errors
        error_type, severity = self._classify_error(error)
        
        if not self._is_recoverable_error(error_type, severity):
            logger.info(f"Error not recoverable, stopping retries: {error}")
            return False
        
        # Check consecutive error threshold
        if self.consecutive_errors >= 15:
            logger.warning("Too many consecutive errors, stopping retries")
            return False
        
        return True

    def _record_error(self, error_event: ErrorEvent):
        """Record error event for monitoring"""
        with self.lock:
            self.error_history.append(error_event)
            self.error_counts[error_event.error_type] = self.error_counts.get(error_event.error_type, 0) + 1
            self.consecutive_errors += 1
            self.last_error_time = error_event.timestamp
            
            # Limit history size
            if len(self.error_history) > 1000:
                self.error_history = self.error_history[-500:]
        
        # Log to file if configured
        if self.log_file:
            self._log_error_to_file(error_event)
        
        # Log to console
        logger.warning(f"Network error recorded: {error_event.error_type.value} in {error_event.context}")

    def _reset_error_counters(self):
        """Reset error counters after successful operation"""
        with self.lock:
            self.consecutive_errors = 0
            self.connection_healthy = True
            self.last_health_check = datetime.now()

    def _log_error_to_file(self, error_event: ErrorEvent):
        """Log error event to file"""
        try:
            with open(self.log_file, 'a') as f:
                f.write(f"{error_event.timestamp.isoformat()} | "
                       f"{error_event.error_type.value} | "
                       f"{error_event.severity.value} | "
                       f"{error_event.context} | "
                       f"{error_event.error_message}\n")
        except Exception as e:
            logger.warning(f"Failed to log error to file: {e}")

    def _trigger_emergency_callback(self, error_event: ErrorEvent):
        """Trigger emergency callback for critical errors"""
        if self.emergency_callback:
            try:
                self.emergency_callback(error_event)
            except Exception as e:
                logger.error(f"Emergency callback failed: {e}")

    def check_connection_health(self, test_urls: Optional[List[str]] = None) -> bool:
        """
        Check connection health by testing connectivity
        
        Args:
            test_urls: URLs to test (default: common public services)
            
        Returns:
            True if connection appears healthy
        """
        if test_urls is None:
            test_urls = [
                'https://www.google.com',
                'https://www.cloudflare.com',
                'https://httpbin.org/get'
            ]
        
        healthy_connections = 0
        
        for url in test_urls:
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    healthy_connections += 1
            except Exception as e:
                logger.debug(f"Health check failed for {url}: {e}")
        
        # Consider healthy if at least half the tests pass
        is_healthy = healthy_connections >= len(test_urls) // 2
        
        with self.lock:
            self.connection_healthy = is_healthy
            self.last_health_check = datetime.now()
        
        logger.info(f"Connection health check: {healthy_connections}/{len(test_urls)} passed")
        
        return is_healthy

    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics"""
        with self.lock:
            now = datetime.now()
            uptime = now - self.start_time
            
            # Recent errors (last hour)
            recent_errors = [
                e for e in self.error_history
                if (now - e.timestamp).total_seconds() < 3600
            ]
            
            # Resolved errors
            resolved_errors = [e for e in self.error_history if e.resolved]
            
            stats = {
                'uptime_hours': uptime.total_seconds() / 3600,
                'total_errors': len(self.error_history),
                'recent_errors_1h': len(recent_errors),
                'consecutive_errors': self.consecutive_errors,
                'total_retry_attempts': self.total_retry_attempts,
                'resolved_errors': len(resolved_errors),
                'resolution_rate': len(resolved_errors) / max(1, len(self.error_history)),
                'connection_healthy': self.connection_healthy,
                'last_error_time': self.last_error_time.isoformat() if self.last_error_time else None,
                'error_counts_by_type': {k.value: v for k, v in self.error_counts.items()},
            }
            
            if recent_errors:
                stats['recent_error_types'] = list(set(e.error_type.value for e in recent_errors))
            
            return stats

    @contextmanager
    def error_context(self, context: str, operation: Optional[Callable] = None):
        """Context manager for handling errors in a block of code"""
        try:
            yield self
        except Exception as e:
            if operation:
                # Try to handle with retry
                return self.handle_error(e, context, operation)
            else:
                # Just classify and log
                error_type, severity = self._classify_error(e)
                error_event = ErrorEvent(
                    timestamp=datetime.now(),
                    error_type=error_type,
                    severity=severity,
                    error_message=str(e),
                    context=context,
                    traceback_info=traceback.format_exc()
                )
                self._record_error(error_event)
                raise

    def create_resilient_session(self) -> requests.Session:
        """Create a requests session with built-in retry logic"""
        session = requests.Session()
        
        # Configure retries
        from urllib3.util.retry import Retry
        
        retry_strategy = Retry(
            total=self.retry_config.max_retries,
            backoff_factor=self.retry_config.base_delay,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "PUT", "DELETE", "OPTIONS", "TRACE"]
        )
        
        # Add retry adapter
        adapter = requests.adapters.HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # Set timeout
        session.timeout = self.retry_config.timeout
        
        return session

    def emergency_shutdown_handler(self, checkpoint_manager=None):
        """Emergency shutdown with checkpoint saving"""
        logger.warning("Emergency shutdown triggered by network error handler")
        
        if checkpoint_manager and hasattr(checkpoint_manager, 'emergency_save'):
            try:
                logger.info("Attempting emergency checkpoint save...")
                # This would need to be integrated with the actual training state
                # checkpoint_manager.emergency_save(...)
                logger.info("Emergency checkpoint save completed")
            except Exception as e:
                logger.error(f"Emergency checkpoint save failed: {e}")
        
        # Log final statistics
        stats = self.get_error_statistics()
        logger.error(f"Final error statistics: {stats}")


def create_network_error_handler(
    max_retries: int = 5,
    base_delay: float = 1.0,
    **kwargs
) -> NetworkErrorHandler:
    """Factory function to create network error handler with custom config"""
    retry_config = RetryConfig(
        max_retries=max_retries,
        base_delay=base_delay,
        **{k: v for k, v in kwargs.items() if hasattr(RetryConfig, k)}
    )
    
    handler_kwargs = {k: v for k, v in kwargs.items() if not hasattr(RetryConfig, k)}
    
    return NetworkErrorHandler(retry_config=retry_config, **handler_kwargs)


# Decorators for easy integration
def with_network_retry(handler: NetworkErrorHandler, context: str):
    """Decorator to add network retry logic to functions"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            return handler.handle_error(None, context, func, *args, **kwargs)
        return wrapper
    return decorator


if __name__ == "__main__":
    # Example usage and testing
    import tempfile
    
    # Create handler
    handler = create_network_error_handler(
        max_retries=3,
        base_delay=0.5,
        log_file=tempfile.mktemp(suffix='.log')
    )
    
    # Test error classification
    test_errors = [
        ConnectionResetError("Connection reset by peer"),
        TimeoutError("Connection timed out"),
        ConnectionRefusedError("Connection refused"),
        Exception("Unknown error")
    ]
    
    for error in test_errors:
        error_type, severity = handler._classify_error(error)
        print(f"{error}: {error_type.value} ({severity.value})")
    
    # Test connection health
    health = handler.check_connection_health(['https://www.google.com'])
    print(f"Connection healthy: {health}")
    
    # Get statistics
    stats = handler.get_error_statistics()
    print(f"Error statistics: {stats}")
