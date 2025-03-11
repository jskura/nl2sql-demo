# Enhanced Logging System for NL2SQL Demo

This document explains the enhanced logging system implemented in the NL2SQL demo application.

## Overview

The application uses a comprehensive logging system that provides:

1. **Structured Logging**: Logs are formatted consistently with timestamps, log levels, and contextual information.
2. **Multiple Output Destinations**: Logs can be sent to the console, files, and specialized error logs.
3. **JSON Logging**: For machine-readable logs that can be easily parsed and analyzed.
4. **Request Tracking**: Each HTTP request is assigned a unique ID that is included in all related log entries.
5. **Performance Monitoring**: Automatic logging of function execution times and operation durations.

## Configuration

Logging is configured through environment variables in the `.env` file:

```
# Logging configuration
# Log level: DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_LEVEL=DEBUG
# Log format: standard, detailed, json
LOG_FORMAT=standard
# Enable file logging
LOG_TO_FILE=true
# Maximum log file size in bytes (default: 10MB)
LOG_FILE_MAX_SIZE=10485760
# Number of backup log files to keep
LOG_FILE_BACKUP_COUNT=10
# Enable JSON logging
LOG_JSON_ENABLED=true
```

## Log Files

When file logging is enabled, the following log files are created in the `logs` directory:

- `app_YYYY-MM-DD.log`: Contains all log messages at the configured level.
- `error_YYYY-MM-DD.log`: Contains only ERROR and CRITICAL level messages.
- `json_YYYY-MM-DD.log`: Contains logs in JSON format for machine processing.

Log files are automatically rotated when they reach the configured maximum size.

## Log Levels

The application uses standard Python logging levels:

- **DEBUG**: Detailed information, typically useful only for diagnosing problems.
- **INFO**: Confirmation that things are working as expected.
- **WARNING**: An indication that something unexpected happened, or may happen in the future.
- **ERROR**: Due to a more serious problem, the software has not been able to perform a function.
- **CRITICAL**: A serious error, indicating that the program itself may be unable to continue running.

## Request Tracking

Each HTTP request is assigned a unique ID (UUID) that is:

1. Added to all log entries related to that request
2. Included in the response headers as `X-Request-ID`
3. Available in the request state as `request.state.request_id`

This makes it easy to trace all log entries related to a specific request.

## Performance Monitoring

The application includes utilities for performance monitoring:

### Function Decorators

```python
from app.utils.performance import log_execution_time, async_log_execution_time

@log_execution_time
def my_function():
    # Function code here

@async_log_execution_time
async def my_async_function():
    # Async function code here
```

These decorators automatically log the execution time of functions.

### Context Manager

```python
from app.utils.performance import PerformanceTracker

with PerformanceTracker("operation_name"):
    # Code to be timed
```

The context manager logs the duration of the code block and includes the operation name in the log entry.

## JSON Logging

When JSON logging is enabled, log entries in the JSON log file are structured as follows:

```json
{
  "timestamp": "2023-06-01T12:34:56.789Z",
  "name": "app.services.nl2sql_service",
  "level": "INFO",
  "module": "nl2sql_service",
  "line": 123,
  "message": "Generated SQL query",
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "duration_ms": 234.56,
  "function": "generate_basic_sql"
}
```

This format makes it easy to parse and analyze logs using tools like ELK Stack, Splunk, or custom scripts.

## Middleware

The application includes a middleware that:

1. Generates a unique request ID for each request
2. Logs request details including method, path, and client IP
3. Measures and logs request duration
4. Adds the request ID to response headers
5. Logs detailed information about failed requests

## Best Practices

When adding new code to the application:

1. Get a logger for your module:
   ```python
   import logging
   logger = logging.getLogger(__name__)
   ```

2. Use appropriate log levels:
   ```python
   logger.debug("Detailed debug information")
   logger.info("Something noteworthy happened")
   logger.warning("Something unexpected happened")
   logger.error("An error occurred")
   logger.critical("A critical error occurred")
   ```

3. Include contextual information:
   ```python
   logger.info(
       "User action completed",
       extra={"user_id": user_id, "action": action_name}
   )
   ```

4. Use performance tracking for expensive operations:
   ```python
   with PerformanceTracker("database_query"):
       results = db.execute_query(...)
   ```

5. Use the function decorators for timing function execution:
   ```python
   @log_execution_time
   def process_data(data):
       # Processing code
   ``` 