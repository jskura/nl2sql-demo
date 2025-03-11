import os
import logging
import logging.config
from pathlib import Path
import json
from datetime import datetime

from app.core.config import (
    LOG_LEVEL, 
    LOG_FORMAT, 
    LOG_TO_FILE, 
    LOG_FILE_MAX_SIZE, 
    LOG_FILE_BACKUP_COUNT,
    LOG_JSON_ENABLED,
    DEBUG
)

# Create logs directory if it doesn't exist
logs_dir = Path(__file__).parent.parent.parent / 'logs'
logs_dir.mkdir(exist_ok=True)

# Get current date for log file naming
current_date = datetime.now().strftime('%Y-%m-%d')

# Define logging configuration
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        },
        'detailed': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(pathname)s:%(lineno)d - %(message)s'
        },
        'json': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            '()': 'app.core.logging_config.JsonFormatter',
        }
    },
    'handlers': {
        'console': {
            'level': 'DEBUG' if DEBUG else LOG_LEVEL,
            'formatter': LOG_FORMAT if LOG_FORMAT in ['standard', 'detailed'] else 'standard',
            'class': 'logging.StreamHandler',
            'stream': 'ext://sys.stdout',
        },
    },
    'loggers': {
        '': {  # root logger
            'handlers': ['console'],
            'level': LOG_LEVEL,
            'propagate': True
        },
        'app': {
            'handlers': ['console'],
            'level': 'DEBUG' if DEBUG else LOG_LEVEL,
            'propagate': False
        },
        'app.services': {
            'handlers': ['console'],
            'level': 'DEBUG' if DEBUG else LOG_LEVEL,
            'propagate': False
        },
        'app.api': {
            'handlers': ['console'],
            'level': LOG_LEVEL,
            'propagate': False
        },
        'uvicorn': {
            'handlers': ['console'],
            'level': LOG_LEVEL,
            'propagate': False
        },
        'uvicorn.error': {
            'handlers': ['console'],
            'level': LOG_LEVEL,
            'propagate': False
        }
    }
}

# Add file handlers if LOG_TO_FILE is enabled
if LOG_TO_FILE:
    LOGGING_CONFIG['handlers']['file'] = {
        'level': 'DEBUG',
        'formatter': 'detailed',
        'class': 'logging.handlers.RotatingFileHandler',
        'filename': str(logs_dir / f'app_{current_date}.log'),
        'maxBytes': LOG_FILE_MAX_SIZE,
        'backupCount': LOG_FILE_BACKUP_COUNT,
        'encoding': 'utf8'
    }
    
    LOGGING_CONFIG['handlers']['error_file'] = {
        'level': 'ERROR',
        'formatter': 'detailed',
        'class': 'logging.handlers.RotatingFileHandler',
        'filename': str(logs_dir / f'error_{current_date}.log'),
        'maxBytes': LOG_FILE_MAX_SIZE,
        'backupCount': LOG_FILE_BACKUP_COUNT,
        'encoding': 'utf8'
    }
    
    # Add file handlers to loggers
    for logger_name in LOGGING_CONFIG['loggers']:
        LOGGING_CONFIG['loggers'][logger_name]['handlers'].extend(['file', 'error_file'])

# Add JSON logging if enabled
if LOG_JSON_ENABLED and LOG_TO_FILE:
    LOGGING_CONFIG['handlers']['json_file'] = {
        'level': LOG_LEVEL,
        'formatter': 'json',
        'class': 'logging.handlers.RotatingFileHandler',
        'filename': str(logs_dir / f'json_{current_date}.log'),
        'maxBytes': LOG_FILE_MAX_SIZE,
        'backupCount': LOG_FILE_BACKUP_COUNT,
        'encoding': 'utf8'
    }
    
    # Add json_file handler to app loggers
    for logger_name in ['app', 'app.services']:
        LOGGING_CONFIG['loggers'][logger_name]['handlers'].append('json_file')

class JsonFormatter(logging.Formatter):
    """
    Formatter that outputs JSON strings after parsing the log record.
    """
    def format(self, record):
        logobj = {}
        logobj['timestamp'] = datetime.utcnow().isoformat()
        logobj['name'] = record.name
        logobj['level'] = record.levelname
        logobj['module'] = record.module
        logobj['line'] = record.lineno
        logobj['message'] = record.getMessage()
        
        if hasattr(record, 'request_id'):
            logobj['request_id'] = record.request_id
            
        if hasattr(record, 'user'):
            logobj['user'] = record.user
            
        if record.exc_info:
            logobj['exception'] = self.formatException(record.exc_info)
            
        if hasattr(record, 'duration'):
            logobj['duration_ms'] = record.duration
            
        # Add any extra attributes
        for key, value in record.__dict__.items():
            if key not in ['args', 'asctime', 'created', 'exc_info', 'exc_text', 'filename',
                          'funcName', 'id', 'levelname', 'levelno', 'lineno', 'module',
                          'msecs', 'message', 'msg', 'name', 'pathname', 'process',
                          'processName', 'relativeCreated', 'stack_info', 'thread', 'threadName']:
                if not key.startswith('_'):
                    logobj[key] = value
            
        return json.dumps(logobj)

def configure_logging():
    """Configure logging based on the defined configuration"""
    logging.config.dictConfig(LOGGING_CONFIG)
    
    # Create a logger for this module
    logger = logging.getLogger(__name__)
    logger.debug("Logging configured successfully")
    
    return logger

# Helper function to add request_id to log record
def get_request_logger(request_id=None):
    """Get a logger with request_id information"""
    logger = logging.getLogger('app')
    
    if request_id:
        old_factory = logging.getLogRecordFactory()
        
        def record_factory(*args, **kwargs):
            record = old_factory(*args, **kwargs)
            record.request_id = request_id
            return record
            
        logging.setLogRecordFactory(record_factory)
        
    return logger 