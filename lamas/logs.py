#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from datetime import datetime
from pathlib import Path

from loguru import logger as _logger

from lamas.const import METAGPT_ROOT

_print_level = "INFO"


def define_log_level(print_level="INFO", logfile_level="DEBUG", name: str = None, dataset: str = None, flags: str = None, weight: str = None):
    """Adjust the log level to above level

    Args:
        print_level: Console log level
        logfile_level: File log level
        name: Optional prefix name
        dataset: Optional dataset name (e.g., GSM8K, MATH, HumanEval)
        flags: Optional experiment flags (e.g., sequential_cp_no_norm)
        weight: Optional weight value for log naming (e.g., 0.001)
    """
    global _print_level
    _print_level = print_level

    current_date = datetime.now()
    formatted_date = current_date.strftime("%Y%m%d_%H%M%S")

    # Build log name with dataset, flags, weight, and timestamp
    name_parts = []
    if dataset:
        name_parts.append(dataset)
    if flags:
        name_parts.append(flags)
    if weight is not None:
        # Format weight to remove unnecessary decimals (e.g., 0.001 -> w0_001, 0 -> w0_000)
        weight_str = str(weight).replace('.', '_')
        if weight_str == '0':
            weight_str = '0_000'
        name_parts.append(f"w{weight_str}")
    if name:
        name_parts.append(name)

    if name_parts:
        log_name = f"{'_'.join(name_parts)}_{formatted_date}"
    else:
        log_name = formatted_date

    # Use absolute path to ensure logs are saved correctly
    try:
        metagpt_root = Path(METAGPT_ROOT).resolve()
        log_file_path = metagpt_root / f"logs/{log_name}.txt"
    except Exception as e:
        # If METAGPT_ROOT resolution fails, use current working directory
        print(f"WARNING: Failed to resolve METAGPT_ROOT ({METAGPT_ROOT}): {e}", file=sys.stderr)
        metagpt_root = Path.cwd()
        log_file_path = metagpt_root / f"logs/{log_name}.txt"
    
    # Ensure logs directory exists with proper error handling
    try:
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
    except (OSError, PermissionError) as e:
        # Fallback to current working directory if METAGPT_ROOT fails
        fallback_path = Path.cwd() / f"logs/{log_name}.txt"
        try:
            fallback_path.parent.mkdir(parents=True, exist_ok=True)
            log_file_path = fallback_path
            print(f"WARNING: Failed to create log directory at {metagpt_root}/logs, using fallback: {log_file_path}", file=sys.stderr)
        except Exception as e2:
            print(f"ERROR: Failed to create fallback log directory: {e2}", file=sys.stderr)
            # If all else fails, just use stderr
            _logger.remove()
            _logger.add(sys.stderr, level=print_level)
            return _logger
    
    # Write header at the beginning of the log file
    try:
        with open(log_file_path, "w", encoding="utf-8") as f:
            f.write(f"Log started at: {current_date.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Log file path: {log_file_path}\n")
            f.write(f"METAGPT_ROOT: {METAGPT_ROOT}\n")
            f.write("-" * 80 + "\n\n")
    except (OSError, PermissionError) as e:
        # If we can't write to the log file, at least print the error
        print(f"ERROR: Failed to write log file at {log_file_path}: {e}", file=sys.stderr)
        # Try fallback to current directory
        fallback_path = Path.cwd() / f"logs/{log_name}.txt"
        fallback_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(fallback_path, "w", encoding="utf-8") as f:
                f.write(f"Log started at: {current_date.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Log file path: {fallback_path} (fallback)\n")
                f.write(f"METAGPT_ROOT: {METAGPT_ROOT}\n")
                f.write("-" * 80 + "\n\n")
            log_file_path = fallback_path
            print(f"Using fallback log file: {log_file_path}", file=sys.stderr)
        except Exception as e2:
            print(f"ERROR: Failed to write fallback log file: {e2}", file=sys.stderr)
            # If all else fails, just use stderr
            _logger.remove()
            _logger.add(sys.stderr, level=print_level)
            return _logger

    _logger.remove()
    # Add stderr handler with timestamp format
    _logger.add(
        sys.stderr,
        level=print_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>"
    )
    try:
        # Add file handler with detailed timestamp format
        _logger.add(
            log_file_path,
            level=logfile_level,
            mode="a",  # Append mode to preserve header
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}"
        )
        print(f"Logging to: {log_file_path}", file=sys.stderr)
    except Exception as e:
        print(f"WARNING: Failed to add file handler for {log_file_path}: {e}", file=sys.stderr)
        # Continue with just stderr logging

    return _logger


logger = define_log_level()


def log_llm_stream(msg):
    _llm_stream_log(msg)


def set_llm_stream_logfunc(func):
    global _llm_stream_log
    _llm_stream_log = func


def _llm_stream_log(msg):
    if _print_level in ["INFO"]:
        print(msg, end="")
