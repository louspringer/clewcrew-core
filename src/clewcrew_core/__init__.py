"""
clewcrew-core

Core orchestration and workflow management for the clewcrew portfolio.
This package provides the central workflow orchestration system that coordinates
all expert agents, validators, and recovery engines.
"""

__version__ = "0.1.0"
__author__ = "Lou Springer"
__email__ = "lou@example.com"

from .orchestrator import ClewcrewOrchestrator, ClewcrewState

__all__ = [
    "ClewcrewOrchestrator",
    "ClewcrewState",
]
