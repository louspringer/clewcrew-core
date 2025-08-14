"""
clewcrew Core - Multi-Agent Hallucination Detection & Recovery System

This package provides the core orchestration system for coordinating
multiple expert agents to detect and recover from hallucinations.
"""

from .orchestrator import ClewcrewOrchestrator, ClewcrewState
from .validators import (
    BaseValidator,
    SecurityValidator,
    CodeQualityValidator,
    TestValidator,
    BuildValidator,
    ArchitectureValidator,
    ModelValidator,
)
from .recovery import (
    BaseRecoveryEngine,
    SyntaxRecoveryEngine,
    IndentationFixer,
    ImportResolver,
    TypeAnnotationFixer,
)

__all__ = [
    "ClewcrewOrchestrator",
    "ClewcrewState",
    "BaseValidator",
    "SecurityValidator",
    "CodeQualityValidator",
    "TestValidator",
    "BuildValidator",
    "ArchitectureValidator",
    "ModelValidator",
    "BaseRecoveryEngine",
    "SyntaxRecoveryEngine",
    "IndentationFixer",
    "ImportResolver",
    "TypeAnnotationFixer",
]

__version__ = "0.1.0"
