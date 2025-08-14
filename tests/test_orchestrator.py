"""
Tests for the clewcrew-core orchestrator.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, AsyncMock

from clewcrew_core.orchestrator import ClewcrewOrchestrator, ClewcrewState


class TestClewcrewState:
    """Test the ClewcrewState model."""

    def test_state_creation(self):
        """Test creating a clewcrew state."""
        state = ClewcrewState(project_path="/test/project")
        assert state.project_path == "/test/project"
        assert state.hallucinations_detected == []
        assert state.recovery_actions == []
        assert state.confidence_score == 0.0
        assert state.current_phase == "detection"

    def test_state_defaults(self):
        """Test state default values."""
        state = ClewcrewState(project_path="/test/project")
        assert state.errors == []
        assert state.warnings == []
        assert state.metadata == {}


class TestClewcrewOrchestrator:
    """Test the ClewcrewOrchestrator class."""

    @pytest.fixture
    def orchestrator(self):
        """Create a test orchestrator."""
        return ClewcrewOrchestrator("/test/project")

    def test_orchestrator_initialization(self, orchestrator):
        """Test orchestrator initialization."""
        assert orchestrator.project_path == Path("/test/project")
        assert "security" in orchestrator.agents
        assert "code_quality" in orchestrator.agents
        assert "test" in orchestrator.agents
        assert "build" in orchestrator.agents
        assert "architecture" in orchestrator.agents
        assert "model" in orchestrator.agents
        assert "mcp" in orchestrator.agents

    def test_orchestrator_validators(self, orchestrator):
        """Test that validators are properly initialized."""
        assert "security" in orchestrator.validators
        assert "code_quality" in orchestrator.validators
        assert "test" in orchestrator.validators
        assert "build" in orchestrator.validators
        assert "architecture" in orchestrator.validators
        assert "model" in orchestrator.validators

    def test_orchestrator_recovery_engines(self, orchestrator):
        """Test that recovery engines are properly initialized."""
        assert "syntax" in orchestrator.recovery_engines
        assert "indentation" in orchestrator.recovery_engines
        assert "imports" in orchestrator.recovery_engines
        assert "types" in orchestrator.recovery_engines

    def test_workflow_creation(self, orchestrator):
        """Test that the workflow is properly created."""
        assert orchestrator.workflow is not None
        # The workflow should be a compiled LangGraph workflow
        assert hasattr(orchestrator.workflow, "ainvoke")


@pytest.mark.asyncio
class TestClewcrewOrchestratorAsync:
    """Test async functionality of the ClewcrewOrchestrator."""

    @pytest.fixture
    def mock_orchestrator(self):
        """Create a mock orchestrator for testing."""
        orchestrator = ClewcrewOrchestrator("/test/project")

        # Mock the agents to avoid actual execution
        for name in orchestrator.agents:
            orchestrator.agents[name] = Mock()
            orchestrator.agents[name].detect_hallucinations = AsyncMock(return_value=[])

        # Mock the validators
        for name in orchestrator.validators:
            orchestrator.validators[name] = Mock()
            orchestrator.validators[name].validate_findings = AsyncMock(return_value=[])

        # Mock the recovery engines
        for name in orchestrator.recovery_engines:
            orchestrator.recovery_engines[name] = Mock()
            orchestrator.recovery_engines[name].execute_recovery = AsyncMock(
                return_value={"success": True}
            )

        return orchestrator

    async def test_detect_hallucinations(self, mock_orchestrator):
        """Test hallucination detection."""
        result = await mock_orchestrator.detect_hallucinations("/test/project")
        assert isinstance(result, list)
        assert len(result) == 0  # Mock returns empty list

    async def test_recover_from_issues(self, mock_orchestrator):
        """Test recovery from issues."""
        test_issues = [
            {"type": "syntax_error", "file": "test.py", "line": 10},
            {"type": "indentation_error", "file": "test.py", "line": 15},
        ]

        result = await mock_orchestrator.recover_from_issues(test_issues)
        assert isinstance(result, dict)
        assert "fix_syntax_error" in result
        assert "fix_indentation" in result


if __name__ == "__main__":
    pytest.main([__file__])
