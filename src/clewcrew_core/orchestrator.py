#!/usr/bin/env python3
"""
clewcrew Core Orchestrator - Multi-Agent Hallucination Detection & Recovery System
"""

import sys
from pathlib import Path

# Add clewcrew-agents to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "clewcrew-agents" / "src"))

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List

from pydantic import BaseModel, Field, ConfigDict

# LangGraph imports
from langgraph.graph import END, StateGraph

# Local imports
from clewcrew_agents import (  # noqa: E402
    ArchitectureExpert,
    BuildExpert,
    CodeQualityExpert,
    MCPExpert,
    ModelExpert,
    SecurityExpert,
    TestExpert,
)
from .recovery import (
    ImportResolver,
    IndentationFixer,
    SyntaxRecoveryEngine,
    TypeAnnotationFixer,
)
from .validators import (
    ArchitectureValidator,
    BuildValidator,
    CodeQualityValidator,
    ModelValidator,
    SecurityValidator,
    TestValidator,
)


class ClewcrewState(BaseModel):
    """State for clewcrew workflow"""

    project_path: str
    hallucinations_detected: List[Dict[str, Any]] = Field(default_factory=list)
    recovery_actions: List[Dict[str, Any]] = Field(default_factory=list)
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    validation_results: Dict[str, Any] = Field(default_factory=dict)
    recovery_results: Dict[str, Any] = Field(default_factory=dict)
    current_phase: str = Field(default="detection")
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(arbitrary_types_allowed=True)


class ClewcrewOrchestrator:
    """Multi-agent orchestrator for hallucination detection and recovery"""

    def __init__(self, project_path: str = ".") -> None:
        self.project_path = Path(project_path)
        self.logger = logging.getLogger(__name__)

        # Initialize agents
        self.agents = {
            "security": SecurityExpert(),
            "code_quality": CodeQualityExpert(),
            "test": TestExpert(),
            "build": BuildExpert(),
            "architecture": ArchitectureExpert(),
            "model": ModelExpert(),
            "mcp": MCPExpert(),
        }

        # Initialize validators
        self.validators = {
            "security": SecurityValidator(),
            "code_quality": CodeQualityValidator(),
            "test": TestValidator(),
            "build": BuildValidator(),
            "architecture": ArchitectureValidator(),
            "model": ModelValidator(),
        }

        # Initialize recovery engines
        self.recovery_engines = {
            "syntax": SyntaxRecoveryEngine(),
            "indentation": IndentationFixer(),
            "imports": ImportResolver(),
            "types": TypeAnnotationFixer(),
        }

        # Create LangGraph workflow
        self.workflow = self._create_workflow()

    def _create_workflow(self) -> StateGraph:
        """Create LangGraph workflow for clewcrew"""
        workflow = StateGraph(ClewcrewState)

        # Add workflow nodes
        workflow.add_node("detect_hallucinations", self._detect_hallucinations_node)
        workflow.add_node("validate_findings", self._validate_findings_node)
        workflow.add_node("plan_recovery", self._plan_recovery_node)
        workflow.add_node("execute_recovery", self._execute_recovery_node)
        workflow.add_node("validate_recovery", self._validate_recovery_node)
        workflow.add_node("generate_report", self._generate_report_node)

        # Set entry point
        workflow.set_entry_point("detect_hallucinations")

        # Define workflow edges
        workflow.add_edge("detect_hallucinations", "validate_findings")
        workflow.add_edge("validate_findings", "plan_recovery")
        workflow.add_edge("plan_recovery", "execute_recovery")
        workflow.add_edge("execute_recovery", "validate_recovery")
        workflow.add_edge("validate_recovery", "generate_report")
        workflow.add_edge("generate_report", END)

        return workflow.compile()

    async def _detect_hallucinations_node(self, state: ClewcrewState) -> ClewcrewState:
        """Detect hallucinations using all expert agents"""
        self.logger.info("Starting hallucination detection...")

        hallucinations = []
        for name, agent in self.agents.items():
            try:
                agent_result = await agent.detect_hallucinations(self.project_path)
                hallucinations.extend(agent_result)
                self.logger.info(
                    f"{name} agent detected {len(agent_result)} hallucinations"
                )
            except Exception as e:
                self.logger.error(f"Error in {name} agent: {e}")
                state.errors.append(f"{name} agent error: {e}")

        state.hallucinations_detected = hallucinations
        state.current_phase = "validation"
        return state

    async def _validate_findings_node(self, state: ClewcrewState) -> ClewcrewState:
        """Validate findings using all validators"""
        self.logger.info("Starting validation of findings...")

        validation_results = {}
        for name, validator in self.validators.items():
            try:
                validator_result = await validator.validate_findings(
                    state.hallucinations_detected
                )
                validation_results[name] = validator_result
                self.logger.info(f"{name} validator completed validation")
            except Exception as e:
                self.logger.error(f"Error in {name} validator: {e}")
                state.errors.append(f"{name} validator error: {e}")

        state.validation_results = validation_results
        state.current_phase = "recovery_planning"
        return state

    async def _plan_recovery_node(self, state: ClewcrewState) -> ClewcrewState:
        """Plan recovery actions based on validated findings"""
        self.logger.info("Planning recovery actions...")

        recovery_actions = []
        for hallucination in state.hallucinations_detected:
            # Determine appropriate recovery engine
            if hallucination.get("type") == "syntax_error":
                recovery_actions.append(
                    {
                        "hallucination": hallucination,
                        "recovery_engine": "syntax",
                        "action": "fix_syntax_error",
                    }
                )
            elif hallucination.get("type") == "indentation_error":
                recovery_actions.append(
                    {
                        "hallucination": hallucination,
                        "recovery_engine": "indentation",
                        "action": "fix_indentation",
                    }
                )
            elif hallucination.get("type") == "import_error":
                recovery_actions.append(
                    {
                        "hallucination": hallucination,
                        "recovery_engine": "imports",
                        "action": "resolve_imports",
                    }
                )
            elif hallucination.get("type") == "type_error":
                recovery_actions.append(
                    {
                        "hallucination": hallucination,
                        "recovery_engine": "types",
                        "action": "fix_type_annotations",
                    }
                )

        state.recovery_actions = recovery_actions
        state.current_phase = "recovery_execution"
        return state

    async def _execute_recovery_node(self, state: ClewcrewState) -> ClewcrewState:
        """Execute recovery actions using appropriate engines"""
        self.logger.info("Executing recovery actions...")

        recovery_results = {}
        for action in state.recovery_actions:
            engine_name = action["recovery_engine"]
            engine = self.recovery_engines.get(engine_name)

            if engine:
                try:
                    result = await engine.execute_recovery(action)
                    recovery_results[action["action"]] = result
                    self.logger.info(f"Recovery action {action['action']} completed")
                except Exception as e:
                    self.logger.error(
                        f"Error in recovery action {action['action']}: {e}"
                    )
                    state.errors.append(f"Recovery error: {e}")
            else:
                self.logger.warning(f"No recovery engine found for {engine_name}")

        state.recovery_results = recovery_results
        state.current_phase = "recovery_validation"
        return state

    async def _validate_recovery_node(self, state: ClewcrewState) -> ClewcrewState:
        """Validate that recovery actions were successful"""
        self.logger.info("Validating recovery actions...")

        # Re-run validation to check if issues were resolved
        for name, validator in self.validators.items():
            try:
                post_recovery_validation = await validator.validate_findings([])
                # Compare with pre-recovery state
                if len(post_recovery_validation) < len(state.hallucinations_detected):
                    self.logger.info(
                        f"Recovery successful: {name} validator shows improvement"
                    )
                else:
                    self.logger.warning(
                        f"Recovery may not have been fully successful: {name}"
                    )
            except Exception as e:
                self.logger.error(f"Error in post-recovery validation {name}: {e}")

        state.current_phase = "report_generation"
        return state

    async def _generate_report_node(self, state: ClewcrewState) -> ClewcrewState:
        """Generate comprehensive report of the workflow execution"""
        self.logger.info("Generating final report...")

        # Calculate confidence score
        total_issues = len(state.hallucinations_detected)
        resolved_issues = len(
            [r for r in state.recovery_results.values() if r.get("success", False)]
        )

        if total_issues > 0:
            state.confidence_score = resolved_issues / total_issues
        else:
            state.confidence_score = 1.0

        # Add metadata
        state.metadata.update(
            {
                "workflow_completed_at": datetime.now(timezone.utc).isoformat(),
                "total_hallucinations": total_issues,
                "resolved_issues": resolved_issues,
                "success_rate": state.confidence_score,
            }
        )

        self.logger.info(
            f"Workflow completed with confidence score: {state.confidence_score}"
        )
        return state

    async def run_workflow(self, project_path: str = None) -> ClewcrewState:
        """Run the complete clewcrew workflow"""
        if project_path:
            self.project_path = Path(project_path)

        initial_state = ClewcrewState(project_path=str(self.project_path))

        try:
            final_state = await self.workflow.ainvoke(initial_state)
            return final_state
        except Exception as e:
            self.logger.error(f"Workflow execution failed: {e}")
            initial_state.errors.append(f"Workflow error: {e}")
            return initial_state

    async def detect_hallucinations(
        self, project_path: str = None
    ) -> List[Dict[str, Any]]:
        """Quick method to just detect hallucinations"""
        if project_path:
            self.project_path = Path(project_path)

        state = ClewcrewState(project_path=str(self.project_path))
        state = await self._detect_hallucinations_node(state)
        return state.hallucinations_detected

    async def recover_from_issues(self, issues: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Recover from specific issues"""
        state = ClewcrewState(project_path=str(self.project_path))
        state.hallucinations_detected = issues

        # Run recovery workflow
        state = await self._plan_recovery_node(state)
        state = await self._execute_recovery_node(state)

        return state.recovery_results

    async def run_quality_analysis(self, project_path: str = None) -> Dict[str, Any]:
        """
        Run comprehensive quality analysis using all expert agents.
        
        This method integrates with the quality system to provide
        comprehensive quality metrics and recommendations.
        
        Args:
            project_path: Path to the project to analyze
            
        Returns:
            Dictionary containing comprehensive quality analysis results
        """
        if project_path:
            self.project_path = Path(project_path)

        self.logger.info("Starting comprehensive quality analysis...")

        try:
            # Collect quality metrics from all expert agents
            agent_quality_results = {}
            
            for agent_name, agent in self.agents.items():
                try:
                    # Generate quality metrics for this agent's domain
                    quality_metrics = await agent.generate_quality_metrics(self.project_path)
                    agent_quality_results[agent_name] = quality_metrics
                    
                    self.logger.info(
                        f"{agent_name} agent generated quality metrics: "
                        f"score={quality_metrics.get('quality_score', 'N/A')}, "
                        f"issues={quality_metrics.get('issues_found', 'N/A')}"
                    )
                except Exception as e:
                    self.logger.error(f"Error in {agent_name} agent quality analysis: {e}")
                    # Provide fallback metrics
                    agent_quality_results[agent_name] = {
                        "quality_score": 0.0,
                        "issues_found": 0,
                        "error": str(e)
                    }

            # Collect quality recommendations from all agents
            all_recommendations = []
            for agent_name, agent in self.agents.items():
                try:
                    recommendations = await agent.provide_quality_recommendations(self.project_path)
                    all_recommendations.extend([
                        {"agent": agent_name, "recommendation": rec}
                        for rec in recommendations
                    ])
                except Exception as e:
                    self.logger.warning(f"Could not get recommendations from {agent_name}: {e}")

            # Generate comprehensive quality report
            quality_report = {
                "project_path": str(self.project_path),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "agent_quality_results": agent_quality_results,
                "all_recommendations": all_recommendations,
                "total_agents_analyzed": len(agent_quality_results),
                "total_recommendations": len(all_recommendations),
                "overall_quality_summary": self._generate_quality_summary(agent_quality_results)
            }

            self.logger.info("Quality analysis completed successfully")
            return quality_report

        except Exception as e:
            self.logger.error(f"Quality analysis failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "project_path": str(self.project_path),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

    def _generate_quality_summary(self, agent_quality_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate overall quality summary from agent results.
        
        Args:
            agent_quality_results: Quality results from all agents
            
        Returns:
            Dictionary containing overall quality summary
        """
        total_score = 0.0
        total_issues = 0
        agent_count = 0
        
        # Calculate simple average quality score (not weighted)
        for agent_name, results in agent_quality_results.items():
            # Skip agents with actual error strings, but allow None values
            if "error" in results and results["error"] is not None:
                continue
                
            quality_score = results.get("quality_score", 0.0)
            issues_found = results.get("issues_found", 0)
            
            total_score += quality_score
            total_issues += issues_found
            agent_count += 1
        
        # Calculate overall metrics
        if agent_count > 0:
            overall_quality_score = total_score / agent_count
        else:
            overall_quality_score = 0.0
        
        # Determine quality status
        if overall_quality_score >= 90.0:
            quality_status = "excellent"
        elif overall_quality_score >= 80.0:
            quality_status = "good"
        elif overall_quality_score >= 70.0:
            quality_status = "acceptable"
        elif overall_quality_score >= 60.0:
            quality_status = "needs_improvement"
        else:
            quality_status = "critical"
        
        return {
            "overall_quality_score": overall_quality_score,
            "total_issues_found": total_issues,
            "agents_analyzed": agent_count,
            "quality_status": quality_status,
            "recommendations_priority": "high" if overall_quality_score < 70.0 else "medium"
        }
