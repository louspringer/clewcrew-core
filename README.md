# 🧠 clewcrew-core

**Core orchestration and workflow management for the clewcrew portfolio.**

clewcrew-core provides the central workflow orchestration system that coordinates all expert agents, validators, and recovery engines to detect and resolve AI hallucinations in code.

## 🚀 Quick Start

### Installation

```bash
# Install with pip
pip install clewcrew-core

# Install with UV
uv add clewcrew-core
```

### Basic Usage

```python
from clewcrew_core import ClewcrewOrchestrator

# Initialize the orchestrator
orchestrator = ClewcrewOrchestrator("/path/to/project")

# Detect hallucinations
hallucinations = await orchestrator.detect_hallucinations()

# Run complete workflow
final_state = await orchestrator.run_workflow()

# Recover from specific issues
recovery_results = await orchestrator.recover_from_issues(hallucinations)
```

## 🏗️ Architecture

### Core Components

- **ClewcrewOrchestrator**: Main orchestrator class that manages the workflow
- **ClewcrewState**: State management for workflow execution
- **LangGraph Integration**: Built on LangGraph for robust workflow orchestration

### Workflow Stages

1. **Detection**: Expert agents detect hallucinations in code
2. **Validation**: Validators confirm and prioritize findings
3. **Recovery Planning**: System plans appropriate recovery actions
4. **Recovery Execution**: Recovery engines fix identified issues
5. **Recovery Validation**: System validates that fixes were successful
6. **Report Generation**: Comprehensive report of the workflow execution

## 🔧 Dependencies

- **clewcrew-common**: Shared utilities and patterns
- **clewcrew-framework**: Base classes and abstractions
- **langgraph**: Workflow orchestration
- **langchain**: AI agent framework
- **pydantic**: Data validation and settings management

## 📚 Documentation

- [API Reference](https://github.com/louspringer/clewcrew-core#readme)
- [Workflow Guide](https://github.com/louspringer/clewcrew-core#workflow-guide)
- [Examples](https://github.com/louspringer/clewcrew-core#examples)

## 🧪 Testing

```bash
# Run tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src/clewcrew_core --cov-report=html
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](https://github.com/louspringer/clewcrew-core/blob/main/CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Ready to orchestrate the clewcrew revolution!** 🧠✨




