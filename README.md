# MiniGrid CrewAI: Multi-Paradigm AI Agent Research Platform

A comprehensive research platform exploring different approaches to embodied AI and robotic manipulation, from simulation to real-world deployment. This project systematically compares neural, symbolic, and hybrid approaches to AI agent development across multiple frameworks.

## 🎯 Project Overview

This workspace implements and compares various AI paradigms for solving sequential decision-making tasks, progressing from simple grid environments to real robotic manipulation. The core research question: **"What's the most effective way to create AI agents that can reason about and manipulate their environment?"**

## 🏗️ System Architecture

The project follows a **Sense-Plan-Code-Act** paradigm across all implementations:

1. **Sense**: Environment perception and state estimation
2. **Plan**: High-level task planning (PDDL, LLM, or hybrid)
3. **Code**: Action implementation (direct execution or code generation)
4. **Act**: Physical execution and outcome evaluation

## 📁 Folder Structure & Approaches

### 🤖 Pure Neural Approaches

#### [`openai/`](openai/)
**Single LLM Agent System**
- Direct GPT integration with MiniGrid environments
- Text-based observation processing
- Structured output parsing for actions
- Comprehensive logging and analysis

```bash
cd openai && python main.py
```

#### [`openai_dspy/`](openai_dspy/)
**Self-Optimizing LLM Agent**
- DSPy framework integration for automatic prompt optimization
- Human demonstration collection and learning
- Few-shot learning from successful patterns
- Optimal path simulation and comparison

```bash
cd openai_dspy && python main.py
```

### 👥 Multi-Agent Systems

#### [`crewai/`](crewai/)
**Collaborative Multi-Agent Framework**
- CrewAI-based agent orchestration
- Specialized agent roles (Navigator, Observer, Decision-maker)
- Task delegation and coordination
- Distributed problem-solving

```bash
cd crewai && python -m src.crewai_gym.main
```

### 🧠 Hybrid Symbolic-Neural

#### [`sense-plan-act/`](sense-plan-act/)
**Curriculum-Based Learning System**
- PDDL planning with LLM assistance
- Automatic Python code generation
- Progressive difficulty through BabyAI/MiniGrid tasks
- Dynamic code injection and hot reloading

```bash
cd sense-plan-act && python main.py
```

#### [`pddl-code-minigrid/`](pddl-code-minigrid/)
**PDDL + Code Generation**
- Classical planning with modern LLM code generation
- Symbolic reasoning for robust problem-solving
- Failure-driven replanning and code regeneration

```bash
cd pddl-code-minigrid && python main.py
```

### 🔧 Pure Symbolic

#### [`pddl-llm/`](pddl-llm/)
**Classical AI Planning**
- Pure PDDL domain definitions
- Traditional planning algorithms
- LLM-assisted domain and problem generation
- Plan validation and optimization

### 🦾 Real-World Deployment

#### [`ros2-ur5/`](ros2-ur5/)
**Physical Robot Integration**
- Complete ROS2 Humble + Gazebo 11 simulation
- UR5 robotic arm with Robotiq 2F-85 gripper
- RGB-D camera integration
- MoveIt 2 motion planning
- Real-world validation of simulation strategies

```bash
# Setup (see ros2-ur5/ros-install_pretty2.sh for full installation)
cd ros2-ur5
source working_commands.sh
```

## 🚀 Quick Start

### Prerequisites
```bash
# Python dependencies
pip install crewai gymnasium minigrid openai dspy-ai unified-planning

# For ROS2 (Ubuntu 22.04)
# See ros2-ur5/ros-install_pretty2.sh for complete setup
```

### Running Different Approaches

1. **Start with simple LLM approach**:
```bash
cd openai && python main.py
```

2. **Try optimized learning**:
```bash
cd openai_dspy && python main.py
```

3. **Explore multi-agent collaboration**:
```bash
cd crewai && python -m src.crewai_gym.main
```

4. **Test hybrid planning**:
```bash
cd sense-plan-act && python main.py
```

5. **Deploy to real robot**:
```bash
cd ros2-ur5 && ./working_commands.sh
```

## 📊 Comparison Framework

All approaches are evaluated on:
- **Task Success Rate**: Completion percentage across environments
- **Sample Efficiency**: Steps/episodes needed to learn
- **Generalization**: Performance on unseen tasks
- **Robustness**: Handling of edge cases and failures
- **Interpretability**: Understanding of decision-making process
- **Real-world Transfer**: Simulation-to-reality gap

## 🔬 Research Insights

### Key Findings
1. **Pure Neural**: Fast prototyping, but limited systematic reasoning
2. **Multi-Agent**: Better task decomposition, improved robustness
3. **Hybrid**: Best of both worlds - systematic planning + flexible execution
4. **Pure Symbolic**: Most interpretable, but requires extensive domain knowledge
5. **Real Robot**: Validates simulation findings, reveals practical constraints

### Evolution Path
```
Simple LLM → Optimized LLM → Multi-Agent → Hybrid → Symbolic → Real Robot
   ↓              ↓              ↓           ↓         ↓         ↓
Learning    Self-Improvement  Collaboration Planning  Logic   Validation
```

## 🛠️ Development

### Adding New Approaches
1. Create new folder following naming convention
2. Implement Sense-Plan-Code-Act interface
3. Add evaluation metrics
4. Update this README

### Common Utilities
- Environment wrappers in each folder
- Shared evaluation metrics
- Logging and visualization tools
- Configuration management

## 📈 Future Directions

- **Foundation Models**: Integration with larger, more capable models
- **Multimodal**: Vision-language integration for richer perception
- **Transfer Learning**: Cross-domain knowledge transfer
- **Hierarchical Planning**: Multi-level abstraction and planning
- **Human-AI Collaboration**: Interactive learning and correction

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📚 Citation

If you use this work in your research, please cite:

```bibtex
@misc{minigrid-crewai-2025,
  title={MiniGrid CrewAI: Multi-Paradigm AI Agent Research Platform},
  author={Your Name},
  year={2025},
  url={https://github.com/your-username/minigrid-crewai}
}
```

---

**Note**: This is an active research project. Individual components may be in different stages of development. Check individual folder READMEs for specific setup instructions and current status.