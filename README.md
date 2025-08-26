# Multimodal planning in robotics

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

**Note**: The final version of the system is in the `sense-plan-act/` and `ros2-ur5/` folders. Other folders represent early prototypes and alternative approaches. A polished version of these two will be released soon.

### 🤖 Pure Neural Approaches

#### [`openai/`](openai/)
**Single LLM Agent System**
- Direct GPT integration with MiniGrid environments
- Text-based observation processing
- Structured output parsing for actions
- Comprehensive logging and analysis


#### [`openai_dspy/`](openai_dspy/)
**Self-Optimizing LLM Agent**
- DSPy framework integration for automatic prompt optimization
- Human demonstration collection and learning
- Few-shot learning from successful patterns
- Optimal path simulation and comparison


### 👥 Multi-Agent Systems

#### [`crewai/`](crewai/)
**Collaborative Multi-Agent Framework**
- CrewAI-based agent orchestration
- Specialized agent roles (Navigator, Observer, Decision-maker)
- Task delegation and coordination
- Distributed problem-solving


### 🔧 Pure Symbolic

#### [`pddl-llm/`](pddl-llm/)
**Classical AI Planning**
- Pure PDDL domain definitions
- Traditional planning algorithms
- LLM-assisted domain and problem generation
- Plan validation and optimization


### 🧠 Hybrid Symbolic-Neural

#### [`sense-plan-act/`](sense-plan-act/)
**Curriculum-Based Learning System**
- PDDL planning with LLM assistance
- Automatic Python code generation
- Progressive difficulty through BabyAI/MiniGrid tasks
- Dynamic code injection and hot reloading


#### [`pddl-code-minigrid/`](pddl-code-minigrid/)
**PDDL + Code Generation**
- Classical planning with modern LLM code generation
- Symbolic reasoning for robust problem-solving
- Failure-driven replanning and code regeneration
- Similar to sense-plan-act but LLM does blind planning


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

<!-- 
## 📊 Comparison Framework

All approaches are evaluated on:
- **Task Success Rate**: Completion percentage across environments
- **Sample Efficiency**: Steps/episodes needed to learn
- **Generalization**: Performance on unseen tasks
- **Robustness**: Handling of edge cases and failures
- **Interpretability**: Understanding of decision-making process
- **Real-world Transfer**: Simulation-to-reality gap -->

## 🔬 Research Insights

### Key Findings
1. **Pure Neural**: Fast prototyping, but limited systematic reasoning
2. **Multi-Agent**: Better task decomposition, improved robustness
3. **Pure Symbolic**: Most interpretable, but requires extensive domain knowledge
4. **Hybrid**: Best of both worlds - systematic planning + flexible execution
5. **Real Robot**: Validates simulation findings, reveals practical constraints

<!-- 
## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details. -->


## 📚 Citation

If you use this work in your research, please cite:

```bibtex
@misc{pesjak2025multimodal,
  title={Multimodal planning in robotics},
  author={Drejc Pesjak},
  year={2025},
  url={https://github.com/DrejcPesjak/minigrid-crewai}
}
```
