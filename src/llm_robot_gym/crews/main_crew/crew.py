from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from .types import AgentAction

import os
os.environ["OPENAI_API_KEY"] = "NONE"

@CrewBase
class LlmRobotGymCrew:
	"""LlmRobotGym crew"""
	
	ollama = LLM(model="ollama/llama3.2", base_url="http://localhost:11434")

	@agent
	def brain_agent(self) -> Agent:
		return Agent(
			config=self.agents_config['brain_agent'],
			verbose=True,
			llm=self.ollama,
		)

	@task
	def main_task(self) -> Task:
		return Task(
			config=self.tasks_config['main_task'],
			output_pydantic=AgentAction
		)

	@crew
	def crew(self) -> Crew:
		"""Creates the LlmRobotGym crew"""
		return Crew(
			agents=self.agents,
			tasks=self.tasks,
			process=Process.sequential,
			# memory=True,
			verbose=True
		)