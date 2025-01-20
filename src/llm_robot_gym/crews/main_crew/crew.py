from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai.memory.short_term.short_term_memory import ShortTermMemory
from crewai.memory.entity.entity_memory import EntityMemory
from crewai.memory.storage.rag_storage import RAGStorage
from .types import AgentAction
# LOAD ENVIRONMENT VARIABLES .env
from dotenv import load_dotenv
load_dotenv("/home/drew99/School/MastersT/crewai-gym/github/minigrid-crewai/src/.env")

# import os
# os.environ["OPENAI_API_KEY"] = "NONE"
# os.environ["GEMINI_API_KEY"] = "NONE"
# os.environ["CREWAI_STORAGE_DIR"] = "/home/drew99/School/MastersT/crewai-gym/llm_robot_gym/src/llm_robot_gym/storage"
# os.environ['LITELLM_LOG'] = 'DEBUG'

@CrewBase
class LlmRobotGymCrew:
	"""LlmRobotGym crew"""
	
	# ollama = LLM(model="ollama/llama3.2", base_url="http://localhost:11434")

	gemini = LLM(model="gemini/gemini-1.5-flash-latest")

	# embedder = {
    #     "provider": "ollama",
    #     "config": {
    #         "model": "nomic-embed-text"
    #     }
    # }
	# stm = ShortTermMemory(
	# 	embedder_config=embedder,
	# 	# storage=RAGStorage(
	# 	# 	type="short_term",
	# 	# )
	# )
	# stm.reset() # Reset the short-term memory from previous runs
	# em = EntityMemory(
	# 	embedder_config=embedder,
	# 	# storage=RAGStorage(
	# 	# 	type="entities",
	# 	# )
	# )

	@agent
	def brain_agent(self) -> Agent:
		return Agent(
			config=self.agents_config['brain_agent'],
			verbose=True,
			llm=self.gemini,
			# memory=True,
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
			# short_term_memory=self.stm,
			# entity_memory=self.em,
			# embedder=self.embedder,
			verbose=True
		)