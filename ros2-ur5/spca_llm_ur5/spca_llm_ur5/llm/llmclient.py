import os
from openai import OpenAI
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv() 

# class PDDLResponse(BaseModel):
#     domain: str
#     problem: str
# class CoderResponse(BaseModel):
#     code: str

class ChatGPTClient:
    def __init__(self, model_name: str, output_format: BaseModel):
        print(f"Using model: {model_name}")

        # ollama/llama4
        # openai/gpt-4o
        # gemini/gemini-2.5-flash

        if model_name.startswith("ollama"):
            self.client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
        
        elif model_name.startswith("openai"):
            API_KEY = os.environ.get("OPENAI_API_KEY")
            self.client = OpenAI(api_key=API_KEY)

        elif model_name.startswith("gemini"):
            API_KEY = os.environ.get("GEMINI_API_KEY")
            self.client = OpenAI(
                api_key=API_KEY,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
            )

        self.model_name = model_name.split("/")[-1]
        self.output_format = output_format

    def chat_completion(self, messages):
        # print(messages[-2:])
        if not self.model_name == "codex-mini-latest":
            result = self.client.beta.chat.completions.parse(
                model=self.model_name,
                messages=messages,
                response_format=self.output_format,
            )
            return result.choices[0].message.parsed
        else:
            # https://platform.openai.com/docs/guides/structured-outputs?api-mode=responses
            response = self.client.responses.parse(
                model=self.model_name,
                input=messages,
                text_format=self.output_format,
            )
            return response.output[-1].content[-1].parsed
    
    def image_description(self, messages) -> str:
        """
        Multimodal plain-text response for a single image prompt.
        `messages` is already a list of dicts (system+user with input_text + input_image).
        Returns plain text (no JSON).
        """
        resp = self.client.responses.create(
            model=self.model_name,
            input=messages,
        )
        return resp.output_text
