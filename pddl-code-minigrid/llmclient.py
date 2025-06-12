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
        API_KEY = os.environ.get("OPENAI_API_KEY")
        self.client = OpenAI(api_key=API_KEY)
        self.model_name = model_name #o1
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
            