from typing import (
    Callable,
    List,
)

import openai

import tiktoken
from llm.basellm import BaseLLM
from retry import retry
from dotenv import load_dotenv 
load_dotenv()
import os 



                      
                      
                    
# client.chat.completions.create(                       
                        
TEMPERATURE = 0.0
# endpoint = os.environ["OPENAI_API_BASE"]
# api_key = os.environ["OPENAI_API_KEY"]


class OpenAIChat(BaseLLM):
    """Wrapper around OpenAI Chat large language models."""

    def __init__(
        self,
        # openai_api_key: str,
        api_key: str = os.environ["OPENAI_API_KEY"],
        endpoint: str = os.environ["OPENAI_API_BASE"],
        api_version: str = os.environ["OPENAI_API_VERSION"],
        max_tokens: int = 1000,
        temperature: float = TEMPERATURE,
    ) -> None:
        # self.client = openai.AzureOpenAI(azure_endpoint=endpoint,
        #                     api_key=api_key,
        #                     api_version=api_version
        #                 )
        # openai.api_key = openai_api_key

        self.deployment = os.environ["OPENAI_DEPLOYMENT_NAME"]
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.client =  openai.AzureOpenAI(
                    azure_endpoint=endpoint,
                    api_key=api_key,
                    api_version=api_version
                )

    @retry(tries=3, delay=1)
    def generate(
        self,
        messages: List[str],
    ) -> str:
        # print("openai.__version__", openai.__version__)
        print("In generate")
        try:
            completions = self.client.chat.completions.create(
                            model=self.deployment,
                            messages=messages,
                            temperature=self.temperature,
                        )
            return completions.choices[0].message.content
        # catch context length / do not retry
        except openai.error.InvalidRequestError as e:
            return str(f"Error: {e}")
        # catch authorization errors / do not retry
        except openai.error.AuthenticationError as e:
            return "Error: The provided OpenAI API key is invalid"
        except Exception as e:
            print(f"Retrying LLM call {e}")
            raise Exception()

    async def generateStreaming(
        self,
        messages: List[str],
        onTokenCallback=Callable[[str], None],
    ) -> str:
        # print("openai.__version__", openai.__version__)
        print("In generateStreaming")
        print(messages)
        result = []
        # completions = openai.ChatCompletion.create(
        #     model=self.model,
        #     temperature=self.temperature,
        #     max_tokens=self.max_tokens,
        #     messages=messages,
        #     stream=True,
        # )
        completions = self.client.chat.completions.create(
                            model=self.deployment,
                            messages=messages,
                            temperature=self.temperature,
                            stream=True,
                        )
        result = []
        print(type(completions))
        print(completions)
        for message in completions:
            # Process the streamed messages or perform any other desired action
            print("message", message)
            print(message.choices[0].delta.content)
            delta = message.choices[0].delta
            print("result", result)
            if delta.content:
                result.append(delta.content)
            # print(delta)
            # if "content" in delta:
            #     print()
            #     result.append(delta["content"])
            # if len
            print("result", result)
            # await onTokenCallback(result)
            # print("onTokenCallback")
        return result

    def num_tokens_from_string(self, string: str) -> int:
        encoding = tiktoken.encoding_for_model(self.model)
        num_tokens = len(encoding.encode(string))
        return num_tokens

    def max_allowed_token_length(self) -> int:
        # TODO: list all models and their max tokens from api
        return 2049


# from typing import (
#     Callable,
#     List,
# )

# import openai

# import tiktoken
# from llm.basellm import BaseLLM
# from retry import retry
# from dotenv import load_dotenv 
# load_dotenv()
# import os 



                      
                      
                    
# # client.chat.completions.create(                       
                        
# temperature = 0.0
# MODEL_NAME = os.getenv('OPENAI_MODEL_NAME')
# openai.api_type = os.getenv("OPENAI_API_TYPE")
# openai.api_base = os.getenv("OPENAI_API_BASE")
# openai.api_version = os.getenv("OPENAI_API_VERSION")
# openai.api_key = os.getenv("OPENAI_API_KEY")

# DEPLOYMENT_NAME = os.getenv('OPENAI_DEPLOYMENT_NAME')

# class OpenAIChat(BaseLLM):
#     """Wrapper around OpenAI Chat large language models."""

#     def __init__(
#         self,
#         openai_api_key: str,
#         model_name: str = MODEL_NAME,
#         max_tokens: int = 1000,
#         temperature: float = 0.0,
#     ) -> None:
#         openai.api_key = openai_api_key
#         self.model = model_name
#         self.max_tokens = max_tokens
#         self.temperature = temperature

#     @retry(tries=3, delay=1)
#     def generate(
#         self,
#         messages: List[str],
#     ) -> str:
#         print("openai.__version__", openai.__version__)
#         try:
#             completions = openai.ChatCompletion.create(
#                 engine = DEPLOYMENT_NAME,
#                 # deployment_id = DEPLOYMENT_NAME,
#                 # model=self.model,
#                 # temperature=self.temperature,
#                 # max_tokens=self.max_tokens,
#                 messages=messages,
#             )
#             return completions.choices[0].message.content
#         # catch context length / do not retry
#         except openai.error.InvalidRequestError as e:
#             return str(f"Error: {e}")
#         # catch authorization errors / do not retry
#         except openai.error.AuthenticationError as e:
#             return "Error: The provided OpenAI API key is invalid"
#         except Exception as e:
#             print(f"Retrying LLM call {e}")
#             raise Exception()

#     async def generateStreaming(
#         self,
#         messages: List[str],
#         onTokenCallback=Callable[[str], None],
#     ) -> str:
#         print("openai.__version__", openai.__version__)
#         result = []
#         # completions = openai.ChatCompletion.create(
#         #     model=self.model,
#         #     temperature=self.temperature,
#         #     max_tokens=self.max_tokens,
#         #     messages=messages,
#         #     stream=True,
#         # )
#         completions = openai.ChatCompletion.create(
#                 engine = DEPLOYMENT_NAME,
#                 deployment_id = DEPLOYMENT_NAME,
#                 model=self.model,
#                 temperature=self.temperature,
#                 max_tokens=self.max_tokens,
#                 messages=messages,
#                 stream=True,
#             )
#         result = []
#         for message in completions:
#             # Process the streamed messages or perform any other desired action
#             delta = message["choices"][0]["delta"]
#             if "content" in delta:
#                 result.append(delta["content"])
#             await onTokenCallback(message)
#         return result

#     def num_tokens_from_string(self, string: str) -> int:
#         encoding = tiktoken.encoding_for_model(self.model)
#         num_tokens = len(encoding.encode(string))
#         return num_tokens

#     def max_allowed_token_length(self) -> int:
#         # TODO: list all models and their max tokens from api
#         return 2049
    
# class OpenAIChat(BaseLLM):
#     """Wrapper around OpenAI Chat large language models."""

#     def __init__(
#         self,
#         openai_api_key: str,
#         model_name: str = "gpt-3.5-turbo",
#         max_tokens: int = 1000,
#         temperature: float = 0.0,
#     ) -> None:
#         openai.api_key = openai_api_key
#         self.model = model_name
#         self.max_tokens = max_tokens
#         self.temperature = temperature

#     @retry(tries=3, delay=1)
#     def generate(
#         self,
#         messages: List[str],
#     ) -> str:
#         try:
#             completions = openai.ChatCompletion.create(
#                 model=self.model,
#                 temperature=self.temperature,
#                 max_tokens=self.max_tokens,
#                 messages=messages,
#             )
#             return completions.choices[0].message.content
#         # catch context length / do not retry
#         except openai.error.InvalidRequestError as e:
#             return str(f"Error: {e}")
#         # catch authorization errors / do not retry
#         except openai.error.AuthenticationError as e:
#             return "Error: The provided OpenAI API key is invalid"
#         except Exception as e:
#             print(f"Retrying LLM call {e}")
#             raise Exception()

#     async def generateStreaming(
#         self,
#         messages: List[str],
#         onTokenCallback=Callable[[str], None],
#     ) -> str:
#         result = []
#         completions = openai.ChatCompletion.create(
#             model=self.model,
#             temperature=self.temperature,
#             max_tokens=self.max_tokens,
#             messages=messages,
#             stream=True,
#         )
#         result = []
#         for message in completions:
#             # Process the streamed messages or perform any other desired action
#             delta = message["choices"][0]["delta"]
#             if "content" in delta:
#                 result.append(delta["content"])
#             await onTokenCallback(message)
#         return result

#     def num_tokens_from_string(self, string: str) -> int:
#         encoding = tiktoken.encoding_for_model(self.model)
#         num_tokens = len(encoding.encode(string))
#         return num_tokens

#     def max_allowed_token_length(self) -> int:
#         # TODO: list all models and their max tokens from api
#         return 2049
