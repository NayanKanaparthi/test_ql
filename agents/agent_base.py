# agents/agent_base.py

import ollama
from abc import ABC, abstractmethod
from loguru import logger
import os
from dotenv import load_dotenv

# Load environment variables
#load_dotenv()

#openai.api_key = os.getenv("OPENAI_API_KEY")

class AgentBase(ABC):
    def __init__(self, name, max_retries=2, verbose=True):
        self.name = name
        self.max_retries = max_retries
        self.verbose = verbose

    @abstractmethod
    def execute(self, *args, **kwargs):
        pass

    def call_llama(self, messages, temperature=0.7, max_tokens=150):
        retries = 0
        while retries < self.max_retries:
            try:
                if self.verbose:
                    logger.info(f"[{self.name}] Sending messages to ollama:")
                    for msg in messages:
                        logger.debug(f"  {msg['role']}: {msg['content']}")

               #calling ollama using chat api
                response = ollama.chat(model='llama3.2:3b', messages=messages)
                reply = response['message']['content']  

                if self.verbose:
                    logger.info(f"[{self.name}] Received response: {reply}")
                return reply
            except Exception as e:
                retries += 1
                logger.error(f"[{self.name}] Error during Ollama call: {e}. Retry {retries}/{self.max_retries}")
        raise Exception(f"[{self.name}] Failed to get response from Ollama after {self.max_retries} retries.")