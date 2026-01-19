import os
from dotenv import load_dotenv
from mem0 import Memory
import time

load_dotenv()

config = {
    "llm": {
        "provider": "openai",
        "config": {
            "model": "meta/llama-3.3-70b-instruct",
            "api_key": os.getenv("NVIDIA_API_KEY"),
            "openai_base_url": "https://integrate.api.nvidia.com/v1",
            "temperature": 1.0,
        }
    },
    "embedder": {
        "provider": "openai",
        "config": {
            "model": "text-embedding-3-small",
            "api_key": os.getenv("OPENAI_API_KEY"),
        }
    },
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "embedding_model_dims": 1536
        }
    }
}

client = Memory.from_config(config)

messages = [
  { "role": "user", "content": "Hi, I'm Alex. I'm a vegetarian and I'm allergic to nuts." },
  { "role": "assistant", "content": "Hello Alex! I see that you're a vegetarian with a nut allergy." }
]

add_result = client.add(messages, user_id="alex")
print("Add result:", add_result)

# Use a more relevant query for semantic search
query = "What dietary restrictions does Alex have?"

result = client.search(query, user_id="alex")
print(f"\nQuery: {query}")
print(f"Result: {result}")
