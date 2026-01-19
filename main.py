import os
import gc
from dotenv import load_dotenv

# Force use official Hugging Face API
os.environ["HF_ENDPOINT"] = "https://huggingface.co"

from mem0 import Memory

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
        "provider": "huggingface",
        "config": {
            "model": "BAAI/bge-large-en-v1.5",
        }
    },
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "embedding_model_dims": 1024
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

# Clean up to avoid shutdown warning
client.vector_store.client.close()
del client
gc.collect()
