import os
import gc
from dotenv import load_dotenv

# Force use official Hugging Face API
os.environ["HF_ENDPOINT"] = "https://huggingface.co"

from mem0 import Memory
from token_tracker import TokenTracker
from cosine_search import cosine_search

load_dotenv()


tracker = TokenTracker()

config = {
    "llm": {
        "provider": "openai",
        "config": {
            "model": "meta/llama-3.3-70b-instruct",
            "api_key": os.getenv("NVIDIA_API_KEY"),
            "openai_base_url": "https://integrate.api.nvidia.com/v1",
            "temperature": 1.0,
            "response_callback": tracker.callback,
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
            "embedding_model_dims": 1024,
            "path": "./vector_db",
            "on_disk": True
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

query = "What dietary restrictions does Alex have?"

# result = client.search(query, user_id="alex")
# print(f"\nQuery: {query}")
# print(f"Result: {result}")

# Custom cosine similarity search
print("\n--- Custom Cosine Search ---")
result = cosine_search(client, query, user_id="alex", limit=5, threshold=0.5)
for item in result:
    print(f"  Score: {item['score']:.4f} | Memory: {item['memory']}")

# Print token usage summary
tracker.summary()

# Clean up to avoid shutdown warning
client.vector_store.client.close()
del client
gc.collect()
