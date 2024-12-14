from main import Agent
import os

api_key = os.getenv('COHERE_API_KEY')

agent = Agent(api_key)

# docs = agent.load_paper("1706.03762v7.pdf")

embeddings = agent.document_embeddings("1706.03762v7.pdf")

print(embeddings)