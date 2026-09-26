from google import genai
import os

client = genai.Client(vertexai=True, api_key=os.environ["GOOGLE_API_KEY"])
resp = client.models.generate_content(
    model="gemini-2.5-flash",  # use a model name your console shows
    contents="Say hello in one sentence.",
)
print(resp.text)