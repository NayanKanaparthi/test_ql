import ollama

response = ollama.chat(model='llama3.2:3b',messages=[
    {"role": "user",
     "content": "why is the sky blue?"}
])

print(response['message']['content'])
