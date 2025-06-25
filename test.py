from client import LLMClient

# Test single prompt
client = LLMClient()
result = client("What is 2+2?")
print(result)

# Test multiple prompts
results = client(["Hello", "How are you?"])
print(results)
