

from langchain_ollama import OllamaLLM

def main():
    #Point at local model 
    llm = OllamaLLM(model="deepseek-r1")  

    
    prompt = "Hello from Ollama! Please tell me a fun fact."
    response = llm.invoke(prompt)         

    print("🦙 Ollama says:\n", response)

if __name__ == "__main__":
    main()
