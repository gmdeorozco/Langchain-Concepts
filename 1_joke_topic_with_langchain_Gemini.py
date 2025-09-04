from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from colorama import Fore

load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")

prompt_template = ChatPromptTemplate.from_template("Tell me a short joke about {topic}")
output_parser = StrOutputParser()

def generate(text):
    """Generate text based on input"""
    chain = prompt_template | llm | output_parser
    return chain.invoke({"topic": text})

def start():
    """Start the application"""
    instructions = ("Type your JOKE topic and press ENTER.")
    print(Fore.BLUE + "\n\x1B[3m" + instructions + "\x1B[0m" + Fore.RESET)
    
    while True:
        joke_topic = input(Fore.YELLOW + "\n\x1B[3m" + "JOKE TOPIC: " + "\x1B[0m" + Fore.RESET)
        if joke_topic.lower() == "exit":
            print("BYE!")
            break
        joke = generate(joke_topic)
        print(Fore.GREEN + "\n\x1B[3m" + joke + "\x1B[0m" + Fore.RESET)
        
if __name__ == "__main__":
    start()