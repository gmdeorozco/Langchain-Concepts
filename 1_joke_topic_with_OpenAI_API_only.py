from dotenv import load_dotenv
from openai import OpenAI
from colorama import Fore


load_dotenv()
llm = "gpt-5"
openAI_client = OpenAI()


def generate(text):
    """Generate text based on input"""
    instructions = f"Tell me a short joke about {text}"
    contents = [{
        
        "role":"user",
        "content":[
            {
                "type":"input_text",
                "text":f"{instructions}"
             }
        ]
        }
    ]

    
    
    joke = openAI_client.responses.create(
        model=llm,
        input=contents
    ).output_text
    
    return joke

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