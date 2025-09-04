from dotenv import load_dotenv
from google import genai 
from google.genai import types
from colorama import Fore
import os


load_dotenv()
llm = "gemini-2.5-flash-lite"
genai_client = genai.Client(
    vertexai=os.environ.get("VEREXAI"),
    project=os.environ.get("PROJECT"),
    location=os.environ.get("LOCATION")
)


def generate(text):
    """Generate text based on input"""
    instructions = f"Tell me a short joke about {text}"
    contents = [
        types.Content(
        role="user",
        parts=[
            types.Part.from_text(text=instructions)
        ]
        )
    ]
    
    generate_content_config = types.GenerateContentConfig(
        temperature=1,
        top_p=0.95,
        max_output_tokens=256

    )
    
    joke = genai_client.models.generate_content(
        model=llm,
        contents=contents,
        config=generate_content_config
    ).text
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