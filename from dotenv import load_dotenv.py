from dotenv import load_dotenv
load_dotenv() ## loading all the environment variables
import google.generativeai as genai
import textwrap
from IPython.display import display
from IPython.display import Markdown   
import os
GOOGLE_API_KEY = 'AIzaSyDd3i0jHtnM1XqhA9t7msPzyLJcn8vxooM'
genai.configure(api_key=GOOGLE_API_KEY)


model = genai.GenerativeModel('gemini-pro')
chat = model.start_chat(history=[])
question = input("Usuario: ")
conversations = []

def get_gemini_response(question):
    while True:
    
     response=chat.send_message(question)
     print (response.text)
     question = input("Usuario: ")
     
     
     
def generate_image(text):
    try:
        model = genai.GenerativeModel('gemini-pro')  # Instancia el modelo Gemini
        image = model.generate_image(text)  # Genera la imagen
        return image.to_bytes()  # Convierte la imagen a bytes
    except Exception as e:
        print("Error al generar la imagen:", e)
        return None