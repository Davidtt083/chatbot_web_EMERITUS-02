from dotenv import load_dotenv
load_dotenv() ## loading all the environment variables
from flask import Flask, render_template, request, session, jsonify
import google.generativeai as genai
from flow.responses import get_gemini_response 
import os
from gemini.promts import instruccion2,documents
import google.generativeai as genai
from gtts import gTTS
import tempfile
import pandas as pd
import numpy as np
from IPython.display import display
from IPython.display import Markdown
import textwrap
import markdown
from markupsafe import Markup
import re


safety_settings = [
     {
        "category": "HARM_CATEGORY_DANGEROUS",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE",
    },
]


prompt_parts = [
  "input: ",
  "output: ",
]
model_embedding = 'models/text-embedding-004'
conversations = []
instruction = instruccion2
AUDIO_FOLDER = 'templates/audio_files'
if not os.path.exists(AUDIO_FOLDER):
    os.makedirs(AUDIO_FOLDER)
    
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key="AIzaSyB4hQU2bxTC80L4sYPG4_OgvItdTqUwTl0")

def clean_text(text):
    """
    Función para limpiar el texto y eliminar las etiquetas HTML.
    """
    # Eliminar las etiquetas HTML utilizando expresiones regulares
    cleaned_text = re.sub(r'<[^>]+>|\*|#', '', text)
    return cleaned_text

app = Flask(__name__, static_folder='templates')
conversations = []
app.secret_key = os.getenv("SECRET_KEY")



@app.route('/', methods=['GET', 'POST'])
def home():
    session.clear()
    if 'conversations' not in session:
        session['conversations'] = []
        
    if request.method == 'GET':
        return render_template('index.html', conversations=session['conversations'])

    elif request.method == 'POST':
        df = pd.DataFrame(documents)
        df.columns = ['Title', 'Text']
        
        def embed_fn(title, text):
            return genai.embed_content(model=model_embedding,
                             content=text,
                             task_type="retrieval_document",
                             title=title)["embedding"]

        df['Embeddings'] = df.apply(lambda row: embed_fn(row['Title'], row['Text']), axis=1)
               
        query = request.form['question']
       
        def find_best_passage(query, dataframe):
            query_embedding = genai.embed_content(model=model_embedding,
                                        content=query,
                                        task_type="retrieval_query")
            dot_products = np.dot(np.stack(dataframe['Embeddings']), query_embedding["embedding"])
            idx = np.argmax(dot_products)
            return dataframe.iloc[idx]['Text']
        
        passage = find_best_passage(query, df)
        print(passage)
        
        question = request.form['question']
        
        # Configurar el modelo con la instrucción del sistema que incluye el passage
        generation_config = {
            "temperature": 0.6,
            "top_p": 1,
            "top_k": 1,
            "max_output_tokens": 2048,
        }
        
        system_instruction = f"{instruccion2}\n\nContext: {passage}"
        
        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config=generation_config,
        )
        
        # Crear una nueva conversación con la instrucción del sistema
        chat = model.start_chat(history=[])
        chat.send_message(system_instruction, stream=False)
        
        # Enviar solo la pregunta al modelo
        response = chat.send_message(question, stream=False)
        
        bot_response = response.text
        print(f"System Instruction:\n{system_instruction}")
        print(f"Question: {question}")
        print(f"Response: {bot_response}")
        
        response_lines = [Markup(line.replace('**', '<b>').replace('**', '</b>').replace('*', '<li>')) for line in bot_response.split('\n') if line.strip()]

        session['conversations'].append({'user': question, 'bot': response_lines})
        
        return jsonify({'response': response_lines})
    
if __name__ == '__main__':
    app.run(debug=True, port=int(os.environ.get('PORT', 4000)))