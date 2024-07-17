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

generation_config = {
 "temperature": 0.6,
  "top_p": 0.95,
  "top_k": 64,
  "max_output_tokens": 8192,
}

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
model = genai.GenerativeModel(model_name="gemini-1.5-pro",
                              generation_config=generation_config,
                              safety_settings=safety_settings)

prompt_parts = [
  "input: ",
  "output: ",
]
model_embedding = 'models/embedding-001'
chat = model.start_chat(history=[])
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
    if 'conversations' not in session:
        session['conversations'] = [
              'user': '',
              'bot': ['Hola, soy el asistente virtual de Emeritus Latam. Estoy aquí para acompañarte en este ejercicio académico. Puedes preguntarme sobre quiénes somos, cuál es nuestro portafolio con EGADE Business School, ¿En qué puedo ayudarte hoy? ']
        ]
        
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
       
        request_e = genai.embed_content(model=model_embedding,
                              content=query,
                              task_type="retrieval_query")
        
        def find_best_passage(query, dataframe):
            """
            Compute the distances between the query and each document in the dataframe
            using the dot product.
            """
            query_embedding = genai.embed_content(model=model_embedding,
                                        content=query,
                                        task_type="retrieval_query")
            dot_products = np.dot(np.stack(dataframe['Embeddings']), query_embedding["embedding"])
            idx = np.argmax(dot_products)
            return dataframe.iloc[idx]['Text'] # Return text from index with max value
        
        passage = find_best_passage(query, df)
        print(passage)
        
        question = request.form['question']
        
        full_prompt = f"{instruction}\\n{passage}"
        
        response = chat.send_message(question + f"""{instruction}\n\n{passage}""")
        
        # Imprimir los tokens de entrada y salida
        print("Input tokens:", len(full_prompt.split()))
        print("Output tokens:", len(response.text.split()))
        
        bot_response = response.text
        print(full_prompt)
                
        response_lines = [Markup(line.replace('**', '<b>').replace('**', '</b>').replace('*', '<li>')) for line in bot_response.split('\n') if line.strip()]

        session['conversations'].append({'user': question, 'bot': response_lines})
        
       
        
        
        return jsonify({'response': response_lines})
    

    
if __name__ == '__main__':
    app.run(debug=True, port=int(os.environ.get('PORT', 4000)))