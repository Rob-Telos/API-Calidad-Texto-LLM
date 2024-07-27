#Para pruebas accesar a localhost:8000/docs y hacer uso de los distintos endpoints
#Si es necesario conectar la api a una aplicacion externa hacer uso de ngrok
#ngrok http 8000 en la CLI para exponer el servidor local a una liga externa.
import openai
import spacy
import textstat
import language_tool_python
from collections import Counter
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging
import re
from nltk.tokenize import sent_tokenize, word_tokenize, RegexpTokenizer
import json
import os

logging.basicConfig(level=logging.INFO)

#Carga de modelo Spacy y Language Tool
nlp = spacy.load("es_core_news_md")
tool = language_tool_python.LanguageTool('es')

#Carga de API KEY
openai.api_key = ''

app = FastAPI()
#Modelo de datos para prompt
class Prompt(BaseModel):
    text: str

prompts_data = []
#Reemplazar con ruta adecuada
ruta_json = ''

#Funcion que guarda resultados
def guardar_json(data, filename=ruta_json):
    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        logging.info(f"Guardado en {filename}")
    except Exception as e:
        logging.error(f"Error guardando a JSON: {e}")

#Funcion que carga resultados
def cargar_json(filename=ruta_json):
    try:
        if os.path.exists(filename):
            with open(filename, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            return []
    except Exception as e:
        logging.error(f"Error cargando JSON: {e}")
        return []

prompts_data = cargar_json()

# Función para realizar la conexión con la API de OpenAI y obtener respuesta de ella
def conexion(prompt):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            max_tokens=1000,
            temperature=0.8,
            n=1,
            stop=None,
            messages=[#Configuracion del comportamiento del modelo
                {"role": "system", "content": "Eres un asistente amigable y das informacion breve de 2 lineas"},
                {"role": "user", "content": prompt}
            ]
        )
        respuesta = response['choices'][0]['message']['content'].strip()
        
        #Calculo de metricas con ayuda de funciones, tanto entradas y salidas
        input_cohesion = cohesion(prompt)
        input_gramatica = grammaticality(prompt)
        input_complejidad = complejidad(prompt)
        input_riqueza = calcular_riqueza(prompt)
        input_huerta = calcular_fernandez_huerta(prompt)
        input_inflesz = calcular_inflesz(prompt)

        output_cohesion = cohesion(respuesta)
        output_gramatica = grammaticality(respuesta)
        output_complejidad = complejidad(respuesta)
        output_riqueza = calcular_riqueza(respuesta)
        output_huerta = calcular_fernandez_huerta(respuesta)
        output_inflesz = calcular_inflesz(respuesta)

        return respuesta, (input_cohesion, input_gramatica, input_complejidad, input_riqueza, input_huerta, input_inflesz), (output_cohesion, output_gramatica, output_complejidad, output_riqueza, output_huerta, output_inflesz)
    except Exception as e:
        logging.error(f"Error de conexion: {e}")
        return f"ocurrió un error: {e}", None, None
    
#Funcion para medir cohesion de texto NPPC/NTPP
def cohesion(text):
    try:
        doc = nlp(text)
        cohesion = 0
        for token in doc:
            if token.head != token:
                cohesion += token.similarity(token.head)
        return cohesion / len(doc) if len(doc) > 0 else 0
    except Exception as e:
        logging.error(f"Error en funcion de cohesion: {e}")
        return 0

#Funcion para medir grmatica de texto 1-NEG/NEGT
def grammaticality(text):
    try:
        matches = tool.check(text)
        return 1 - len(matches) / len(text.split()) if len(text.split()) > 0 else 0
    except Exception as e:
        logging.error(f"Error en funcion de gramaticalidad: {e}")
        return 0

# Funcion para medir la complejidad NPU/NP
def complejidad(texto):
    try:
        textstat.set_lang("es")
        num_palabras = textstat.lexicon_count(texto, removepunct=True)
        palabras = texto.split()
        num_palabras_unicas = len(Counter(palabras))
        relacion_palabras_unicas = num_palabras_unicas / num_palabras
        return relacion_palabras_unicas
    except Exception as e:
        logging.error(f"Error en la función complejidad: {e}")
        return 0
#Funcion que implementa tecnica MATTR para la funcion de riqueza lexica
def calculate_mattr(text, window_size=100):
    words = word_tokenize(text, language='spanish')
    ttr_values = []
    for i in range(len(words) - window_size + 1):
        window = words[i:i + window_size]
        unique_words = len(set(window))
        ttr = unique_words / window_size
        ttr_values.append(ttr)
    return sum(ttr_values) / len(ttr_values) if ttr_values else 0

#Funcion para el calculo de silabas, usada para indices
def calcular_silabas(palabra):
    try:
        palabra = palabra.lower().replace('ü', 'u')
        palabra = re.sub(r'(ue|ui|iu|ie|eu|ei|oi|io|ou|uo|au|ai|aí|aú|eí|eú|oí|oú)', 'a', palabra)
        return len(re.findall(r'[aeiouáéíóú]', palabra))
    except Exception as e:
            logging.error(f"Error en la función silabas: {e}")
            return 0
#Funcion para calcular riqueza lexica
def calcular_riqueza(texto, window_size=100):
    try:
        riqueza = calculate_mattr(texto, window_size)
        return riqueza
    except Exception as e:
            logging.error(f"Error en la función riqueza: {e}")
            return 0

#Funcion con calculos necesarios para indices
def calculos(texto):
    oraciones = sent_tokenize(texto, language='spanish')
    tokenizer = RegexpTokenizer(r'\w+')
    palabras = tokenizer.tokenize(texto)
    total_silabas = sum(calcular_silabas(palabra) for palabra in palabras)
    total_palabras = len(palabras)
    total_frases = len(oraciones)
    
    return total_silabas, total_palabras, total_frases

#Funcion para calcular indice FERNANDEZ-HUERTA
def calcular_fernandez_huerta(texto):
    try:
        total_silabas, total_palabras, total_frases = calculos(texto)

        P = (total_silabas / total_palabras) * 100
        F = (total_palabras / total_frases)

        huerta = 206.84 - (0.60 * P) - (1.02 * F)
        return huerta
    except Exception as e:
        logging.error(f"Error en la función calcular_fernandez_huerta: {e}")
        return 0
    
#Funcion para calcular indice INFLESZ
def calcular_inflesz(texto):
    try:
        total_silabas, total_palabras, total_frases = calculos(texto)

        inflesz = 206.835 - (62.3 * (total_silabas / total_palabras)) - (total_palabras / total_frases)
        return inflesz
    except Exception as e:
        logging.error(f"Error en la función calcular_inflesz: {e}")
        return 0

# Endpoint para obtener todos los prompts almacenados
@app.get("/prompts/")
async def get_prompts():
    global prompts_data
    prompts_data = cargar_json()
    if not prompts_data:
        raise HTTPException(status_code=404, detail="No se ha publicado algun prompt.")
    return prompts_data

# Endpoint para procesar un nuevo prompt
@app.post("/procesar_prompt/")
async def procesar_prompt(prompt: Prompt):
    global prompts_data
    respuesta, input_scores, output_scores = conexion(prompt.text)
    if "ocurrió un error" in respuesta:
        raise HTTPException(status_code=500, detail=respuesta)
    prompt_data = {
        "prompt": prompt.text,
        "respuesta": respuesta,
        "Entradas": {
            "cohesion": input_scores[0],
            "gramatica": input_scores[1],
            "complejidad": input_scores[2],
            "riqueza": input_scores[3],
            "huerta": input_scores[4],
            "inflesz": input_scores[5]
        },
        "Salidas": {
            "cohesion": output_scores[0],
            "gramatica": output_scores[1],
            "complejidad": output_scores[2],
            "riqueza": output_scores[3],
            "huerta": output_scores[4],
            "inflesz": output_scores[5]
        }
    }
    prompts_data.append(prompt_data)
    guardar_json(prompts_data)
    return prompt_data

# Configuracion de la aplicacion, trabajando en el puerto 8000
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000, log_level="info")



