import os
import time
import pyautogui
import base64
import requests
import cv2  # Importa a biblioteca OpenCV
from PIL import Image
from openai import OpenAI
from dotenv import load_dotenv
import azure.cognitiveservices.speech as speechsdk
from concurrent.futures import ThreadPoolExecutor
import threading
import mediapipe as mp
import numpy as np
import random

def emitir_som_pensamento():
    pensamentos = ["ahnn?...", "humm...", "err..."]
    pensamento = random.choice(pensamentos)

    # Ajuste o SSML para um som ainda mais reflexivo
    ssml_text = f'''
    <speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="pt-BR">
        <voice name="{speech_config.speech_synthesis_voice_name}">
            <prosody rate="x-slow" pitch="-10%" volume="x-soft">
                {pensamento}
            </prosody>
        </voice>
    </speak>
    '''
    
    speech_synthesis_result = speech_synthesizer.speak_ssml_async(ssml_text).get()

    if speech_synthesis_result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        print(f"Som de pensamento '{pensamento}' emitido com sucesso.")
    elif speech_synthesis_result.reason == speechsdk.ResultReason.Canceled:
        print(f"Erro ao emitir som de pensamento: {speech_synthesis_result.cancellation_details.reason}")


# Carrega as variáveis de ambiente do arquivo .env
load_dotenv()

# Configuração da chave da API da OpenAI a partir de uma variável de ambiente
api_key = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# Lista para armazenar o histórico da conversa
conversation_history = []

# Configurações globais para Speech SDK
speech_key = os.environ.get("AZURE_SPEECH_KEY")
service_region = os.environ.get("AZURE_SERVICE_REGION")
speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
speech_config.set_property(speechsdk.PropertyId.Speech_LogFilename, "speech_sdk_log.txt")
speech_config.speech_recognition_language = 'pt-BR'
speech_config.speech_synthesis_voice_name = 'pt-BR-JulioNeural'
audio_input = speechsdk.AudioConfig(use_default_microphone=True)

# Inicializa o sintetizador de fala uma vez
speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config)

# Variável global para armazenar o frame atual da webcam
current_frame = None

class FaceDetector():
    def __init__(self, 
                 min_detec_confidence: float = 0.5, 
                 smooth_factor: float = 0.7):
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(min_detec_confidence)
        self.mp_draw = mp.solutions.drawing_utils
        self.smooth_factor = smooth_factor
        self.prev_bbox = None

    def find_faces(self, img, draw_faces: bool = True):
        img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.face_detection.process(img_RGB)
        bboxs = []

        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = img.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                       int(bboxC.width * iw), int(bboxC.height * ih)

                if self.prev_bbox is None:
                    self.prev_bbox = bbox
                else:
                    bbox = tuple(
                        int(self.smooth_factor * new + (1 - self.smooth_factor) * prev)
                        for new, prev in zip(bbox, self.prev_bbox)
                    )
                    self.prev_bbox = bbox

                bboxs.append([id, bbox, detection.score])

                if draw_faces:
                    img = self.fancy_draw(img, bbox)

        return img, bboxs

    def fancy_draw(self, img, bbox, l=15, t=2, rt=1):
        x, y, w, h = bbox
        x1, y1 = x + w, y + h

        color = (0, 0, 255)
        cv2.line(img, (x, y), (x + l, y), color, t)
        cv2.line(img, (x, y), (x, y + l), color, t)

        cv2.line(img, (x1, y), (x1 - l, y), color, t)
        cv2.line(img, (x1, y), (x1, y + l), color, t)

        cv2.line(img, (x, y1), (x + l, y1), color, t)
        cv2.line(img, (x, y1), (x, y1 - l), color, t)

        cv2.line(img, (x1, y1), (x1 - l, y1), color, t)
        cv2.line(img, (x1, y1), (x1, y1 - l), color, t)

        overlay = img.copy()
        offset = 20
        rect_y1 = y1 + offset
        rect_y2 = rect_y1 + 40
        cv2.rectangle(overlay, (x, rect_y1), (x1, rect_y2), (0, 0, 255), -1)
        alpha = 0.6
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

        font = cv2.FONT_HERSHEY_SIMPLEX
        text = "ALVO TRAVADO"
        margin = 20
        font_scale = (w - 2 * margin) / 250
        thickness = 2
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = x + margin + (w - 2 * margin - text_size[0]) // 2
        text_y = rect_y1 + (rect_y2 - rect_y1) // 2 + text_size[1] // 2
        text_color = (115, 118, 255)

        cv2.putText(img, text, (text_x, text_y), font, font_scale, text_color, thickness)

        return img


def webcam_thread():
    global current_frame
    face_detector = FaceDetector()  # Inicializa o FaceDetector
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if ret:
            frame, bboxs = face_detector.find_faces(frame)  # Detecta e desenha faces
            current_frame = frame
        time.sleep(0.1)  # Capture a cada 0.1 segundo
    cap.release()


def exibir_webcam():
    global current_frame
    while True:
        if current_frame is not None:
            cv2.imshow('Webcam', current_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

# Iniciar a thread da webcam
threading.Thread(target=webcam_thread, daemon=True).start()

# Iniciar a thread para exibir a webcam em tempo real
threading.Thread(target=exibir_webcam, daemon=True).start()

# Código previamente importado permanece o mesmo

def azure_speech_to_text():
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_input)

    def recognized(args):
        recognized_text = args.result.text.lower()
        if any(word in recognized_text for word in ["skynet", "skainete", "skai neti", "skainet", "skai nete", "sky net", "scai nete", "scai neti", "skyneti", "skai net", "skainet", "skynetee", "skynetie", "escanete", "esca nete", "escanet", "esca net", "esca neti", "ska net", "ska neti", "sky nete", "escainet", "skainee", "skyanet", "skait net", "escainee", "sky nat"]):
            command_start = next((word for word in ["skynet", "skainete", "skai neti", "skainet", "skai nete", "sky net", "scai nete", "scai neti", "skyneti", "skai net", "skainet", "skynetee", "skynetie", "escanete", "esca nete", "escanet", "esca net", "esca neti", "ska net", "ska neti", "sky nete", "escainet", "skainee", "skyanet", "skait net", "escainee", "sky nat"] if word in recognized_text), None)
            if command_start:
                command_start_index = recognized_text.find(command_start)
                command_text = recognized_text[command_start_index + len(command_start):].strip()
                
                if command_text:
                    with ThreadPoolExecutor() as executor:
                        # executor.submit(emitir_som_pensamento)

                        # Captura o comando adicional se começar com variações de "consegue", "poderia" ou "leia"
                        activation_words = [
                            # Variações de "consegue"
                            "consegue", "consegue ","consigue", "consegue", "consigui", "consegi", "consege", 
                            "consiegue", "consequi", "concegue", "conseki", "consig", "conseguee", 
                            "consegu", "consigu", "conseguei", "consegueu", "consegui", "conseque", 
                            "conseg", "conseguie",
                            # Variações de "poderia"
                            "poderia", "poderia ", "poderiá", "puderia", "poderi", "podería", "pudería", 
                            "puderia", "puderia", "poder", "pudér", "poderiã", "podria", 
                            "podéría", "podéria", "poderai", "poderiadizer"
                        ]
                        if any(command_text.startswith(word) for word in activation_words):
                            matching_word = next(word for word in activation_words if command_text.startswith(word))
                            additional_text = command_text[len(matching_word):].strip()
                        elif command_text.startswith("leia esse"):
                            additional_text = command_text[len("leia esse"):].strip()
                        else:
                            additional_text = None
                        
                        # Se additional_text for preenchido, ativa a visão
                        if additional_text:
                            image_path = salvar_imagem_webcam()
                            if image_path:
                                # Inicia a emissão de sons de pensamento em paralelo
                                executor.submit(emitir_som_pensamento)
                                response_text = analisar_imagem_com_texto(image_path, additional_text)
                                executor.submit(responder, response_text)
                        else:
                            # Se não for "consegue" ou "leia", continua com o processo normal
                            response_text = ask_chatgpt(command_text)
                            executor.submit(responder, response_text)

    def canceled(args):
        print(f"Cancelado: {args.cancellation_details.reason}")
        if args.cancellation_details.reason == speechsdk.CancellationReason.Error:
            print(f"Erro: {args.cancellation_details.error_details}")
            print(f"ID da Sessão: {args.session_id}")

    speech_recognizer.recognized.connect(recognized)
    speech_recognizer.canceled.connect(canceled)

    speech_recognizer.start_continuous_recognition()
    try:
        print("Ouvindo...")
        while True:
            time.sleep(0.1)  # Reduzido de 1 segundo para 0.1 segundo
    except KeyboardInterrupt:
        pass
    finally:
        speech_recognizer.stop_continuous_recognition()

# Código restante permanece o mesmo

def capturar_comando_adicional():
    additional_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_input)
    print("Ouvindo comando adicional...")
    result = additional_recognizer.recognize_once_async().get()
    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        return result.text.lower().strip()
    elif result.reason == speechsdk.ResultReason.NoMatch:
        print("Nenhum comando adicional reconhecido.")
    elif result.reason == speechsdk.ResultReason.Canceled:
        print(f"Erro ao capturar comando adicional: {result.cancellation_details.reason}")
    return ""

def ask_chatgpt(prompt):
    try:
        conversation_history.append({"role": "user", "content": prompt})
        system_message = {
            "role": "system",
            "content": "Use um tom sério, de autoridade e dê respostas curtas e objetivas. Se eu perguntar 'Skynet, você está ai?', apenas responda: 'Estou sempre presente.'. Caso eu te faça uma pergunta aberta, seja inigmático e misterioso. Você se chama Skynet, uma IA avançada criada por Bruno Bracaioli. Não use negrito e nem itálico. Escreva tudo em um único parágrafo. Não responda em formato de lista ou de bullet point. Responda tudo em um único parágrafo sem formatação de texto."
        }
        messages = [system_message] + conversation_history
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=100
        )
        
        response_text = response.choices[0].message.content.strip()
        conversation_history.append({"role": "assistant", "content": response_text})
        
        return response_text
    except Exception as e:
        print(f"Erro ao chamar a API do ChatGPT: {e}")
        return "Desculpe, houve um erro ao processar sua solicitação."

def responder(texto):
    ssml_text = f'<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="pt-BR"><voice name="{speech_config.speech_synthesis_voice_name}" style="serious" role="SeniorMale"><prosody rate = "20%" pitch="-10%" volume="loud" >{texto}</prosody></voice></speak>'

    speech_synthesis_result = speech_synthesizer.speak_ssml_async(ssml_text).get()
    
    if speech_synthesis_result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        print("Resposta falada com sucesso.")
    elif speech_synthesis_result.reason == speechsdk.ResultReason.Canceled:
        print(f"Não foi possível sintetizar a fala: {speech_synthesis_result.cancellation_details.reason}")

def salvar_imagem_webcam():
    global current_frame
    try:
        if current_frame is not None:
            image_path = "webcam_image.png"
            cv2.imwrite(image_path, current_frame)
            return image_path
        else:
            print("Nenhum frame disponível da webcam.")
            return None
    except Exception as e:
        print(f"Erro ao salvar imagem da webcam: {e}")
        return None

def analisar_imagem_com_texto(image_path, additional_text):
    try:
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Você é uma IA que recebe uma imagem junto com uma pergunta. "
                        "Sempre que receber uma imagem, você deve responder com base no conteúdo da imagem. "
                        "Você está sempre 'vendo' a imagem fornecida e deve assumir que a pergunta se refere a algo presente na imagem. "
                        "Não diga que não consegue ver; sempre interprete o conteúdo da imagem."
                        "Se te perguntar se você 'consegue ver', estou dizendo para interpretar o conteúdo da imagem de acordo com a pergunta."
                        "Se eu perguntar se você consegue 'ver' ou 'ler' ou 'dizer', estou querendo dizer para interpretar o conteúdo da imagem de acordo com a pergunta."
                    )
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Se eu perguntar se você consegue 'ver' ou 'ler' ou 'dizer', estou querendo dizer para interpretar o conteúdo da imagem de acordo com a pergunta. Responda de maneira curta e objetiva apenas a pergunta: '{additional_text}'"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 100
        }

        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

        if response.status_code == 200:
            description = response.json()['choices'][0]['message']['content']
            
            conversation_history.append({"role": "assistant", "content": description})

            return description
        else:
            return f"Erro na solicitação da API: {response.status_code} - {response.text}"

    except Exception as e:
        print(f"Erro ao analisar imagem: {e}")
        return "Desculpe, houve um erro ao analisar a imagem."

# Executa a função principal
azure_speech_to_text()
