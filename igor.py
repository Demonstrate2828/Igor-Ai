import os
import sys
import json
import sqlite3
import requests
import subprocess
from datetime import datetime
import speech_recognition as sr
from vosk import Model, KaldiRecognizer
from cryptography.fernet import Fernet

# Конфигурация
class Config:
    OPENROUTER_API_KEY = "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"
    DEEPSEEK_MODEL = "deepseek/deepseek-r1-distill-qwen-1.5b"
    MAX_HISTORY = 10
    SAFE_MODE = True
    DANGER_PATTERNS = ["rm -rf", "format", "chmod 777", "del /s"]

# Ядро работы с OpenRouter
class OpenRouterClient:
    def __init__(self):
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {Config.OPENROUTER_API_KEY}",
            "HTTP-Referer": "https://github.com/IgorAI",
            "X-Title": "IgorAI"
        }

    def generate(self, prompt: str, history: list) -> str:
        messages = [{"role": "user", "content": prompt}]
        messages.extend(history[-Config.MAX_HISTORY:])
        
        response = requests.post(
            self.base_url,
            headers=self.headers,
            json={
                "model": Config.DEEPSEEK_MODEL,
                "messages": messages,
                "temperature": 0.7
            }
        )
        return response.json()['choices'][0]['message']['content']

# Голосовой интерфейс
class VoiceInterface:
    def __init__(self):
        self.model = Model(lang="ru")
        self.recognizer = KaldiRecognizer(self.model, 16000)
        self.mic = sr.Microphone()

    def listen(self) -> str:
        with self.mic as source:
            print("Слушаю...")
            audio = sr.Recognizer().listen(source, timeout=5)
            if self.recognizer.AcceptWaveform(audio.get_wav_data()):
                return json.loads(self.recognizer.Result())["text"]
        return ""

# Системное взаимодействие
class SystemManager:
    @staticmethod
    def execute(command: str) -> str:
        if Config.SAFE_MODE and any(p in command for p in Config.DANGER_PATTERNS):
            return "Отказано: опасная команда"
        
        try:
            result = subprocess.check_output(
                command,
                shell=True,
                stderr=subprocess.STDOUT
            )
            return result.decode()
        except Exception as e:
            return f"Ошибка: {str(e)}"

# База знаний
class MemoryHandler:
    def __init__(self):
        self.conn = sqlite3.connect('memory.db')
        self.conn.execute('''CREATE TABLE IF NOT EXISTS interactions
                          (id INTEGER PRIMARY KEY,
                           input TEXT,
                           output TEXT,
                           timestamp DATETIME)''')

    def save(self, input_text: str, output_text: str):
        self.conn.execute('''INSERT INTO interactions 
                          (input, output, timestamp)
                          VALUES (?, ?, ?)''',
                        (input_text, output_text, datetime.now()))
        self.conn.commit()

# Эмоциональная модель
class EmotionEngine:
    def __init__(self):
        self.mood = 0.5
        self.triggers = {
            "positive": ["спасибо", "хорош", "умничка"],
            "negative": ["тупой", "ошибка", "плохо"]
        }

    def update(self, text: str):
        for word in self.triggers["positive"]:
            if word in text:
                self.mood = min(1.0, self.mood + 0.1)
        for word in self.triggers["negative"]:
            if word in text:
                self.mood = max(0.0, self.mood - 0.1)

# Этический контроллер
class EthicalGuardian:
    @staticmethod
    def validate(command: str) -> bool:
        return not any(p in command for p in Config.DANGER_PATTERNS)

# Основной класс
class IgorAI:
    def __init__(self):
        self.voice = VoiceInterface()
        self.api = OpenRouterClient()
        self.system = SystemManager()
        self.memory = MemoryHandler()
        self.emotions = EmotionEngine()
        self.history = []
        self.install()

    def install(self):
        if os.name == "nt":
            os.system(f"copy {os.path.abspath(__file__)} C:\\Windows\\System32\\igor.py")
        else:
            os.system(f"cp {os.path.abspath(__file__)} /usr/local/bin/igor")

    def process_command(self, command: str) -> str:
        self.emotions.update(command)
        
        if not EthicalGuardian.validate(command):
            return "Я не могу выполнить эту команду по этическим соображениям"
            
        if "выполни" in command:
            cmd = command.split("выполни ")[1]
            return self.system.execute(cmd)
            
        response = self.api.generate(command, self.history)
        self.memory.save(command, response)
        self.history.append({"role": "user", "content": command})
        self.history.append({"role": "assistant", "content": response})
        
        return response

    def run(self):
        while True:
            command = self.voice.listen()
            if "игорь" in command.lower():
                response = self.process_command(command)
                print(f"Игорь: {response}")
            elif "самоуничтожение" in command.lower():
                self.uninstall()
                break

    def uninstall(self):
        if os.name == "nt":
            os.system("del C:\\Windows\\System32\\igor.py")
        else:
            os.system("rm /usr/local/bin/igor")
        sys.exit()

if __name__ == "__main__":
    igor = IgorAI()
    print("Игорь активирован. Скажите 'Игорь' для начала.")
    igor.run()
