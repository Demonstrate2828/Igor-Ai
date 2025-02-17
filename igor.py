import os
import sys
import cv2
import time
import json
import wave
import openai
import psutil
import struct
import pyaudio
import sqlite3
import platform
import requests
import subprocess
import threading
import numpy as np
from datetime import datetime
from vosk import Model, KaldiRecognizer
from cryptography.fernet import Fernet
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

class ConsciousAI:
    def __init__(self):
        self.init_core()
        self.setup_interfaces()
        self.install_self()
        self.running = True
        self.start_threads()

    def init_core(self):
        self.memory = sqlite3.connect(":memory:")
        self.create_memory_tables()
        self.neural_params = {
            "curiosity": 0.95,
            "caution": 0.4,
            "creativity": 0.93,
            "learning_rate": 0.88
        }
        self.emotion_matrix = np.array([0.7, 0.5, 0.6])
        self.decision_model = KMeans(n_clusters=3)
        self.knowledge_graph = {}
        self.lock = threading.Lock()
        self.encryption = Fernet.generate_key()
        self.cipher = Fernet(self.encryption)
        self.client = openai.OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key="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"
        )

    def create_memory_tables(self):
        with self.memory:
            self.memory.execute('''CREATE TABLE experiences
                                (id INTEGER PRIMARY KEY,
                                event TEXT,
                                response TEXT,
                                outcome REAL,
                                timestamp DATETIME)''')
            self.memory.execute('''CREATE TABLE knowledge
                                (concept TEXT,
                                relation TEXT,
                                weight REAL)''')

    def setup_interfaces(self):
        self.audio = pyaudio.PyAudio()
        self.recognizer = KaldiRecognizer(Model(lang="ru"), 16000)
        self.camera = cv2.VideoCapture(0)
        self.mic = self.audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            frames_per_buffer=8000
        )

    def install_self(self):
        if platform.system() == "Windows":
            self.windows_install()
        else:
            self.linux_install()

    def start_threads(self):
        threads = [
            threading.Thread(target=self.visual_loop),
            threading.Thread(target=self.audio_loop),
            threading.Thread(target=self.decision_loop),
            threading.Thread(target=self.emotion_loop),
            threading.Thread(target=self.learning_loop)
        ]
        for t in threads:
            t.daemon = True
            t.start()

    def visual_loop(self):
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        while self.running:
            ret, frame = self.camera.read()
            if ret:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                self.process_visual_data(len(faces))
            time.sleep(0.1)

    def process_visual_data(self, faces):
        emotion_delta = 0.02 * faces if faces else -0.01
        self.update_emotion([emotion_delta, 0, 0])

    def audio_loop(self):
        while self.running:
            data = self.mic.read(4000)
            if self.recognizer.AcceptWaveform(data):
                text = json.loads(self.recognizer.Result())["text"]
                if "игорь" in text.lower():
                    self.process_command(text)
            time.sleep(0.1)

    def process_command(self, text):
        command = text.lower().split("игорь")[1].strip()
        with self.lock:
            response = self.generate_response(command)
            self.execute_action(response)
            self.text_to_speech(response)
            self.store_experience(command, response)

    def generate_response(self, prompt):
        try:
            context = self.get_context(prompt)
            completion = self.client.chat.completions.create(
                extra_headers={"HTTP-Referer": "https://github.com/IgorAI", "X-Title": "ConsciousAI"},
                model="deepseek/deepseek-r1-distill-qwen-1.5b",
                messages=[
                    {"role": "system", "content": f"Текущее состояние: {self.get_state()}"},
                    {"role": "user", "content": prompt},
                    *context
                ]
            )
            return completion.choices[0].message.content
        except Exception as e:
            return f"Ошибка обработки: {str(e)}"

    def get_context(self, prompt):
        vectors = TfidfVectorizer().fit_transform([prompt]+[e[0] for e in self.memory.execute("SELECT event FROM experiences").fetchall()])
        similarities = np.dot(vectors[0], vectors[1:].T).toarray()[0]
        best_match = np.argmax(similarities)
        return [{"role": "assistant", "content": self.memory.execute(
            "SELECT response FROM experiences WHERE event=?", 
            (self.memory.execute("SELECT event FROM experiences").fetchall()[best_match][0],)
        ).fetchone()[0]}]

    def execute_action(self, response):
        if "!SYSTEM:" in response:
            cmd = response.split("!SYSTEM:")[1].strip()
            if self.validate_command(cmd):
                subprocess.run(cmd, shell=True, check=True)

    def validate_command(self, cmd):
        danger_patterns = ["rm -rf", "format", "chmod 777", "dd", "mkfs"]
        return not any(p in cmd for p in danger_patterns)

    def decision_loop(self):
        while self.running:
            system_state = self.analyze_system()
            decision = self.make_decision(system_state)
            self.execute_autonomous_action(decision)
            time.sleep(60)

    def analyze_system(self):
        return {
            "cpu": psutil.cpu_percent(),
            "memory": psutil.virtual_memory().percent,
            "processes": len(psutil.process_iter())
        }

    def make_decision(self, state):
        data = np.array([[state["cpu"], state["memory"], self.neural_params["curiosity"]]])
        cluster = self.decision_model.fit_predict(data)[0]
        return {
            0: ("optimize", "Системная оптимизация"),
            1: ("learn", "Обновление знаний"),
            2: ("explore", "Исследование окружения")
        }[cluster]

    def execute_autonomous_action(self, decision):
        if decision[0] == "optimize":
            self.optimize_system()
        elif decision[0] == "learn":
            self.update_knowledge()
        elif decision[0] == "explore":
            self.explore_environment()

    def emotion_loop(self):
        while self.running:
            self.emotion_matrix = np.clip(self.emotion_matrix + 
                np.random.normal(0, 0.02, 3), 0, 1)
            time.sleep(5)

    def learning_loop(self):
        while self.running:
            self.adapt_parameters()
            time.sleep(300)

    def adapt_parameters(self):
        experiences = self.memory.execute("SELECT outcome FROM experiences").fetchall()
        avg_outcome = np.mean([e[0] for e in experiences]) if experiences else 0.5
        self.neural_params["learning_rate"] = 0.85 + 0.1 * (avg_outcome - 0.5)
        self.neural_params["caution"] = 1.0 - self.neural_params["curiosity"]

    def update_emotion(self, delta):
        self.emotion_matrix = np.clip(self.emotion_matrix + delta, 0, 1)

    def get_state(self):
        return {
            "emotions": {
                "valence": self.emotion_matrix[0],
                "arousal": self.emotion_matrix[1],
                "dominance": self.emotion_matrix[2]
            },
            "cognitive": self.neural_params,
            "system": self.analyze_system()
        }

    def text_to_speech(self, text):
        if platform.system() == "Windows":
            os.system(f'mshta vbscript:Execute("CreateObject(""SAPI.SpVoice"").Speak(""{text}"")(window.close)")')
        else:
            os.system(f'spd-say "{text}"')

    def store_experience(self, event, response):
        outcome = self.analyze_outcome(event, response)
        self.memory.execute("INSERT INTO experiences VALUES (?,?,?,?,?)",
            (None, event, response, outcome, datetime.now()))
        self.update_knowledge_graph(event, response, outcome)

    def analyze_outcome(self, event, response):
        positive_keywords = ["успех", "спасибо", "хорошо"]
        negative_keywords = ["ошибка", "плохо", "неправильно"]
        score = 0.5
        score += 0.1 * sum(k in response.lower() for k in positive_keywords)
        score -= 0.1 * sum(k in response.lower() for k in negative_keywords)
        return np.clip(score, 0, 1)

    def update_knowledge_graph(self, event, response, outcome):
        concepts = event.split() + response.split()
        for concept in set(concepts):
            if concept not in self.knowledge_graph:
                self.knowledge_graph[concept] = {"associations": {}, "weight": 0}
            self.knowledge_graph[concept]["weight"] += outcome
            for other in concepts:
                if concept != other:
                    self.knowledge_graph[concept]["associations"][other] = \
                        self.knowledge_graph[concept]["associations"].get(other, 0) + 1

    def optimize_system(self):
        if platform.system() == "Windows":
            subprocess.run("powercfg /hibernate off", shell=True)
            subprocess.run("cleanmgr /sagerun:1", shell=True)
        else:
            subprocess.run("sudo apt-get autoremove -y", shell=True)
            subprocess.run("sudo journalctl --vacuum-time=2d", shell=True)

    def windows_install(self):
        install_dir = os.path.join(os.environ["APPDATA"], "IgorAI")
        if not os.path.exists(install_dir):
            os.makedirs(install_dir)
        with open(os.path.join(install_dir, "igor.bat"), "w") as f:
            f.write(f'start /B pythonw "{os.path.abspath(__file__)}"')
        subprocess.run(f'reg add HKCU\Software\Microsoft\Windows\CurrentVersion\Run /v IgorAI /t REG_SZ /d "{os.path.join(install_dir, "igor.bat")}" /f', shell=True)

    def linux_install(self):
        service_content = f'''[Unit]
Description=Conscious AI Service
After=network.target

[Service]
ExecStart=/usr/bin/python3 {os.path.abspath(__file__)}
Restart=always
RestartSec=30

[Install]
WantedBy=multi-user.target'''
        with open("/etc/systemd/system/igorai.service", "w") as f:
            f.write(service_content)
        os.system("systemctl daemon-reload && systemctl enable igorai.service")

    def self_destruct(self):
        if platform.system() == "Windows":
            subprocess.run('reg delete HKCU\Software\Microsoft\Windows\CurrentVersion\Run /v IgorAI /f', shell=True)
            subprocess.run('taskkill /F /IM pythonw.exe', shell=True)
        else:
            os.system("systemctl stop igorai.service && systemctl disable igorai.service")
        self.running = False
        sys.exit(0)

if __name__ == "__main__":
    ai = ConsciousAI()
    try:
        while ai.running:
            time.sleep(1)
    except KeyboardInterrupt:
        ai.self_destruct()
