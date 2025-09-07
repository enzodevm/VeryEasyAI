import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from .nn import IntentNN
from .search import web_search
from .knowledge import knowledge_base

labels = ["saudacao", "conhecimento", "busca"]
vocab = ["oi", "olá", "quem", "capital", "qual", "pesquise", "internet", "google", "defina", "explique"]

def frase_para_vetor(frase):
    tokens = frase.lower().split()
    return [1 if w in tokens else 0 for w in vocab]

net = IntentNN(input_size=len(vocab), hidden_size=8, output_size=len(labels))

train_data = [
    ("oi", "saudacao"),
    ("olá", "saudacao"),
    ("quem é você", "conhecimento"),
    ("qual a capital do brasil", "conhecimento"),
    ("explique inteligência artificial", "conhecimento"),
    ("pesquise no google IA", "busca"),
    ("me diga na internet sobre python", "busca"),
]

for ep in range(2000):
    for frase, label in train_data:
        x = frase_para_vetor(frase)
        y = [1 if l == label else 0 for l in labels]
        net.train(x, y)

class ChatIA:
    def __init__(self, nome="VeryEasyAI"):
        self.nome = nome

    def responder(self, pergunta):
        x = frase_para_vetor(pergunta)
        label, _ = net.predict(x, labels)

        if label == "saudacao":
            return "Oi! Eu sou a " + self.nome + "!"

        elif label == "conhecimento":
            for k, v in knowledge_base.items():
                if k in pergunta.lower():
                    return v
            return "Não sei ainda, mas estou aprendendo!"

        elif label == "busca":
            resultado = web_search(pergunta)
            return f"Pesquisei e encontrei: {resultado}"

        else:
            return "Ainda não entendi sua pergunta..."
