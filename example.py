
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from veryeasyai.hybrid import ChatIA

bot = ChatIA()

print(bot.responder("oi"))
print(bot.responder("qual a capital do brasil"))
print(bot.responder("explique inteligência artificial"))
print(bot.responder("pesquise no google programação python"))
