# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 04:13:31 2019

@author: Devdarshan
"""

import speech_recognition as sr
r = sr.Recognizer()
mic = sr.Microphone()
with mic as source:
    r.adjust_for_ambient_noise(source)
    audio = r.listen(source)
t=[]
t = r.recognize_google(audio)
print(t)