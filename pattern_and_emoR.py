#pattern

from pattern.en import sentiment

print(sentiment("I am excited."))


"""
from transformers import pipeline

emotion = pipeline('sentiment-analysis', model='arpanghoshal/EmoRoBERTa')


emotion_labels = emotion("I'm excited")

print(emotion_labels)

#1,44 - 1,59 E1

#2,24 - 2,43 B1

#7,17 - 7,46 B2

# 6,28 - 6,40 C1

#3,27 - 3,36 B4

# 4,05 - 4,19 B3
"""