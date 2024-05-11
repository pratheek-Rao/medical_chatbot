import streamlit as st
import nltk
import numpy as np
import tflearn
import tensorflow as tf
import random
import json
from nltk.stem.lancaster import LancasterStemmer

# Download NLTK data
nltk.download('punkt')

# Load intents file
with open('intents.json') as file:
    data = json.load(file)

stemmer = LancasterStemmer()

# Extract data from intents file
words = []
labels = []
docs_x = []
docs_y = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        docs_x.append(wrds)
        docs_y.append(intent["tag"])
        
    if intent["tag"] not in labels:
        labels.append(intent["tag"])    

words = [stemmer.stem(w.lower()) for w in words if w != "?"]
words = sorted(list(set(words)))
labels = sorted(labels)

# Load trained model
net = tflearn.input_data(shape=[None, len(words)])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(labels), activation='softmax')
net = tflearn.regression(net)

model = tflearn.DNN(net)
model.load("model.tflearn")

def chatbot_response(sentence):
    results = model.predict([bag_of_words(sentence, words)])
    results_index = np.argmax(results)
    tag = labels[results_index]

    for tg in data["intents"]:
        if tg['tag'] == tag:
            responses = tg['responses']

    return random.choice(responses)

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
            
    return np.array(bag)

# Streamlit App
def main():
    st.title("Chatbot Demo")

    st.markdown("""
    Welcome to the Chatbot Demo! Type a message and press Enter to chat with the bot.
    """)

    with st.form(key='chat_form'):
        user_input = st.text_input("You:")
        submit_button = st.form_submit_button(label='Send')

        if submit_button:
            bot_response = chatbot_response(user_input)
            st.text_area("Bot:", value=bot_response, height=100, max_chars=None, key=None)

if __name__ == '__main__':
    main()
