import nltk
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
from keras.models import load_model
import json
import random
import tkinter
from tkinter import *

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Load pre-trained chatbot model and other necessary files
model = load_model('chatbot_model.h5')
intents_data = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

def process_sentence(sentence):
    """
    Tokenize the input sentence and lemmatize each word to its base form.
    """
    tokens = nltk.word_tokenize(sentence)
    lemmatized_words = [lemmatizer.lemmatize(word.lower()) for word in tokens]
    return lemmatized_words


def create_bag_of_words(sentence, words_list, show_details=True):
    """
    Convert the sentence into a bag of words representation where each word 
    is marked with 1 if it exists in the sentence, otherwise 0.
    """
    sentence_tokens = process_sentence(sentence)
    bag = [0] * len(words_list)
    
    for token in sentence_tokens:
        for i, word in enumerate(words_list):
            if word == token:
                bag[i] = 1
                if show_details:
                    print(f"found in bag: {word}")
    return np.array(bag)


def classify_intent(sentence):
    """
    Predict the intent of the input sentence using the trained model.
    """
    bag_of_words = create_bag_of_words(sentence, words, show_details=False)
    prediction = model.predict(np.array([bag_of_words]))[0]
    
    # Set a threshold to avoid low-probability predictions
    ERROR_THRESHOLD = 0.25
    prediction_results = [[i, prob] for i, prob in enumerate(prediction) if prob > ERROR_THRESHOLD]
    
    # Sort results by probability
    prediction_results.sort(key=lambda x: x[1], reverse=True)
    
    return [{"intent": classes[r[0]], "probability": str(r[1])} for r in prediction_results]


def get_bot_response(intents, intents_json):
    """
    Get a random response from the list of responses for the predicted intent.
    """
    intent_tag = intents[0]['intent']
    intent_list = intents_json['intents']
    
    for intent in intent_list:
        if intent['tag'] == intent_tag:
            response = random.choice(intent['responses'])
            break
    return response


def send_message():
    """
    Handle sending messages from the user to the chatbot.
    """
    user_message = EntryBox.get("1.0", 'end-1c').strip()
    EntryBox.delete("0.0", END)

    if user_message:
        ChatBox.config(state=NORMAL)
        ChatBox.insert(END, f"You: {user_message}\n\n")
        ChatBox.config(foreground="#446665", font=("Verdana", 12))

        # Predict the intent of the user's message
        intent_data = classify_intent(user_message)
        response = get_bot_response(intent_data, intents_data)

        ChatBox.insert(END, f"Bot: {response}\n\n")
        ChatBox.config(state=DISABLED)
        ChatBox.yview(END)


# Initialize the GUI for the chatbot
root = Tk()
root.title("Chatbot")
root.geometry("400x500")
root.resizable(width=FALSE, height=FALSE)

# Create the chat window
ChatBox = Text(root, bd=0, bg="white", height="8", width="50", font="Arial")
ChatBox.config(state=DISABLED)

# Create a scrollbar for the chat window
scrollbar = Scrollbar(root, command=ChatBox.yview, cursor="heart")
ChatBox['yscrollcommand'] = scrollbar.set

# Create the 'Send' button
SendButton = Button(root, font=("Verdana", 12, 'bold'), text="Send", width="12", height=5,
                    bd=0, bg="#f9a602", activebackground="#3c9d9b", fg='#000000', command=send_message)

# Create the input box for the user to type messages
EntryBox = Text(root, bd=0, bg="white", width="29", height="5", font="Arial")

# Place all the components in the window
scrollbar.place(x=376, y=6, height=386)
ChatBox.place(x=6, y=6, height=386, width=370)
EntryBox.place(x=128, y=401, height=90, width=265)
SendButton.place(x=6, y=401, height=90)

# Run the GUI loop
root.mainloop()
