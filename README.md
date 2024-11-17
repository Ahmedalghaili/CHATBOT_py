# Chatbot Project

This project demonstrates how to build a simple chatbot using Natural Language Processing (NLP) and deep learning techniques. The chatbot uses a trained model to classify user inputs and generate appropriate responses. It features a graphical user interface (GUI) where users can interact with the chatbot.

## How do Chatbots Work?

Chatbots are intelligent software systems designed to simulate human-like conversations. Powered by Natural Language Processing (NLP), chatbots interpret and respond to human language. NLP is divided into two primary components:

- **NLU (Natural Language Understanding)**: The machine's ability to understand human language, such as English.
- **NLG (Natural Language Generation)**: The machine's ability to generate text resembling human-written sentences.

When you ask a chatbot a question, such as "Hey, what’s on the news today?", the bot processes the input by identifying the **intent** and **entity**. In this case:
- The **intent** is `get_news` (the action the user wants to perform).
- The **entity** is `today` (the specific detail about the action).

The chatbot uses machine learning models to understand the user's question by extracting these components.
## Prerequisites
To implement the chatbot, we will be using Keras, which is a Deep Learning library, NLTK, which is a Natural Language Processing toolkit, and some helpful libraries. Run the below command to make sure all the libraries are installed:
```bash
pip install tensorflow keras pickle nltk 

## Running the Chatbot

To run the chatbot, first train the model by running the following command:

```bash
python train_chatbot.py
