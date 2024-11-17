import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
import random
import nltk
from nltk.stem import WordNetLemmatizer
import json
import pickle

# Initialize lemmatizer and variables
lemmatizer = WordNetLemmatizer()
words = []
classes = []
documents = []
ignore_symbols = ['!', '?', ',', '.']

# Load and parse the intents JSON file
with open('intents.json') as file:
    intents_data = json.load(file)

# Tokenize and process the patterns from the intents
for intent in intents_data['intents']:
    for pattern in intent['patterns']:
        tokens = nltk.word_tokenize(pattern)
        words.extend(tokens)
        documents.append((tokens, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

print(documents)

# Preprocess the words: lemmatize and remove unwanted characters
words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_symbols]
words = sorted(list(set(words)))

# Sort the classes for consistency
classes = sorted(list(set(classes)))

# Output some statistics
print(f"Total documents: {len(documents)}")
print(f"Total classes: {len(classes)} - {classes}")
print(f"Unique lemmatized words: {len(words)} - {words}")

# Save the processed words and classes to disk
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Prepare the training data
training_sentences = []
training_labels = []

# Prepare an empty array for the output labels
empty_output = [0] * len(classes)

# Create the "bag of words" for each document
for doc in documents:
    bag_of_words = []
    pattern_words = doc[0]
    # Lemmatize the words in the pattern
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    
    # Create a "bag of words" where 1 represents the presence of a word in the pattern
    for word in words:
        bag_of_words.append(1 if word in pattern_words else 0)
    
    # Create the output row, which is a binary vector
    output_row = list(empty_output)
    output_row[classes.index(doc[1])] = 1
    
    training_sentences.append(bag_of_words)
    training_labels.append(output_row)

# Shuffle the data and convert to a NumPy array
training_data = list(zip(training_sentences, training_labels))
random.shuffle(training_data)
training_sentences, training_labels = zip(*training_data)

train_x = np.array(training_sentences)
train_y = np.array(training_labels)

print("Training data is prepared")

# Build the neural network model
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile the model using SGD optimizer
sgd_optimizer = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd_optimizer, metrics=['accuracy'])

# Train the model and save it
history = model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.h5')

print("Model training complete and saved as chatbot_model.h5")
