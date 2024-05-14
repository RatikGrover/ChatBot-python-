import numpy as np
import keras
import nltk
from nltk.stem import WordNetLemmatizer
import pickle
import json
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import random
import tensorflow as tf
from tensorflow.keras.optimizers import SGD

import matplotlib.pyplot as plt

nltk.download('omw-1.4')

# Initialize lists
words = []
classes = []
documents = []
ignore = ['?', '!', ',', "'s"]

# Load intents from JSON file
with open('intents.json', 'r', encoding='utf-8') as file:
    intents = json.load(file)

lemmatizer = WordNetLemmatizer()

# Process intents and populate words, classes, documents
for intent in intents['intents']:
    for pattern in intent['patterns']:
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        documents.append((w, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize words and filter out ignored characters
words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

# Save words and classes to pickle files
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Prepare training data
training = []

for doc in documents:
    bag = []
    pattern = doc[0]
    pattern = [lemmatizer.lemmatize(word.lower()) for word in pattern]

    for word in words:
        bag.append(1) if word in pattern else bag.append(0)

    output_row = [0] * len(classes)
    output_row[classes.index(doc[1])] = 1

    training.append((bag, output_row))

# Shuffle training data
random.shuffle(training)

# Extract X_train (features) and y_train (labels) from training
train_x = np.array([sample[0] for sample in training])  # Features
train_y = np.array([sample[1] for sample in training])  # Labels



# Model creation
model=Sequential()
model.add(Dense(128,activation='relu',input_shape=(len(train_x[0]),)))
model.add(Dropout(0.5))
model.add(Dense(64,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]),activation='softmax'))

#Adjust learning rate and optimizer
adam=keras.optimizers.Adam(0.001)
model.compile(optimizer=adam,loss='categorical_crossentropy',metrics=['accuracy'])
weights=model.fit(train_x,train_y,epochs=200,batch_size=10,verbose=1) 

#save   
model.save('chatbot_anik.h5',weights)









