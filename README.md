# ChatBot-python-
Creating a simple Python chatbot using natural language processing and deep learning.

TOOLS USED
- nltk libarary
- keras
- flask
  
MAIN STEPS

- getting the right data - we are storing the questions answers in an JSON format in a file - intents.json
- cleaning, lematizing and everthing that is necesaary for the initial setup and saving 2 pickle files as - words.pkl and classes.pkl
- and creating Bag of Words
- Create model - 3 layers. First layer 128 neurons, second layer 64 neurons and 3rd output layer contains number of neurons
- and in the app.py we are creating a flask backend - restful api to connect with the index.html which contains the javascript
- it gives a json object as a response
WEBSITE - Takes the user input from js and displays the result in a ui

