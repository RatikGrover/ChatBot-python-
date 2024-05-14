# -*- coding: utf-8 -*-


import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

from nltk.stem import WordNetLemmatizer
lemmatizer=WordNetLemmatizer()
import pickle
import numpy as np
from flask_cors import CORS


from tensorflow.keras.models import load_model
model = load_model('chatbot_anik.h5')
import json
import random
intents=json.loads(open('intents.json',encoding="utf8").read())
words=pickle.load(open('words.pkl','rb'))
classes=pickle.load(open('classes.pkl','rb'))

#clean_up
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

#bow
def bow(sentence, words, show_details=True):
    #tokenize the pattern
    sentence_words= clean_up_sentence(sentence)
    bag = [0]*len(words)
    
    
    for s in sentence_words:
        for i,w in enumerate(words):
            if(w==s):
                
                bag[i]=1
                if show_details:
                    print("Found in bag: %s" % w)
    
    return(np.array(bag))
    
#predict_class
def predict_class(sentence,model):
    p = bow(sentence, words, show_details=False)
    res=model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    
    #sort by strength
    results.sort(key=lambda x: x[1], reverse=True)
    result_list = []
    
    
    
    for r in results:
        result_list.append({"intent": classes[r[0]],"probability": str(r[1])})
        
    return result_list
    


#getResponse
def getResponse(ints, intents_json):
    tag=ints[0]['intent']
    list_of_intents=intents_json['intents']
    
    for i in list_of_intents:
        if (i['tag']==tag):
            result = random.choice(i['responses'])
            break
    return result
    
    
#response
def chatbot_response(msg):
    ints = predict_class(msg,model)
    try:
        res = getResponse(ints, intents)
    except:
        print("Exception Error")
        res="I cannot answer that."
    return res
    

#flask code
from flask import Flask, jsonify , render_template  

app = Flask(__name__)
CORS(app)


@app.route('/')
def hello():
    return render_template('index.html')


def decrypt(msg):
    #i/p what+is+machine
    #o/p what is machine
    #swap + with space
    string = msg
    new_string=string.replace("+"," ")
    
    return new_string

@app.route("/query/<sentence>")
def query_chatbot(sentence):
    
    #decrypt user mssg
    dec_msg= decrypt(sentence)
    
    response = chatbot_response(dec_msg)
    
    json_obj = jsonify({"top" : {"res" :response}})
    
    
    return json_obj

app.run()


