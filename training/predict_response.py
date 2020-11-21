words = pickle.load(open('words.pkl','rb'))
labels = pickle.load(open('labels.pkl','rb'))

with open('chatbot_intents.json') as file:
        data = json.load(file,strict=False)

def input_bag(sen,words):
    bag=[0]*len(words)
    wrds=nltk.word_tokenize(sen)
    wrds=[lemm.lemmatize(w.lower()) for w in wrds]

    for s in wrds:
        for i,j in enumerate(words):
            if j==s:
                bag[i]=1
    return (np.array(bag))


def response():
    print('Nav: Hey welcome. I am Nav, the SteerSecure chatbot. Type exit to leave.')
    while True:
        inputs=input('You:')
        if inputs.lower()=='exit':
            break
        x=input_bag(inputs,words)
        res=model.predict(np.array([x]))[0]
        #print(res)
        pred_index=np.argmax(res)
        tag=labels[pred_index]
        #print(tag)
        for t in data['intents']:
            if t['tag']==tag:
                resp=t['responses']
                break
        print("Nav: "+random.choice(resp))

response()
