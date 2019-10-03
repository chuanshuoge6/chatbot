from django.shortcuts import render
import nltk
nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
import numpy
import tflearn
import tensorflow
import random
import json
import pickle

def index(request):
    with open('intents.json') as file:
        data = json.load(file)
    try:
        with open('data.pickle', 'rb') as f:
            words, labels, training, output = pickle.load()
    except:
        # print(data['intents'])
        words = []
        labels = []
        docs_pattern = []
        docs_tag = []

        for intent in data['intents']:
            for pattern in intent['patterns']:
                # get all the words in pattern
                wds = nltk.word_tokenize(pattern)
                # copy word list to words[]
                words.extend(wds)
                # used for input of neural network
                docs_pattern.append(wds)
                # copy cosresponding category, used as output of neural network
                docs_tag.append(intent['tag'])
                # copy categories into labels[]
                if (intent['tag'] not in labels):
                    labels.append(intent['tag'])

        # format words so that they don't contain symbols...
        words = [stemmer.stem(w.lower()) for w in words if w != '?']
        # remove duplicate words, sort
        words = sorted(list(set(words)))
        labels = sorted(labels)
        # one hot encoding, if input entry word exists in dictionary, mark 1 else 0
        training = []
        output = []
        # output default [0,0,0...] length = # of categories
        out_empty = [0 for _ in range(len(labels))]
        # print(words, labels, out_empty)
        # print(docs_pattern, docs_tag)

        for n, doc in enumerate(docs_pattern):
            bag = []
            # clean input words
            wd_ = [stemmer.stem(w) for w in doc]
            # one hot encoding, training input, look through dictionary,
            # mark the words exist in user entry as 1, the rest 0
            for w in words:
                if w in wd_:
                    bag.append(1)
                else:
                    bag.append(0)

            output_ = out_empty[:]
            # the training output, mark 1 category, the rest 0
            output_[labels.index(docs_tag[n])] = 1

            training.append(bag)
            output.append(output_)

        training = numpy.array(training)
        output = numpy.array(output)
        # print(training, output)

        with open('data.pickle', 'wb') as f:
            pickle.dump((words, labels, training, output), f)

    tensorflow.reset_default_graph()

    # input shape
    net = tflearn.input_data(shape=[None, len(training[0])])
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, len(output[0]), activation='softmax')
    net = tflearn.regression(net)

    model = tflearn.DNN(net)

    try:
        model.load('model.tflearn')
    except:
        model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
        model.save('model.tflearn')

    def bag_of_words(s, words):
        bag1 = [0 for _ in range(len(words))]
        s_words = nltk.word_tokenize(s)
        s_words = [stemmer.stem(word.lower()) for word in s_words]

        for se in s_words:
            for i, w in enumerate(words):
                if w == se:
                    bag1[i] = 1

        return numpy.array(bag1)

    def chat(inp):
        results = model.predict([bag_of_words(inp, words)])[0]
        # print(results)
        results_index = numpy.argmax(results)
        tag = labels[results_index]
        # print(tag)
        #print(results, results_index)
        # response if more than 95% sure
        if results[results_index] > 0.95:
            for tg in data['intents']:
                if tg['tag'] == tag:
                    responses = tg['responses']
                    ai_reply = random.choice(responses)
        else:
            ai_reply = 'I didn\'t get that, try again'
        return ai_reply, results, tag, labels

    ai_reply_, ai_probability, ai_category, ai_categories = 'a', 'b', 'c', 'd'
    try:
        user_input = request.POST['userchat']
        ai_reply_, ai_probability, ai_category, ai_categories = chat(user_input)
        #print(ai_reply_, ai_probability, ai_category, ai_categories)
    except Exception as e:
        print(type(e))
        pass

    return render(request, 'index.html',
                  {'reply': ai_reply_,
                   'probability': ai_probability,
                   'category': ai_category,
                   'categories': ai_categories,
                   'cheatSheet': data})