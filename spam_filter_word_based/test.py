import os
import tensorflow as tf
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import random
import pickle
from collections import Counter

path = "data/dictionary_processing/"

with open('data/data.pickle', 'rb') as f:
    tr_data, tr_label, tst_data, tst_label = pickle.load(f)

x = tf.placeholder('float', [None, 2518])
y = tf.placeholder('float')

str_red = WordNetLemmatizer()

def neural_network(data, w1, w2, w3, wOut, b1, b2, b3, bOut):

    hd_layer1 = {'weights': w1,
                 'biases': b1}
    hd_layer2 = {'weights': w2,
                 'biases': b2}
    hd_layer3 = {'weights': w3,
                 'biases': b3}
    output    = {'weights': wOut,
                 'biases': bOut}

    # (input_data*weight) + biases
    l1  = tf.add(tf.matmul(data,hd_layer1['weights']),hd_layer1['biases'])
    l1 = tf.nn.relu(l1) # activation function - threshold function

    l2  = tf.add(tf.matmul(l1,hd_layer2['weights']),hd_layer2['biases'])
    l2 = tf.nn.relu(l2)

    l3  = tf.add(tf.matmul(l2,hd_layer3['weights']),hd_layer3['biases'])
    l3 = tf.nn.relu(l3)

    output  = tf.matmul(l3,output['weights']) + output['biases']

    return output

def evaluate():
    with tf.Session() as sess:

        new_saver = tf.train.import_meta_graph('my_model.ckpt-1000.meta')
        new_saver.restore(sess, tf.train.latest_checkpoint('./'))
        graph = tf.get_default_graph()
        # file_op = tf.summary.FileWriter("tmp/", graph=graph)
        w1 = graph.get_tensor_by_name("w1:0")
        w2 = graph.get_tensor_by_name("w2:0")
        w3 = graph.get_tensor_by_name("w3:0")
        wOut = graph.get_tensor_by_name("wOut:0")
        b1 = graph.get_tensor_by_name("b1:0")
        b2 = graph.get_tensor_by_name("b2:0")
        b3 = graph.get_tensor_by_name("b3:0")
        bOut = graph.get_tensor_by_name("bOut:0")
        prediction = neural_network(x, w1, w2, w3, wOut, b1, b2, b3, bOut)
        # correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        # accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        # print(accuracy.eval({x: tst_data[100:200], y: tst_label[100:200]}))
        output = prediction.eval(feed_dict={x : createFeatures()})[0]
        print("raw output: ", output)
        if output[1] > output[0]:
            print("The input email is a spam.")
        else:
            print("The input email is not a spam.")

def createFeatures():
    dictionary = []
    files = os.listdir(path)

    for file in files:
        with open(path + file, 'r') as f:
            contents = f.read()
            words = word_tokenize(contents)
            dictionary += words

    dictionary = [str_red.lemmatize(i) for i in dictionary]
    dictionary = Counter(dictionary)
    dictCount = []
    for w in dictionary:
        if dictionary[w] > 14:
            dictCount.append(w)

    with open('input.txt', 'r') as f:
        contents = f.read()
        words = word_tokenize(contents)
        words = [str_red.lemmatize(i) for i in words]
        ft = np.zeros(len(dictCount))
        for word in words:
            if word in dictCount:
                idx = dictCount.index(word)
                ft[idx] += 1
    return [list(ft)]

evaluate()