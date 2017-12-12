import tensorflow as tf
import pickle
import numpy as np


with open('data/data.pickle', 'rb') as f:
        tr_data, tr_label, tst_data, tst_label = pickle.load(f)

n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 2
n_batches = 10

x = tf.placeholder('float', [None, len(tr_data[0])])
y = tf.placeholder('float')

weights_1 = tf.Variable(tf.random_normal([len(tr_data[0]), n_nodes_hl1]), name='w1')
biases_1 = tf.Variable(tf.random_normal([n_nodes_hl1]), name='b1')
weights_2 = tf.Variable(tf.random_normal([n_nodes_hl1,n_nodes_hl2]), name='w2')
biases_2 = tf.Variable(tf.random_normal([n_nodes_hl2]), name='b2')
weights_3 = tf.Variable(tf.random_normal([n_nodes_hl2,n_nodes_hl3]), name='w3')
biases_3 = tf.Variable(tf.random_normal([n_nodes_hl3]), name='b3')
weights_out = tf.Variable(tf.random_normal([n_nodes_hl3,n_classes]), name='wOut')
biases_out = tf.Variable(tf.random_normal([n_classes]), name='bOut')

# Neural Network Model
def neural_network(data):
    print("longest word length: ", len(tr_data[0]))


    hd_layer1 = {'weights': weights_1,
                 'biases': biases_1}
    hd_layer2 = {'weights': weights_2,
                 'biases': biases_2}
    hd_layer3 = {'weights': weights_3,
                 'biases': biases_3}
    output    = {'weights': weights_out,
                 'biases': biases_out}

    # (input_data*weight) + biases
    l1  = tf.add(tf.matmul(data,hd_layer1['weights']),hd_layer1['biases'])
    l1 = tf.nn.relu(l1) # activation function - threshold function

    l2  = tf.add(tf.matmul(l1,hd_layer2['weights']),hd_layer2['biases'])
    l2 = tf.nn.relu(l2)

    l3  = tf.add(tf.matmul(l2,hd_layer3['weights']),hd_layer3['biases'])
    l3 = tf.nn.relu(l3)

    output  = tf.matmul(l3,output['weights']) + output['biases']

    return output

# Code to train the neural netowrk
def train_neural_network(x):
    # gives predicted value based on input data
    prediction = neural_network(x)
    # gives cost function (difference) between predicted and actual output
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))
    # Stochastic gradient descent
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    # no. of cycles of feed forward + backprop
    hm_epochs= 10

    saver = tf.train.Saver([weights_1, biases_1, weights_2, biases_2, weights_3, biases_3, weights_out, biases_out])
    # Run tensorFlow
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(hm_epochs):
            epoch_loss = 0
            for i in range(0,len(tr_data),n_batches):
                x_batch = np.array(tr_data[i:i+100])
                y_batch = np.array(tr_label[i:i+100])
                _, c= sess.run([optimizer,cost],feed_dict={x:x_batch,y:y_batch})
                epoch_loss += c
            print('Epoch: {} completed out of:  {} loss: {}'.format(epoch,hm_epochs,epoch_loss))
            # if (epoch_loss == 0):
            #     break
        saver.save(sess, 'my_model.ckpt', global_step=1000)

        correct = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct,'float'))
        print(accuracy.eval({x:tst_data,y:tst_label}))

        # weights_1_val, biases_1_val, weights_2_val, biases_2_val, weights_3_val, biases_3_val, weights_out_val, biases_out_val = sess.run([weights_1, biases_1, weights_2, biases_2, weights_3, biases_3, weights_out, biases_out])
        # np.savetxt('weights1.csv', weights_1_val, delimiter=',')
        # np.savetxt('weights2.csv', weights_2_val, delimiter=',')
        # np.savetxt('weights3.csv', weights_3_val, delimiter=',')
        # np.savetxt('weightsOut.csv', weights_out_val, delimiter=',')
        # np.savetxt('biases1.csv', biases_1_val, delimiter=',')
        # np.savetxt('biases2.csv', biases_2_val, delimiter=',')
        # np.savetxt('biases3.csv', biases_3_val, delimiter=',')
        # np.savetxt('biasesOut.csv', biases_out_val, delimiter=',')

train_neural_network(x)
