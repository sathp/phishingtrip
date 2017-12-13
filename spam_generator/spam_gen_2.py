import numpy as np
import tensorflow as tf

from get_lines_text import *
from get_emails import get_clinton_emails

#define nessecary vars
num_lstm_units = 250
batch_size = 100
num_chars = None
letter_dic = None
reverse_letter_dic = None
max_time_steps = 100
input_string = "cucumber"
len_str = len(input_string)
num_layers = 2

#text, num_chars, letter_dic, reverse_letter_dic = get_moby_dick_batches(batch_size, max_time_steps)
text, num_chars, letter_dic, reverse_letter_dic = get_txt_batches("varney.txt", batch_size, max_time_steps)

print(letter_dic)

#more vars
num_batches = len(text)

#create LSTM, state for LSTM, and placeholders for feeding
#lstm_cell = tf.contrib.rnn.LSTMCell(num_lstm_units) #create our lstm with this number of units
lstm_cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.LSTMCell(num_lstm_units) for size in [num_lstm_units, num_lstm_units]])
lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=0.9) #add dropout for magical reasons
grad_descent_obj = tf.train.AdamOptimizer(.005, name="LSTM_Gradient_Descent") 


print("num_lstm_units = ", num_lstm_units)
print("batch_size = ", batch_size)
print("num_chars = ", num_chars)
print("max_time_steps = ", max_time_steps)
print("num_batches = ", num_batches)


#to be used in final output calculation
probability_w_z = tf.random_uniform([num_lstm_units, num_chars], dtype=tf.float64, name="Probability_weights") #when multiplied by the output, creates the distribution for every character
probability_b_z = tf.random_uniform([1, num_chars], dtype=tf.float64, name="Probability_biases") #biases for above weights
probability_w = tf.Variable(initial_value=probability_w_z, name="Probability_weights", dtype=tf.float64, trainable=True) #when multiplied by the output, creates the distribution for every character
probability_b = tf.Variable(initial_value=probability_b_z, name="Probability_biases", dtype=tf.float64, trainable=True) #biases for above weights

#let's create a graph! (for training)

with tf.variable_scope("train"):
	#placeholders for the feeding dictionary
	input_data_p = tf.placeholder(tf.float64, shape=[max_time_steps, batch_size, num_chars], name="training_labels")

	p_w = probability_w
	p_b = tf.stack([tf.reshape(probability_b, [-1])]*batch_size)
	
	input_data = tf.transpose(input_data_p, [1,0,2])
	
	#train_labels = tf.stack(tf.unstack(input_data_p)) #add buffer to front, remove back (deprecated, keeping just in case)
	train_labels = tf.stack((tf.unstack(input_data_p) + [tf.zeros([batch_size, num_chars], tf.float64)])[1:]) #add buffer to front, remove back

	print("Train_labels dim is ", train_labels.shape)
	
	train_labels = tf.transpose(train_labels, [0,1,2])
	
	
	#compare calculated prob with next time_step
	
	print("Setting up training")
	
	#run through the batches in the NN
	outputs, final_state = tf.nn.dynamic_rnn(cell=lstm_cell, inputs=input_data, dtype=tf.float64, time_major=False)
	
	logits = [tf.matmul(output_at_time_step, p_w) + p_b for output_at_time_step in tf.unstack(tf.transpose(outputs, [1,0,2]))] #the non softmax probabilities, used in error function
	inter = tf.stack(logits)
	probabilities = [tf.nn.softmax(logit) for logit in logits] #the softmax probs, to be returned
	
	probabilities = tf.unstack(tf.transpose(tf.stack(probabilities), [1,0,2]))
	
	logits = tf.unstack(tf.transpose(tf.stack(logits), [0,1,2]))
	train_labels = tf.unstack(tf.transpose(train_labels, [0,1,2]))
	
	lossess = [tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logit) for logit, labels in zip(logits, train_labels)]
	loss = tf.reduce_mean(lossess)
	
	minimizer = grad_descent_obj.minimize(loss, name="Minimizer")

#running graph
with tf.variable_scope("run"):
	batch_of_1 = 1
	#placeholders for the feeding dictionary
	input_data_p_r = tf.placeholder(tf.float64, shape=[len_str, batch_of_1, num_chars], name="running_lables")

	p_w = probability_w
	#p_b = probability_b
	p_b = tf.stack([tf.reshape(probability_b, [-1])]*batch_of_1)
	
	input_data = input_data_p_r
	
	#compare calculated prob with next time_step
	
	print("Setting up running")
	
	#run through the batches in the NN
	#outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, input_data, dtype=tf.float64)
	
	#run through the given word
	
	#state = (hidden state, current state) as by api docs
	state = lstm_cell.zero_state(batch_of_1, dtype=tf.float64)

	print("INP ", input_data)
	print("p_w ", p_w)
	print("p_b ", p_b)
	
	for step in tf.unstack(input_data):
		output, state = lstm_cell(step, state)
	
	temp_logit = tf.nn.softmax(tf.matmul(output, p_w) + p_b)
	output = temp_logit
	
	new_probabilities = []
	new_probabilities.append(temp_logit)
	
	for _ in range(max_time_steps - len(input_string)):
		output, state = lstm_cell(output, state)
		
		#compute extra stuff in the for loop
		temp_logit = tf.nn.softmax(tf.matmul(output, p_w) + p_b)
		output = tf.one_hot([tf.argmax(tf.reshape(temp_logit, [-1]))], num_chars, dtype=tf.float64)
		tf.Print(temp_logit, [temp_logit])
		new_probabilities.append(temp_logit)
	
	probabilities_r = tf.transpose(tf.stack(new_probabilities), [1,0,2])
	
	final_state_r = state

#now run the session on the graph

#create the session
sess = tf.Session()
saver = tf.train.Saver()

#initialize all the variables
init_operation = tf.global_variables_initializer()

#get the losses for the current batch and the total loss
total_loss = 0.0

with sess.as_default():
	sess.run(init_operation)
	#set up a non tensor rep of the state
	numpy_state = (np.zeros([batch_size, num_lstm_units], np.float64), np.zeros([batch_size, num_lstm_units], np.float64))
	
	writer = tf.summary.FileWriter("C:/Users/Krist/Desktop/spam", sess.graph)
	
	#format a string to be completed
	my_string = input_string
	my_string = np.asarray([[get_arr_with_n_as_one(letter_dic[char], num_chars)] for char in my_string])
	
	print(np.asarray(my_string).shape)
	
	i = 0
	
	for runs in range(3):
		for lines in text:
			l = np.transpose(np.asarray(lines), [1,0,2])
			numpy_state, current_loss, p, _, op, interm = sess.run([final_state, loss, probabilities, minimizer, outputs, inter], feed_dict={input_data_p: l})
			print("Line ", i, " Loss ", current_loss)
			i += 1
					
			chars_o = ""
			
			for k in range(0, 5):
				for j in range(len(p[k])):
					chars_o += str(reverse_letter_dic[np.argmax(p[k][j])])
				print(chars_o)
				chars_o = ""		
		
	print("================================================")
	saver.save(sess, "C:/Users/Krist/Desktop/spam/varney")
	p, s = sess.run([probabilities_r, final_state_r], feed_dict={input_data_p_r: my_string})
	
	chars_o = ""

	print(np.asarray(p).shape)
	
	p = p[0]
	
	print(np.asarray(p).shape)
	
	for j in p:
		chars_o += str(reverse_letter_dic[np.argmax(j)])
	print(input_string+chars_o)
		
	
sess.close()
