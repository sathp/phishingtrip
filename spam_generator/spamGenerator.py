import numpy as np
import tensorflow as tf

#create the batches

num_chars = 128

def get_moby_dick_lines():
	book = open('moby10b.txt', 'r')

	text = book.readlines()

	for i in range(len(text)):
		text[i] = [ord(j) for j in text[i]]
	
	#normalizing text

	longest_line = 0

	for i in range(len(text)):
		if(len(text[i]) > longest_line):
			longest_line = len(text[i])

	num_features = 1	

	new_text = np.zeros([len(text), longest_line, num_chars], np.float64)		
		
	for i in range(len(text)):
		for j in range(longest_line):
			if(len(text[i]) > j):
				new_text[i][j][text[i][j]] = text[i][j]
			else: 
				new_text[i][j] = 0
		
	text = new_text
	
	return text
	
#done setting up the text

little_size = 100

text = get_moby_dick_lines() #permute so it's easier to use
ttext = np.zeros([len(text)//little_size, little_size, len(text[0]), num_chars], np.float64)

print(text.shape)
print(ttext.shape)

for i in range(len(ttext) - 1):
	ttext[i] = text[i*little_size:(i+1)*little_size]

text = ttext.transpose([0, 2, 1, 3])
#text = ttext

num_units = 1000 #one thousand neurons?
batch_size = little_size #get length of dataset; i.e. the number of lines
max_time_steps = len(text[0])

print("Number of units is " + str(num_units))
print("Batch size is " + str(batch_size))
print("Time steps is " + str(max_time_steps))

#quit()

lstm_cell = tf.contrib.rnn.LSTMCell(num_units) #create our lstm with this number of units

hidden_state = tf.zeros([batch_size, num_units], tf.float64)
current_state = tf.zeros([batch_size, num_units], tf.float64)
state = (hidden_state, current_state)

loss = 0.0

#text = tf.convert_to_tensor(text)

probability_w = tf.zeros([num_units, 1], tf.float64) #when multiplied by the output, creates the distribution for every character
probability_b = tf.zeros([1, num_chars], tf.float64) #biases for above weights
probabilities = []

words = tf.placeholder(tf.float64, [max_time_steps, batch_size, num_chars], name="words_p_h")
initial_state = state
#initial_state = state = tf.zeros([batch_size, num_units])
loss = 0.0
current_loss = loss

#print(text[0])

#open a new session
#initialize the session and variables
sess = tf.Session()
print("================================RAN SESS=================================")
init_op = tf.global_variables_initializer()
sess.run(init_op)
print("================================RAN INIT=================================")

#code block for creating the graph
for i in range(max_time_steps):
	output, state = lstm_cell(words[i], state)
	
	out_probs = tf.nn.softmax(tf.matmul(output, probability_w) + probability_b)
	
	probabilities.append(out_probs)
	
loss += tf.losses.softmax_cross_entropy(probabilities, words)	
	
final_state = state

#print(probabilities)

numpy_state = (initial_state[0].eval(session=sess), initial_state[1].eval(session=sess))

total_loss = loss

#run the rnn on our set number of time steps
for lines in text:
	print("========================================================================")
	print(tf.convert_to_tensor(lines))
	numpy_state, current_loss = sess.run([final_state, loss], feed_dict={initial_state: numpy_state, words: lines})
	total_loss += current_loss