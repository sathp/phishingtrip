import numpy as np

def get_moby_dick_lines(): #deprecated
	book = open('moby10b.txt', 'r')
	
	letter_dic = {}
	reverse_letter_dic = {}

	text = book.readlines()

	new_text = []
	
	id = 0
	
	for i in range(len(text)):
		if(text[i][0] == '\n' or text[i][0] == ''):
			pass
		else:
			new_text.append(text[i].lower())
			for char in text[i].lower():
				if(char in letter_dic):
					pass
				else:
					letter_dic[char] = id
					reverse_letter_dic[id] = char
					id += 1
					
	num_chars = id
	
	text = new_text
	
	for i in range(len(text)):
		text[i] = [letter_dic[j] for j in text[i]]
	
	#normalizing text

	longest_line = 0

	for i in range(len(text)):
		if(len(text[i]) > longest_line):
			longest_line = len(text[i])

	new_text = np.zeros([len(text), longest_line, num_chars], np.float64)		
		
	for i in range(len(text)):
		for j in range(longest_line):
			if(len(text[i]) > j):
				new_text[i][j][text[i][j]] = 1.0
			else: 
				new_text[i][j] = np.zeros([num_chars], np.float64)
		
	text = new_text
	
	return text, num_chars, letter_dic, reverse_letter_dic
	
def get_arr_with_n_as_one(n, leng):
	li = [0]*leng
	li[n] = 1.0
	return li
	
def get_moby_dick_batches(batch_size, sample_length):
	book = open('moby10b.txt', 'r')
	
	letter_dic = {}
	reverse_letter_dic = {}
	
	text = book.read().lower()
	
	id = 0
	
	for char in text:
		if(char in letter_dic):
			pass
		else:
			letter_dic[char] = id
			reverse_letter_dic[id] = char
			id += 1
					
	num_chars = id
	
	ttext = []
	
	for place in range(len(text)//sample_length):
		ttext.append([get_arr_with_n_as_one(letter_dic[char], num_chars) for char in text[place*sample_length:(place+1)*sample_length]])
	
	text = ttext
	
	ttext = []
	
	for batch in range(len(text)//batch_size):
		ttext.append(text[batch*batch_size:(batch+1)*batch_size])
	
	text = ttext
	
	print(len(text), len(text[0]), len(text[0][0]))
	
	return text, num_chars, letter_dic, reverse_letter_dic

def get_txt_batches(file_str, batch_size, sample_length):
	book = open(file_str, 'r')
	
	letter_dic = {}
	reverse_letter_dic = {}
	
	text = book.read().lower()
	
	id = 0
	
	for char in text:
		if(char in letter_dic):
			pass
		else:
			letter_dic[char] = id
			reverse_letter_dic[id] = char
			id += 1
					
	num_chars = id
	
	ttext = []
	
	for place in range(len(text)//sample_length):
		ttext.append([get_arr_with_n_as_one(letter_dic[char], num_chars) for char in text[place*sample_length:(place+1)*sample_length]])
	
	text = ttext
	
	ttext = []
	
	for batch in range(len(text)//batch_size):
		ttext.append(text[batch*batch_size:(batch+1)*batch_size])
	
	text = ttext
	
	print(len(text), len(text[0]), len(text[0][0]))
	
	return text, num_chars, letter_dic, reverse_letter_dic
		
	
def get_odyssey_lines(): #deprecated
	book = open('odyssey.txt', 'r')

	text = book.readlines()

	new_text = []
	
	for i in range(len(text)):
		if(text[i][0] == '\n' or text[i][0] == ''):
			pass
		else:
			new_text.append(text[i])
	
	text = new_text
	
	for i in range(len(text)):
		text[i] = [ord(j) for j in text[i]]
	
	#normalizing text

	longest_line = 0

	for i in range(len(text)):
		if(len(text[i]) > longest_line):
			longest_line = len(text[i])

	new_text = np.zeros([len(text), longest_line, num_chars], np.float64)		
		
	for i in range(len(text)):
		for j in range(longest_line):
			if(len(text[i]) > j):
				new_text[i][j][text[i][j]] = 1
			else: 
				new_text[i][j] = np.zeros([num_chars])
		
	text = new_text
	
	return text
	
def break_into_batches(batch_size, text, num_chars):

	ttext = np.zeros([len(text)//batch_size, batch_size, len(text[0]), num_chars], np.float64)

	for i in range(len(ttext) - 1):
		ttext[i] = text[i*batch_size:(i+1)*batch_size]

	return ttext.transpose([0, 2, 1, 3])
