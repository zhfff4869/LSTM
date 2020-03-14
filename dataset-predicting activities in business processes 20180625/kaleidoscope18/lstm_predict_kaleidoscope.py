from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense,Dropout,TimeDistributed, Activation
from keras.layers import LSTM
from keras.layers import Embedding
from keras.models import load_model
 

def generate_seq(model, tokenizer, max_length, seed_text, n_words):
	in_text = seed_text
	for _ in range(n_words):
		encoded = tokenizer.texts_to_sequences([in_text])[0]
		encoded = pad_sequences([encoded], maxlen=max_length, padding='pre')
		yhat = model.predict_classes(encoded, verbose=0)
		out_word = ''
		for word, index in tokenizer.word_index.items():
			if index == yhat:
				out_word = word
				break
		in_text += ' ' + out_word
	return in_text
 

with open('log_acron.txt', 'r') as myfile:
    data=myfile.read()

tokenizer = Tokenizer()
tokenizer.fit_on_texts([data])
encoded = tokenizer.texts_to_sequences([data])[0]

vocab_size = len(tokenizer.word_index) + 1
sequences = list()

for i in range(2, len(encoded)):
	sequence = encoded[i-2:i+1]
	sequences.append(sequence)
max_length = max([len(seq) for seq in sequences])
sequences = pad_sequences(sequences, maxlen=max_length, padding='pre')

sequences = array(sequences)
X, y = sequences[:,:-1],sequences[:,-1]

model = load_model('my_model_log_data.h5')

print('Results: ')
print ('\n')
print(generate_seq(model, tokenizer, max_length-1, 'tmqc' , 3))
print(generate_seq(model, tokenizer, max_length-1, 'pack' , 3))
print(generate_seq(model, tokenizer, max_length-1, 'grm27' , 3))
print(generate_seq(model, tokenizer, max_length-1, 'gr' , 3))
print(generate_seq(model, tokenizer, max_length-1, 'wcm18' , 3))
print(generate_seq(model, tokenizer, max_length-1, 'rgm19' , 3))
print(generate_seq(model, tokenizer, max_length-1, 'nqc' , 3))
print(generate_seq(model, tokenizer, max_length-1, 'rqc' , 3))
print(generate_seq(model, tokenizer, max_length-1, 'fm15' , 3))
print(generate_seq(model, tokenizer, max_length-1, 'fgm26' , 3))

