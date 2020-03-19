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
		encoded = tokenizer.texts_to_sequences([in_text])[0]	#use word_index to transform
		encoded = pad_sequences([encoded], maxlen=max_length, padding='pre')	#pad 0 to maxlen,first [2] to [[2]] then to [[0,2]] for input should be 2 dim
		yhat = model.predict_classes(encoded, verbose=0)	#当使用predict()方法进行预测时，返回值是数值，表示样本属于每一个类别的概率，我们可以使用numpy.argmax()方法找到样本以最大概率所属的类别作为样本的预测标签。当使用predict_classes()方法进行预测时，返回的是类别的索引，即该样本所属的类别标签
		out_word=tokenizer.index_word[yhat[0]]	#temp test
		# out_word = ''
		# for word, index in tokenizer.word_index.items():	#items() return a dict_items[(word,index),...] from {word:index,...}
		# 	if index == yhat:
		# 		out_word = word
		# 		break
		in_text += ' ' + out_word
	return in_text
 

with open('log_acron.txt', 'r') as myfile:
    data=myfile.read()

tokenizer = Tokenizer()
tokenizer.fit_on_texts([data])	#split by ' ', remove punctuation,init a vocabulary object,defaut set words to lowercase,word_index is from word to index,index_word is from index to word
encoded = tokenizer.texts_to_sequences([data])[0] #use word_index to transform

vocab_size = len(tokenizer.word_index) + 1	#why +1? word index start from 1, so the index of 0 would be 0 all the time
sequences = list()

for i in range(2, len(encoded)):
	sequence = encoded[i-2:i+1]	#first is [0:3],next is [1:4],next is [2:5];first dim will get 0 as input,1 and 2 as next 2 outputs
	sequences.append(sequence)
max_length = max([len(seq) for seq in sequences])	#sequences len become 4541=4543-2
sequences = pad_sequences(sequences, maxlen=max_length, padding='pre')	#pad each sequence to maxlen

sequences = array(sequences)
X, y = sequences[:,:-1],sequences[:,-1]	#X get the pre 2 int,y get the third int

# model = load_model('my_model_log_data.h5')
model = load_model('model.h5')


print('Results: ')
print ('\n')
print(generate_seq(model, tokenizer, max_length-1, 'tmqc' , 3))	#max_length-1=2
print(generate_seq(model, tokenizer, max_length-1, 'pack' , 3))
print(generate_seq(model, tokenizer, max_length-1, 'grm27' , 3))
print(generate_seq(model, tokenizer, max_length-1, 'gr' , 3))
print(generate_seq(model, tokenizer, max_length-1, 'wcm18' , 3))
print(generate_seq(model, tokenizer, max_length-1, 'rgm19' , 3))
print(generate_seq(model, tokenizer, max_length-1, 'nqc' , 3))
print(generate_seq(model, tokenizer, max_length-1, 'rqc' , 3))
print(generate_seq(model, tokenizer, max_length-1, 'fm15' , 3))
print(generate_seq(model, tokenizer, max_length-1, 'fgm26' , 3))

