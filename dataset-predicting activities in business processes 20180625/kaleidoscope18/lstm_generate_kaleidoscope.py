from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense,Dropout,TimeDistributed, Activation
from keras.layers import LSTM
from keras.layers import Embedding
from keras.models import load_model
 


with open('log_acron.txt', 'r') as myfile:	#read log data
    data=myfile.read()


tokenizer = Tokenizer()	#This class allows to vectorize a text corpus
tokenizer.fit_on_texts([data])	#Updates internal vocabulary based on a list of texts.
encoded = tokenizer.texts_to_sequences([data])[0]	#Transforms each text in texts to a sequence of integers,encoded by word_index dict.use [0] to get the texts for document_count:1
vocab_size = len(tokenizer.word_index) + 1	#+1 take the _len_ element into count? but word_doc=55
print('Vocabulary Size: %d' % vocab_size)

# encode 2 words -> 1 word
sequences = list()	#create a new empty list
for i in range(2, len(encoded)):	
	sequence = encoded[i-2:i+1]	#why is this 3 words?every time >>1 to get new 3 words's integer
	sequences.append(sequence)
print('Total Sequences: %d' % len(sequences))	#is this to make 1st element in sequences have next 2 outputs

max_length = max([len(seq) for seq in sequences])	#max_len of sequence
sequences = pad_sequences(sequences, maxlen=max_length, padding='pre')	#from pre to pad 0 if need
print('Max Sequence Length: %d' % max_length)	#in debugging only show 300 for it's too many

sequences = array(sequences)
X, y = sequences[:,:-1],sequences[:,-1]	#X each sequence except the last one,y only last one,from pre 2 words in X predict 1 in y?
y = to_categorical(y, num_classes=vocab_size)	#example:3->[0,0,0,1,0,0...] that len=num_classes

model = Sequential()
model.add(Embedding(vocab_size, 50, input_length=max_length-1))
model.add(LSTM(50))
model.add(Dropout(0.1))
model.add(Dense(vocab_size, activation='softmax'))
                                                                                                             
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=500, verbose=2,batch_size = 20)

print(model.summary())
model.save('model.h5')  # creates a HDF5 file 
del model  # deletes the existing model

