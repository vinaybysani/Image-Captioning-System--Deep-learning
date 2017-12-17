from vgg16 import VGG16
from keras.applications import inception_v3
import numpy as np
from keras.layers import LSTM, Embedding, TimeDistributed, Dense, RepeatVector, Merge, Activation, Flatten
from keras.preprocessing import image, sequence
import pandas as pd
from keras.models import Sequential
import pickle

DIM_EM = 128


class generate():

    def __init__(self):
        self.samples = None
        self.maximmum_length_of_caption = None
        self.word_index = None
        self.vocabulary_size = None
        self.index_word = None
        self.encoded_images = pickle.load( open( "encoded_images.p", "rb" ) )
        self.initialize_var()


    def data_engine(self, batch_size = 32):
        partial_caps, next_words, images, gen_count = [], [], [], 0
        data_frame = pd.read_csv('Flickr8k_text/flickr_8k_train_dataset.txt', delimiter='\t')
        caps, imgs = [], []
        for i in range(data_frame.shape[0]):
            x = next(data_frame.iterrows())
            caps.append(x[1][1])
            imgs.append(x[1][0])

        tot_cnt = 0
        while True:
            img_cnt = -1
            for txt in caps:
                img_cnt+=1
                for i in range(len(txt.split())-1):
                    tot_cnt+=1
                    partial_caps.append([self.word_index[txt] for txt in txt.split()[:i+1]])
                    nxt_1 = np.zeros(self.vocabulary_size)
                    nxt_1[self.word_index[txt.split()[i+1]]] = 1
                    next_words.append(nxt_1)
                    images.append(self.encoded_images[imgs[img_cnt]])

                    if tot_cnt>=batch_size:
                        next_words = np.asarray(next_words)
                        images = np.asarray(images)
                        partial_caps = sequence.pad_sequences(partial_caps, maxlen=self.maximmum_length_of_caption, padding='post')
                        tot_cnt = 0
                        gen_count+=1
                        yield [[images, partial_caps], next_words]
                        partial_caps, next_words, images = [], [], []

    def img_load(self, path):
        return np.asarray(image.img_to_array(image.load_img(path, target_size=(224,224))))


    def initialize_var(self):
        data_frame = pd.read_csv('Flickr8k_text/flickr_8k_train_dataset.txt', delimiter='\t')
        caps = []
        for i in range(data_frame.shape[0]):
            x = next(data_frame.iterrows())
            caps.append(x[1][1])

        self.samples=0
        for text in caps:
            self.samples += len(text.split()) - 1

        words = [txt.split() for txt in caps]
        u = []
        for w in words:
            u.extend(w)

        u = list(set(u))
        self.vocabulary_size, self.word_index, self.index_word = len(u), {}, {}
        for i, w in enumerate(u):
            self.word_index[w]=i
            self.index_word[i]=w

        max = 0
        for caption in caps:
            if(len(caption.split()) > max):
                max = len(caption.split())
        self.maximmum_length_of_caption = max


    def create_model(self, ret_model = False):
        image = Sequential()
        image.add(Dense(DIM_EM, input_dim = 4096, activation='relu'))
        image.add(RepeatVector(self.maximmum_length_of_caption))

        language = Sequential()
        language.add(Embedding(self.vocabulary_size, 256, input_length=self.maximmum_length_of_caption)) # vocab_size = unique words length(vocab), max_cap_len(40 in this case)
        language.add(LSTM(256,return_sequences=True))
        language.add(TimeDistributed(Dense(DIM_EM)))

        seq = Sequential()
        seq.add(Merge([image, language], mode='concat'))
        seq.add(LSTM(1000,return_sequences=False))
        seq.add(Dense(self.vocabulary_size))
        seq.add(Activation('softmax'))

        if(ret_model==True):
            return seq

        seq.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        return seq

    def get_word(self,index):
        return self.index_word[index]