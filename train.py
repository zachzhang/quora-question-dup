# -*- coding: utf8 -*-

import sys  

import time
import re
import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import StratifiedKFold
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout
from keras.layers.core import Lambda
from keras.layers.merge import concatenate, add, multiply
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.noise import GaussianNoise
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.metrics import euclidean_distances
from pyemd import emd
import nltk

from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


np.random.seed(0)
WNL = WordNetLemmatizer()
STOP_WORDS = set(stopwords.words('english'))
MAX_SEQUENCE_LENGTH = 30
MIN_WORD_OCCURRENCE = 100
REPLACE_WORD = "memento"
EMBEDDING_DIM = 200
NUM_FOLDS = 2
BATCH_SIZE = 1025
EMBEDDING_FILE = "/home/zz1409/glove.6B.200d.txt"


def cutter(word):
    if len(word) < 4:
        return word
    return WNL.lemmatize(WNL.lemmatize(word, "n"), "v")


def preprocess(string):
    string = string.lower().replace(",000,000", "m").replace(",000", "k").replace("′", "'").replace("’", "'") \
        .replace("won't", "will not").replace("cannot", "can not").replace("can't", "can not") \
        .replace("n't", " not").replace("what's", "what is").replace("it's", "it is") \
        .replace("'ve", " have").replace("i'm", "i am").replace("'re", " are") \
        .replace("he's", "he is").replace("she's", "she is").replace("'s", " own") \
        .replace("%", " percent ").replace("₹", " rupee ").replace("$", " dollar ") \
        .replace("€", " euro ").replace("'ll", " will").replace("=", " equal ").replace("+", " plus ")
    string = re.sub('[“”\(\'…\)\!\^\"\.;:,\-\?？\{\}\[\]\\/\*@]', ' ', string)
    string = re.sub(r"([0-9]+)000000", r"\1m", string)
    string = re.sub(r"([0-9]+)000", r"\1k", string)

    #string = string.encode("ascii", "ignore")
    #string = string.decode('utf-8', 'ignore')

    string = ' '.join([cutter(w) for w in string.split()])
    return string


def get_embedding():
    embeddings_index = {}
    f = open(EMBEDDING_FILE,encoding="utf-8")
    for line in f:
        values = line.split()
        word = values[0]
        if len(values) == EMBEDDING_DIM + 1 and word in top_words:
            coefs = np.asarray(values[1:], dtype="float32")
            embeddings_index[word] = coefs
    f.close()
    return embeddings_index


def is_numeric(s):
    return any(i.isdigit() for i in s)


def prepare(q):
    new_q = []
    surplus_q = []
    numbers_q = []
    new_memento = True
    for w in q.split()[::-1]:
        if w in top_words:
            new_q = [w] + new_q
            new_memento = True
        elif w not in STOP_WORDS:
            if new_memento:
                new_q = ["memento"] + new_q
                new_memento = False
            if is_numeric(w):
                numbers_q = [w] + numbers_q
            else:
                surplus_q = [w] + surplus_q
        else:
            new_memento = True
        if len(new_q) == MAX_SEQUENCE_LENGTH:
            break
    new_q = " ".join(new_q)
    return new_q, set(surplus_q), set(numbers_q)


def extract_features(df):
    q1s = np.array([""] * len(df), dtype=object)
    q2s = np.array([""] * len(df), dtype=object)
    features = np.zeros((len(df), 4))

    for i, (q1, q2) in enumerate(list(zip(df["question1"], df["question2"]))):
        q1s[i], surplus1, numbers1 = prepare(q1)
        q2s[i], surplus2, numbers2 = prepare(q2)
        features[i, 0] = len(surplus1.intersection(surplus2))
        features[i, 1] = len(surplus1.union(surplus2))
        features[i, 2] = len(numbers1.intersection(numbers2))
        features[i, 3] = len(numbers1.union(numbers2))

    return q1s, q2s, features


#too slow :(
def word_mover_dist(df, embedding_matrix, embedding_idx):

    vect = CountVectorizer(lowercase=False, token_pattern="\S+", min_df=MIN_WORD_OCCURRENCE)
    vect.vocabulary_ = embedding_idx

    D = euclidean_distances(embedding_matrix)

    q1 = vect.transform( df['question1'] ).toarray().astype(np.float)
    q2 = vect.transform( df['question2'] ).toarray().astype(np.float)

    q1 /= np.expand_dims(q1.sum(axis=1),1)
    q2 /= np.expand_dims(q2.sum(axis=1),1)

    word_move = np.zeros(df.shape[0])

    for i in range(df.shape[0]):
        word_move[i] = emd(q1[i].flatten(), q2[i].flatten(), D)
        print(df.loc[i,'question1'],df.loc[i,'question2'] ,df.loc[i,'is_duplicate'] , word_move[i])

    return word_move


def question_model(nb_words,embedding_matrix , n_dense_inputs):

    embedding_layer = Embedding(nb_words,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)

    lstm_layer = LSTM(75, recurrent_dropout=0.2)

    sequence_1_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype="int32")
    embedded_sequences_1 = embedding_layer(sequence_1_input)
    x1 = lstm_layer(embedded_sequences_1)

    sequence_2_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype="int32")
    embedded_sequences_2 = embedding_layer(sequence_2_input)
    y1 = lstm_layer(embedded_sequences_2)

    features_input = Input(shape=(n_dense_inputs,), dtype="float32")
    features_dense = BatchNormalization()(features_input)
    features_dense = Dense(200, activation="relu")(features_dense)
    features_dense = Dropout(0.2)(features_dense)

    addition = add([x1, y1])
    minus_y1 = Lambda(lambda x: -x)(y1)
    merged = add([x1, minus_y1])
    merged = multiply([merged, merged])
    merged = concatenate([merged, addition])
    merged = Dropout(0.4)(merged)

    merged = concatenate([merged, features_dense])
    merged = BatchNormalization()(merged)
    merged = GaussianNoise(0.1)(merged)

    merged = Dense(150, activation="relu")(merged)

    out = Dense(1, activation="sigmoid")(merged)

    model = Model(inputs=[sequence_1_input, sequence_2_input, features_input], outputs=out)

    model.compile(loss="binary_crossentropy",
                              optimizer="nadam")

    return model


def pos_features(df):

    def _pos(x):

        tags = nltk.pos_tag(nltk.word_tokenize(x))
        tags = [ tag[1] for tag in tags]
        return ' '.join(tags)

    pos_q1 = df['question1'].apply(_pos)
    pos_q2 = df['question2'].apply(_pos)

    vect = CountVectorizer(max_features=50)
    vect.fit(list(df['question1']) + list(df['question2']))

    q1 = vect.transform(pos_q1).toarray()
    q2 = vect.transform(pos_q2).toarray()
    
    print(q1.shape)

    #pos_dist =  np.sqrt(np.sum((q1 - q2)**2 ,axis=1 ))
    pos_dist = np.abs(q1 - q2)

    return pos_dist


def LSA_dist(df):

    vect = TfidfVectorizer(lowercase=False, token_pattern="\S+", min_df=MIN_WORD_OCCURRENCE , stop_words='english')
    vect.fit(list(df['question1']) + list(df['question2']))

    #Q = vect.transform(list(df['question1']) + list(df['question2']))
    q1 = vect.transform(list(df['question1']))
    q2 = vect.transform(list(df['question2']))

    svd = TruncatedSVD(40)
    svd.fit( sparse.vstack([q1,q2]))

    z1 = svd.transform(q1)
    z2 = svd.transform(q2)

    return np.abs(z1 - z2)

    #return np.sqrt(np.sum((z1 - z2)**2 ,axis=1 ))


if __name__=='__main__':
   

    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
   
    #print(word_mover_dist(train.iloc[0:20], embedding_matrix, tok.word_index))

    train["question1"] = train["question1"].fillna("").apply(preprocess)
    train["question2"] = train["question2"].fillna("").apply(preprocess)
    
    #print(np.expand_dims(LSA_dist(train) , 1).shape)
    #quit()

    print("Creating the vocabulary of words occurred more than", MIN_WORD_OCCURRENCE)
    all_questions = pd.Series(train["question1"].tolist() + train["question2"].tolist()).unique()
    vectorizer = CountVectorizer(lowercase=False, token_pattern="\S+", min_df=MIN_WORD_OCCURRENCE)
    vectorizer.fit(all_questions)
    top_words = set(vectorizer.vocabulary_.keys())
    top_words.add(REPLACE_WORD)
    
    embeddings_index = get_embedding()
    print("Words are not found in the embedding:", top_words - embeddings_index.keys())
    top_words = embeddings_index.keys()
    
    print("Train questions are being prepared for LSTM...")
    q1s_train, q2s_train, train_q_features = extract_features(train)

    tokenizer = Tokenizer(filters="")
    tokenizer.fit_on_texts(np.append(q1s_train, q2s_train))
    word_index = tokenizer.word_index
    
    
    data_1 = pad_sequences(tokenizer.texts_to_sequences(q1s_train), maxlen=MAX_SEQUENCE_LENGTH)
    data_2 = pad_sequences(tokenizer.texts_to_sequences(q2s_train), maxlen=MAX_SEQUENCE_LENGTH)
    labels = np.array(train["is_duplicate"])
    
    nb_words = len(word_index) + 1
    embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
    
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)

        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    
    
    #print(word_mover_dist(train.iloc[0:20], embedding_matrix, embedding_indexs))
    #quit()

    np.save('embedding.npy',embedding_matrix)
    pickle.dump(tokenizer,open('tok.p','wb'))

    #pos = np.expand_dims(pos_features(train) , 1)
    pos = pos_features(train)
    #lsa = np.expand_dims(LSA_dist(train) , 1)
    lsa = LSA_dist(train)

    print("Train features are being merged with NLP and Non-NLP features...")
    train_nlp_features = pd.read_csv("data/nlp_features_train.csv",encoding='utf-8')
    #features_train = np.hstack((train_q_features, train_nlp_features))
    features_train = np.hstack((train_q_features, train_nlp_features,pos,lsa))
    
    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True)
    model_count = 0
    
    shuffle=  np.random.permutation(data_1.shape[0])
    split = int( data_1.shape[0] * .8 )

    idx_train = shuffle[:split]
    idx_val = shuffle[split:]

    print("MODEL:", model_count)
    data_1_train = data_1[idx_train]
    data_2_train = data_2[idx_train]
    labels_train = labels[idx_train]
    f_train = features_train[idx_train]
    
    data_1_val = data_1[idx_val]
    data_2_val = data_2[idx_val]
    labels_val = labels[idx_val]
    f_val = features_train[idx_val]
    
    model = question_model(nb_words,embedding_matrix , f_train.shape[1])
    
    best_model_path = "best_model_no_graph" + str(model_count) + ".h5"
    
    model_checkpoint = ModelCheckpoint(best_model_path, save_best_only=True, save_weights_only=True)
    
    early_stopping = EarlyStopping(monitor="val_loss", patience=5)
    
    hist = model.fit([data_1_train, data_2_train, f_train], labels_train,
        validation_data=([data_1_val, data_2_val, f_val], labels_val),
        epochs=15, batch_size=BATCH_SIZE, shuffle=True,
        callbacks=[early_stopping, model_checkpoint], verbose=1)
    
    #model.load_weights(best_model_path)
    print(model_count, "validation loss:", min(hist.history["val_loss"]))
    
    #preds = model.predict([test_data_1, test_data_2, features_test], batch_size=BATCH_SIZE, verbose=1)
    #submission = pd.DataFrame({"test_id": test["test_id"], "is_duplicate": preds.ravel()})
    #submission.to_csv("predictions/preds_no_graph" + str(model_count) + ".csv", index=False)
    
    model_count += 1
        
