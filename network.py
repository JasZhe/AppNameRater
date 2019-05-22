import data_parser
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras import layers
from keras.preprocessing.sequence import pad_sequences
from keras import optimizers

app_title, app_rating = data_parser.get_app_info()

titles_train, titles_test, y_train, y_test = train_test_split(app_title, app_rating, test_size=0.25, random_state=42)

vectorizer = CountVectorizer(min_df=20, lowercase=True)
vectorizer.fit(app_title)

word_index = vectorizer.vocabulary_
word_index = {k:(v + 1) for k,v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<UNK>"] = 1
print(word_index.keys())
print(len(word_index))
print(titles_train[0])
print(titles_test[0])
titles_train = map(lambda title: map(lambda word: "<UNK>" if word.lower() not in word_index else word.lower(), title.split()), titles_train)
titles_test = map(lambda title: map(lambda word: "<UNK>" if word.lower() not in word_index else word.lower(), title.split()), titles_test)
print(titles_train[0])
print(titles_test[0])

X_train = map(lambda title: map(lambda word: word_index[word], title), titles_train)
X_test = map(lambda title: map(lambda word: word_index[word], title), titles_test)
print(X_train[0])
print(X_test[0])

maxlen = 8
X_train = pad_sequences(X_train,
                        value=word_index["<PAD>"],
                        padding='post',
                        maxlen=maxlen)

X_test = pad_sequences(X_test,
                       value=word_index["<PAD>"],
                       padding='post',
                       maxlen=maxlen)
print(X_train[0])

vocab_size = len(word_index)

model = Sequential()
model.add(layers.Embedding(input_dim=vocab_size, output_dim=16, input_length=maxlen))
model.add(layers.Flatten())
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(5, activation='sigmoid'))

model.summary()
model.compile(optimizer=optimizers.Adam(lr=0.01, decay=0),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Currently, slow learning but validation accuracy doesn't increase, maybe parsing data incorrectly?

history = model.fit(X_train, y_train,
                    epochs=100,
                    batch_size=10,
                    validation_data=(X_test, y_test),
                    verbose=1)