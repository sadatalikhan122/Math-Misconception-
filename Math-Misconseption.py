import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from transformers import AutoTokenizer, pipeline

df =pd.read_csv('/kaggle/input/math-misconception/train.csv')
df = df.dropna(subset=['StudentExplanation', 'QuestionText'])


df["combined_text"] = df["QuestionText"] + " " + df["StudentExplanation"]


df["clean_text"] = df["combined_text"].str.replace(r"[^\w\s]", "", regex=True)


le = LabelEncoder()
df['label'] = le.fit_transform(df['Category'])


tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(df["clean_text"])
sequences = tokenizer.texts_to_sequences(df["clean_text"])


max_len = 100  
padded_sequences = pad_sequences(sequences, padding='post', maxlen=max_len)
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, df['label'], test_size=0.2, random_state=42)
model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=64, input_length=max_len))
model.add(LSTM(64, return_sequences=False))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dense(len(le.classes_), activation='softmax'))  # multi-class output
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
history = model.fit(X_train, y_train, epochs=3, batch_size=32, validation_data=(X_test, y_test))
loss, accuracy = model.evaluate(X_test, y_test)
print(f"\nTest Loss: {loss:.4f} | Test Accuracy: {accuracy:.4f}")


