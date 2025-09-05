import gradio as gr
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Load IMDB dataset (preloaded in Keras)
vocab_size = 10000
maxlen = 200

(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=vocab_size)
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)

# Build model
model = keras.Sequential([
    layers.Embedding(input_dim=vocab_size, output_dim=32, input_length=maxlen),
    layers.LSTM(32),
    layers.Dense(1, activation="sigmoid")
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(x_train, y_train, epochs=1, batch_size=128, validation_split=0.2)  # 1 epoch for demo

# Function to predict sentiment
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = keras.preprocessing.text.Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(["sample"])

def predict_sentiment(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=maxlen)
    prediction = model.predict(padded)[0][0]
    sentiment = "😊 Positive" if prediction > 0.5 else "😞 Negative"
    return {sentiment: float(prediction)}

# Gradio Interface
demo = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(lines=3, placeholder="Type a movie review..."),
    outputs="label",
    title="🎬 Sentiment Analysis Demo",
    description="This model predicts whether a review is Positive or Negative."
)

demo.launch()
