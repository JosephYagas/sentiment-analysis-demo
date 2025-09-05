import gradio as gr
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os

# Hyperparameters
vocab_size = 10000
maxlen = 200
model_path = "sentiment_model.h5"

# Load IMDB dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=vocab_size)
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)

# Check if model exists, else train
if os.path.exists(model_path):
    model = keras.models.load_model(model_path)
    print("✅ Loaded saved model")
else:
    print("⚡ Training new model...")
    model = keras.Sequential([
        layers.Embedding(input_dim=vocab_size, output_dim=32, input_length=maxlen),
        layers.LSTM(32),
        layers.Dense(1, activation="sigmoid")
    ])

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(x_train, y_train, epochs=1, batch_size=128, validation_split=0.2)
    model.save(model_path)
    print("💾 Model saved!")

# Load IMDB word index for text encoding
word_index = keras.datasets.imdb.get_word_index()
word_index = {k:(v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

def encode_text(text):
    words = text.lower().split()
    seq = []
    for word in words:
        if word in word_index:
            seq.append(word_index[word])
        else:
            seq.append(word_index["<UNK>"])
    return seq

# Prediction function
def predict_sentiment(text):
    seq = encode_text(text)
    padded = keras.preprocessing.sequence.pad_sequences([seq], maxlen=maxlen)
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
