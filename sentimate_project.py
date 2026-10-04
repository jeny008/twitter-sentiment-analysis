import pandas as pd

# =========================
# 1. Load Data
# =========================
data = pd.read_csv('twitter_training.csv')

# Sample (fast training ke liye)
data = data.sample(10000, random_state=42)

# =========================
# 2. Cleaning
# =========================
data = data.dropna()
data.drop_duplicates(inplace=True)

# Rename columns (adjust if needed)
data.rename(columns={
    'Positive': 'Sentiment',
    'im getting on borderlands and i will murder you all ,': 'Tweets'
}, inplace=True)

# Keep valid sentiments
data = data[data['Sentiment'].isin(['Positive', 'Negative', 'Neutral'])]

# =========================
# 3. Label Encoding
# =========================
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
data['Sentiment'] = le.fit_transform(data['Sentiment'])

# =========================
# 4. Tokenization
# =========================
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(data['Tweets'])

sequences = tokenizer.texts_to_sequences(data['Tweets'])

# Padding
max_len = 100
X = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')

y = data['Sentiment'].values

# =========================
# 5. Train Test Split
# =========================
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# 6. Model (Embedding + RNN)
# =========================
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras import Input

vocab_size = 5000

model = Sequential([
    Input(shape=(max_len,)),

    # 🔥 Embedding Layer
    Embedding(input_dim=vocab_size, output_dim=128, input_length=99,mask_zero=True),

    # RNN
    SimpleRNN(32),

    # Output
    Dense(3, activation='softmax')
])

model.summary()

# =========================
# 7. Compile
# =========================
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# =========================
# 8. Train
# =========================
history = model.fit(
    X_train,
    y_train,
    epochs=5,
    batch_size=128,
    validation_data=(X_test, y_test)
)

# =========================
# 9. Vocab Size
# =========================
print("Vocab Size:", len(tokenizer.word_index) + 1)