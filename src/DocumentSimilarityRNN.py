import json
import os
from collections import Counter
from typing import List
import numpy as np
import tensorflow as tf
from datasets import load_dataset
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import (
    Embedding, SimpleRNN, Dense, Dropout, BatchNormalization,
    LayerNormalization, Subtract, Multiply, Dot, Concatenate
)
from tensorflow.keras import Input
from keras.saving import register_keras_serializable
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from normalizer import RussianNormalizer
from tokenizer import TextTokenizerEnhanced 
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

@register_keras_serializable(package="DocumentSimilarityRNN")
class AbsLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        return tf.abs(inputs)

    def get_config(self):
        config = super().get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class DocumentSimilarityRNN:
    def __init__(
        self,
        max_vocab=100000,
        max_len=40,
        embed_dim=64,
        rnn_units=512,
        batch_size=16,
        epochs=10,
    ):
        self.max_vocab = max_vocab
        self.max_len = max_len
        self.embed_dim = embed_dim
        self.rnn_units = rnn_units
        self.batch_size = batch_size
        self.epochs = epochs

        self.stoi = None
        self.model = None

        self._tokenizer = TextTokenizerEnhanced()
        self._normalizer = RussianNormalizer()

    async def _tokenize(self, texts: List[str]) -> List[List[str]]:
        all_tokens = []
        lengths = []

        for text in texts:
            toks = self._tokenizer.tokenize_with_positions(text)
            toks = [t for t in toks if not t.is_emoji]
            all_tokens.extend(toks)
            lengths.append(len(toks))

        normalized = await self._normalizer.normalize(all_tokens)

        result = []
        idx = 0
        for ln in lengths:
            chunk = normalized[idx: idx + ln]
            result.append([t.value for t in chunk])
            idx += ln

        return result

    async def build_vocab(self, texts: List[str]):
        counter = Counter()
        tokens = await self._tokenize(texts)
        for token in tokens:
            counter.update(token)

        most_common = counter.most_common(self.max_vocab - 2)

        stoi = {w: i + 2 for i, (w, _) in enumerate(most_common)}
        stoi["<PAD>"] = 0
        stoi["<UNK>"] = 1

        self.stoi = stoi

    async def texts_to_seq(self, texts: List[str]):
        all_tokens = await self._tokenize(texts)

        assert len(all_tokens) == len(texts), (
            f"Tokenization mismatch: {len(all_tokens)} tokens vs {len(texts)} texts"
        )

        seqs = []
        for toks in all_tokens:
            seqs.append([self.stoi.get(w, 1) for w in toks])

        arr = pad_sequences(
            seqs,
            maxlen=self.max_len,
            padding="post",
            truncating="post"
        )

        assert arr.shape[0] == len(texts), (
            f"Sequence output mismatch: {arr.shape[0]} vs {len(texts)}"
        )

        return arr

    def build_model(self):
        vocab_size = max(self.stoi.values()) + 1

        input_a = Input(shape=(self.max_len,), name="input_a")
        input_b = Input(shape=(self.max_len,), name="input_b")

        embedding = Embedding(
            vocab_size,
            self.embed_dim,
            mask_zero=False,
            name="embedding"
        )

        rnn = SimpleRNN(
            self.rnn_units,
            dropout=0.2,
            recurrent_dropout=0.2,
            return_sequences=False,
            name="rnn"
        )

        vec_a = rnn(embedding(input_a))
        vec_b = rnn(embedding(input_b))

        norm = LayerNormalization(name="l2norm")
        vec_a = norm(vec_a)
        vec_b = norm(vec_b)

        diff = Subtract(name="subtract")([vec_a, vec_b])
        abs_diff = AbsLayer(name="abs_diff")(diff)

        mul = Multiply(name="multiply")([vec_a, vec_b])

        cos_sim = Dot(axes=1, normalize=True, name="cosine_similarity")([vec_a, vec_b])
        cos_sim = tf.keras.layers.Reshape((1,), name="cos_sim_reshape")(cos_sim)

        features = Concatenate(name="features")([abs_diff, mul, cos_sim])

        x = Dense(64, activation="relu")(features)
        x = Dropout(0.3)(x)
        x = BatchNormalization()(x)

        x = Dense(32, activation="relu")(x)
        x = Dropout(0.3)(x)

        output = Dense(1, activation="sigmoid")(x)

        self.model = Model([input_a, input_b], output)

        self.model.compile(
            optimizer=Adam(learning_rate=3e-4),
            loss="binary_crossentropy",
            metrics=[
                tf.keras.metrics.BinaryAccuracy(name="accuracy"),
                tf.keras.metrics.AUC(name="auc")
            ]
        )

        self.model.summary()


    async def prepare_dataset(self):
        ds = load_dataset("merionum/ru_paraphraser")
        train = ds["train"]
        test = ds["test"]

        texts1 = train["text_1"]
        texts2 = train["text_2"]
        labels = train["class"]
        test_t1 = test["text_1"]
        test_t2 = test["text_2"]
        test_labels = test["class"]
        await self.build_vocab(list(texts1) + list(texts2))

        X1 = await self.texts_to_seq(list(texts1))
        X2 = await self.texts_to_seq(list(texts2))
        y = (np.array(labels) == "1").astype(np.float32)

        X1_test = await self.texts_to_seq(list(test_t1))
        X2_test = await self.texts_to_seq(list(test_t2))
        y_test = (np.array(test_labels) == "1").astype(np.float32)

        return self.balance(X1, X2, y), (X1_test, X2_test, y_test)
    
    def balance(self, X1, X2, y):
        pos_idx = np.where(y == 1)[0]
        neg_idx = np.where(y == 0)[0]

        if len(pos_idx) == 0 or len(neg_idx) == 0:
            raise Exception("Нет примеров одного из классов!")

        neg_idx_down = np.random.choice(neg_idx, size=len(pos_idx), replace=False)

        idx = np.concatenate([pos_idx, neg_idx_down])
        np.random.shuffle(idx)

        return X1[idx], X2[idx], y[idx]

    async def train(self):
        (X1, X2, y), (X1_test, X2_test, y_test) = await self.prepare_dataset()
        self.build_model()

        self.model.fit(
            [X1, X2],
            y,
            validation_data=([X1_test, X2_test], y_test),
            batch_size=self.batch_size,
            epochs=self.epochs,
            shuffle=True,
        )

    async def predict_similarity(self, doc1: str, doc2: str):
        s1 = await self.texts_to_seq([doc1])
        s2 = await self.texts_to_seq([doc2])
        prob = float(self.model.predict([s1, s2])[0][0])
        return prob, (prob >= 0.5)

    def save(self, path: str):
        self.model.save(path)
        with open(path + "_vocab.json", "w", encoding="utf8") as f:
            json.dump(self.stoi, f, ensure_ascii=False, indent=2)

    def load(self, path: str):
        self.model = tf.keras.models.load_model(path, custom_objects={"AbsLayer": AbsLayer})
        with open(path + "_vocab.json", "r", encoding="utf8") as f:
            self.stoi = json.load(f)