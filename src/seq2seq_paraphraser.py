import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


class Seq2SeqParaphraser:
    def __init__(self, latent_dim=256, max_len=40):
        self.latent_dim = latent_dim
        self.max_len = max_len
        self.tokenizer = Tokenizer(filters="", oov_token="<unk>")
        self.encoder_model = None
        self.decoder_model = None
        self.model = None
        self.num_tokens = 0

    def prepare_dataset(self, pairs):
        inputs = []
        outputs = []

        for src, tgt in pairs:
            inputs.append(src)
            outputs.append("<start> " + tgt + " <end>")

        self.tokenizer.fit_on_texts(inputs + outputs)
        self.num_tokens = len(self.tokenizer.word_index) + 1

        encoder_input = pad_sequences(
            self.tokenizer.texts_to_sequences(inputs),
            maxlen=self.max_len,
            padding="post"
        )

        decoder_input_seq = pad_sequences(
            self.tokenizer.texts_to_sequences(outputs),
            maxlen=self.max_len,
            padding="post"
        )

        decoder_target_seq = np.zeros_like(decoder_input_seq)
        decoder_target_seq[:, :-1] = decoder_input_seq[:, 1:]

        return encoder_input, decoder_input_seq, decoder_target_seq

    def train(self, pairs, epochs=30):
        encoder_input, decoder_input, decoder_target = self.prepare_dataset(pairs)

        enc_inputs = Input(shape=(self.max_len,))
        enc_embed = Embedding(self.num_tokens, self.latent_dim, mask_zero=True)(enc_inputs)
        encoder_lstm = LSTM(self.latent_dim, return_state=True)
        _, state_h, state_c = encoder_lstm(enc_embed)
        encoder_states = [state_h, state_c]

        dec_inputs = Input(shape=(self.max_len,))
        dec_embed = Embedding(self.num_tokens, self.latent_dim, mask_zero=True)
        dec_lstm = LSTM(self.latent_dim, return_sequences=True, return_state=True)
        dec_dense = Dense(self.num_tokens, activation="softmax")

        dec_embedded = dec_embed(dec_inputs)
        dec_outputs, _, _ = dec_lstm(dec_embedded, initial_state=encoder_states)
        dec_outputs = dec_dense(dec_outputs)

        self.model = Model([enc_inputs, dec_inputs], dec_outputs)
        self.model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=['accuracy'])
        self.model.summary()
        
        self.model.fit(
            [encoder_input, decoder_input], 
            decoder_target, 
            epochs=epochs,
            batch_size=32,
            verbose=1
        )

        self.encoder_model = Model(enc_inputs, encoder_states)

        dec_state_input_h = Input(shape=(self.latent_dim,))
        dec_state_input_c = Input(shape=(self.latent_dim,))
        dec_states_inputs = [dec_state_input_h, dec_state_input_c]

        dec_inputs_single = Input(shape=(1,))
        dec_embedded_single = dec_embed(dec_inputs_single)
        
        dec_outputs_single, state_h_single, state_c_single = dec_lstm(
            dec_embedded_single, 
            initial_state=dec_states_inputs
        )
        dec_outputs_single = dec_dense(dec_outputs_single)
        
        self.decoder_model = Model(
            [dec_inputs_single] + dec_states_inputs,
            [dec_outputs_single, state_h_single, state_c_single]
        )
        self.save_model("seq2seq_paraphraser")

    def paraphrase(self, text):
        seq = self.tokenizer.texts_to_sequences([text])
        seq = pad_sequences(seq, maxlen=self.max_len, padding="post")

        states = self.encoder_model.predict(seq, verbose=0)
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = self.tokenizer.word_index.get("<start>", 1)
        
        output = []
        recent_words = []
        
        for _ in range(self.max_len):
            preds, h, c = self.decoder_model.predict([target_seq] + states, verbose=0)
            
            logits = preds[0, -1, :].copy()
            
            for word in recent_words:
                if word in self.tokenizer.word_index:
                    idx = self.tokenizer.word_index[word]
                    logits[idx] *= 0.3
            
            token_id = np.argmax(logits)
            
            if token_id == 0:
                break
            
            word = self.tokenizer.index_word.get(token_id, "")
            
            if word == "<end>":
                break
            
            if word in recent_words:
                sorted_indices = np.argsort(logits)[::-1]
                for alt_id in sorted_indices[1:5]:
                    alt_word = self.tokenizer.index_word.get(alt_id, "")
                    if alt_word not in recent_words:
                        token_id = alt_id
                        word = alt_word
                        break
            
            output.append(word)
            
            recent_words.append(word)
            if len(recent_words) > 3:
                recent_words.pop(0)
            
            target_seq[0, 0] = token_id
            states = [h, c]
        
        return " ".join(output)

    def save_model(self, path):
        """Сохраняет полную модель, токенизатор и параметры"""
        import pickle
        import os
        
        os.makedirs(path, exist_ok=True)
        
        self.model.save(f"{path}/model.keras")
        
        self.encoder_model.save(f"{path}/encoder_model.keras")
        self.decoder_model.save(f"{path}/decoder_model.keras")
        
        with open(f"{path}/tokenizer.pkl", "wb") as f:
            pickle.dump(self.tokenizer, f)

        with open(f"{path}/params.pkl", "wb") as f:
            pickle.dump({
                'latent_dim': self.latent_dim,
                'max_len': self.max_len,
                'num_tokens': self.num_tokens
            }, f)

    def load_model(self, path):
        import pickle
        import os
        
        if not os.path.exists(f"{path}/model.keras"):
            raise FileNotFoundError(f"Файл модели не найден: {path}/model.keras")
        
        with open(f"{path}/params.pkl", "rb") as f:
            params = pickle.load(f)
        self.latent_dim = params['latent_dim']
        self.max_len = params['max_len']
        self.num_tokens = params['num_tokens']
        
        with open(f"{path}/tokenizer.pkl", "rb") as f:
            self.tokenizer = pickle.load(f)
        
        self.model = tf.keras.models.load_model(f"{path}/model.keras")
        self.encoder_model = tf.keras.models.load_model(f"{path}/encoder_model.keras")
        self.decoder_model = tf.keras.models.load_model(f"{path}/decoder_model.keras")
