import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import TextVectorization
import re
from collections import Counter
from pathlib import Path
from typing import List, Dict, Tuple
import json
from seq2seq_paraphraser import Seq2SeqParaphraser

class LSTMTemplateGenerator:
    """
    LSTM генератор шаблонов, интегрированный в LSA анализатор.
    """
    
    def __init__(self, model_path: str = None):
        """
        Инициализация с возможностью загрузки модели.
        
        Args:
            model_path: Путь к сохраненной модели
        """
        self.document_themes = [
            "политика",
            "экономика", 
            "спорт",
            "технологии",
            "культура",
            "наука",
            "медицина",
            "образование",
            "происшествия",
            "общество",
            "программирование",
            "животные",
            "быт"
        ]

        self.theme_templates = self._create_templates()
        
        self.model = None
        self.vectorizer = None
        self.theme_to_idx = {t: i for i, t in enumerate(self.document_themes)}
        self.idx_to_theme = {i: t for i, t in enumerate(self.document_themes)}
        
        if model_path:
            self.load_model(model_path)
    
    def _create_templates(self) -> Dict[str, List[str]]:
        return {
            "политика": [
                "Тема '{TOPIC}' раскрывается через {KEYWORDS}.",
                "Политический аспект: {KEYWORDS} в контексте {TOPIC}.",
                "Анализ политической темы '{TOPIC}' показывает {KEYWORDS}."
            ],
            "животные": [
                "Документ ведает нам о `{TOPIC}`. Он также рассказывает о: {KEYWORDS}.",
                "Говоря о `{TOPIC}`, можно также выделить и {KEYWORDS}."
            ],
            "экономика": [
                "Экономическая тема '{TOPIC}', можно выделить следующие ключевые аспекты {KEYWORDS}.",
                "В экономическом контексте '{TOPIC}' важны {KEYWORDS}.",
                "Анализ экономики: тема '{TOPIC}' через {KEYWORDS}."
            ],
            "спорт": [
                "Спортивная тема '{TOPIC}' связана с {KEYWORDS}.",
                "В спортивном обзоре '{TOPIC}' выделяются {KEYWORDS}.",
                "Спортивные аспекты '{TOPIC}': {KEYWORDS}."
            ],
            "технологии": [
                "Технологическая тема '{TOPIC}': {KEYWORDS}.",
                "В технологиях '{TOPIC}' ключевые элементы: {KEYWORDS}.",
                "Технологический анализ '{TOPIC}': {KEYWORDS}."
            ],
            "культура": [
                "Культурная тема '{TOPIC}': {KEYWORDS}.",
                "В культуре '{TOPIC}' важны {KEYWORDS}.",
                "Культурный контекст '{TOPIC}': {KEYWORDS}."
            ],
            "наука": [
                "Научная тема '{TOPIC}': {KEYWORDS}.",
                "В науке '{TOPIC}' исследуются {KEYWORDS}.",
                "Научный анализ '{TOPIC}': {KEYWORDS}."
            ],
            "медицина": [
                "Медицинская тема '{TOPIC}': {KEYWORDS}.",
                "В медицине '{TOPIC}' ключевые аспекты: {KEYWORDS}.",
                "Медицинский контекст '{TOPIC}': {KEYWORDS}."
            ],
            "образование": [
                "Образовательная тема '{TOPIC}': {KEYWORDS}.",
                "В образовании '{TOPIC}' важны {KEYWORDS}.",
                "Образовательный анализ '{TOPIC}': {KEYWORDS}."
            ],
            "происшествия": [
                "Тема происшествий '{TOPIC}': {KEYWORDS}.",
                "В происшествиях '{TOPIC}' отмечаются {KEYWORDS}.",
                "Анализ происшествий '{TOPIC}': {KEYWORDS}."
            ],
            "общество": [
                "Общественная тема '{TOPIC}': {KEYWORDS}.",
                "В обществе '{TOPIC}' обсуждаются {KEYWORDS}.",
            ],
        }
    
    def build_model(self, vocab_size: int = 10000):
        """Построение LSTM модели."""
        model = tf.keras.Sequential([
            layers.Input(shape=(100,)),
            layers.Embedding(vocab_size, 128, mask_zero=True),
            layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
            layers.LSTM(32),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(len(self.document_themes), activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def create_vectorizer(self, corpus: List[str]):
        """Создание векторнаяizer."""
        self.vectorizer = TextVectorization(
            max_tokens=30000,
            output_mode='int',
            output_sequence_length=100,
            standardize='lower_and_strip_punctuation'
        )
        self.vectorizer.adapt(corpus)
    
    def extract_keywords(self, text: str, num: int = 5) -> List[str]:
        """Извлечение ключевых слов."""
        words = re.findall(r'\b[а-яё]{3,}\b', text.lower())
        
        stop_words = {
            'это', 'что', 'который', 'также', 'очень', 'можно', 'будет',
            'есть', 'когда', 'потому', 'чтобы', 'такой', 'только', 'ещё',
            'иногда'
        }
        
        filtered = [w for w in words if w not in stop_words]
        word_counts = Counter(filtered)
        
        return [word for word, _ in word_counts.most_common(num)]
    
    def predict_theme(self, text: str) -> Tuple[str, float]:
        text_vector = self.vectorizer([text]).numpy()
        
        predictions = self.model.predict(text_vector, verbose=0)
        predicted_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_idx]
        
        return self.idx_to_theme[predicted_idx], confidence

    def generate_for_document(self, document: str) -> str:
        theme, confidence = self.predict_theme(document)
        
        keywords = self.extract_keywords(document, num=5)
        
        templates = self.theme_templates.get(theme, self.theme_templates["общество"])
        template = np.random.choice(templates)
        
        if len(keywords) > 0:
            topic_name = keywords[0]
        else:
            topic_name = "документ"
        
        result = template.replace("{TOPIC}", topic_name)
        

        if len(keywords) >= 3:
            keywords_str = f"'{keywords[1]}' и '{keywords[2]}'"
        elif len(keywords) == 2:
            keywords_str = f"'{keywords[1]}'"
        elif len(keywords) == 1:
            keywords_str = "нельзя выделить еще какие-то ключевые понятия."
        else:
            keywords_str = "нельзя выделить какие-то ключевые понятия."
        
        result = result.replace("{KEYWORDS}", keywords_str)

        if (confidence < 0.35):
            return result, None, keywords
        
        paraphraser = Seq2SeqParaphraser()
        paraphraser.load_model("seq2seq_paraphraser")

        return paraphraser.paraphrase(result), theme, keywords
    
    def save_model(self, path: str):
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        if self.model:
            self.model.save(save_path / "lstm_template_model.keras")
        
        if self.vectorizer:
            vocab = self.vectorizer.get_vocabulary()
            with open(save_path / "vocab.json", "w", encoding="utf-8") as f:
                json.dump(vocab, f, ensure_ascii=False)
        
        config = {
            'document_themes': self.document_themes,
            'theme_templates': self.theme_templates
        }
        
        with open(save_path / "config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False)
    
    def load_model(self, path: str):
        load_path = Path(path)
        
        with open(load_path / "config.json", "r", encoding="utf-8") as f:
            config = json.load(f)
        
        self.document_themes = config['document_themes']
        self.theme_templates = config['theme_templates']
        self.theme_to_idx = {t: i for i, t in enumerate(self.document_themes)}
        self.idx_to_theme = {i: t for i, t in enumerate(self.document_themes)}
        
        if (load_path / "lstm_template_model.keras").exists():
            self.model = tf.keras.models.load_model(
                load_path / "lstm_template_model.keras"
            )
        
        if (load_path / "vocab.json").exists():
            with open(load_path / "vocab.json", "r", encoding="utf-8") as f:
                vocab = json.load(f)
            
            self.vectorizer = TextVectorization(
                max_tokens=10000,
                output_mode='int',
                output_sequence_length=100,
                standardize='lower_and_strip_punctuation'
            )
            self.vectorizer.set_vocabulary(vocab)
        
        return self