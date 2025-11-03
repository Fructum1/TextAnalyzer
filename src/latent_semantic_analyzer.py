import numpy as np
from scipy.linalg import svd
from scipy.spatial.distance import cosine
from collections import Counter
from tokenizer import Token, TextTokenizerEnhanced
from normalizer import RussianNormalizer
from gensim.models import Word2Vec, KeyedVectors
import multiprocessing
import asyncio
import gensim.downloader as api
import os

class LatentSemanticAnalyzer:
    def __init__(self, documents: list[str], k = None, num_top_words: int = 10, min_df: int = 1, max_df: float = 1.0):
        """
        Иницизация класса LSA.

        :param documents: Список сырых текстов (до токенизации).
        :param k: Количество тем.
        :param num_top_words: Число топ-слов для вывода по каждой теме.
        """
        self.documents = documents
        self.num_top_words = num_top_words
        self.min_df = min_df
        self.max_df = max_df
        self.normalizer = RussianNormalizer()
        
        self.words = None
        self.k = k
        self.tfidf_matrix = None
        self.word_to_idx = None
        self.idx_to_word = None
        self.idf = None
        self.doc_vectors = None
        self.word_vectors = None
        self.sigma = None
        self.Vt_k = None
        self.w2v_model = None

    async def fit(self):
        """
        Полный процесс: препроцессинг, построение TF-IDF и применение SVD.
        """
        await self._preprocess_documents()
        self._build_vocabulary_with_filtering()
        self._build_tfidf()
        self._apply_svd()
        self._train_word2vec()

    def get_topic_words(self, topic_idx: int, num_words: int = None) -> list:
        """
        Получение топ-слов для конкретной темы с весами.
        """
        if num_words is None:
            num_words = self.num_top_words
        
        topic_weights = self.word_vectors[:, topic_idx]
        top_indices = np.argsort(-np.abs(topic_weights))[:num_words]
        
        top_words = []
        for idx in top_indices:
            word = self.idx_to_word[idx]
            weight = topic_weights[idx]
            top_words.append((word, weight))
        
        return top_words

    def print_results(self, num_top_words: int = 5):
        """
        Читаемый вывод результатов.
        """

        print(f"Использовано {self.k} тем")
        print(f"Размер словаря: {len(self.word_to_idx)} слов")
        
        print(f"\nСингулярные значения (важность тем):")
        for i, sigma in enumerate(self.sigma):
            print(f"  Тема {i+1}: {sigma:.4f}")
        
        print(f"\nТоп-слова по темам:")
        for topic_idx in range(self.k):
            topic_words = self.get_topic_words(topic_idx, num_top_words)
            words_str = ", ".join([f"{word}({weight:.3f})" for word, weight in topic_words])
            print(f"  Тема {topic_idx+1}: {words_str}")

    def document_similarity(self, method: str, doc_idx1: int, doc_idx2: int = None):
        """
        Считает схожесть между двумя документами.
        Если указан только doc_idx1 — выводит топ-N наиболее похожих документов.
        """
        if method == "tdidf":
            return self._document_similarity_idf(doc_idx1, doc_idx2)
        elif method == "w2v":
            return self._document_similarity_w2v(doc_idx1, doc_idx2)
        else: 
            raise ValueError("Выбранный алгоритм для сравнения недоступен.")

    def _document_similarity_idf(self, doc_idx1: int, doc_idx2: int = None):
        if doc_idx2 is None:
            similarities = []
            vec1 = self.doc_vectors[doc_idx1]

            for i, vec2 in enumerate(self.doc_vectors):
                if i == doc_idx1:
                    continue
                if np.all(vec1 == 0) or np.all(vec2 == 0):
                    sim = 0.0
                else:
                    norm1 = np.linalg.norm(vec1)
                    norm2 = np.linalg.norm(vec2)
                    if norm1 == 0 or norm2 == 0:
                        sim = 0.0
                    else:
                        sim = np.dot(vec1 / norm1, vec2 / norm2)
                        sim = (max(-1.0, min(1.0, sim)) + 1) / 2
                similarities.append((i, sim))

            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities

        vec1 = self.doc_vectors[doc_idx1]
        vec2 = self.doc_vectors[doc_idx2]
        
        if np.all(vec1 == 0) or np.all(vec2 == 0):
            return 0.0

        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        vec1_norm = vec1 / norm1
        vec2_norm = vec2 / norm2
        
        similarity = np.dot(vec1_norm, vec2_norm)
        similarity = max(-1.0, min(1.0, similarity))
        
        return (similarity + 1) / 2

    def _document_similarity_w2v(self, doc_idx1: int, doc_idx2: int = None):
        """
        Проверяет схожесть документов, используя только Word2Vec.
        Если указан один индекс — возвращает top_n похожих документов.
        """
        if self.w2v_model is None:
            raise ValueError("Модель Word2Vec не обучена. Сначала вызовите fit().")

        if doc_idx2 is not None:
            v1 = self._get_document_vector_word2vec(doc_idx1)
            v2 = self._get_document_vector_word2vec(doc_idx2)
            if np.all(v1 == 0) or np.all(v2 == 0):
                return 0.0
            sim = 1 - cosine(v1, v2)
            return max(0.0, sim)

        sims = []
        for i in range(len(self.words)):
            if i == doc_idx1:
                continue
            sim = self._document_similarity_w2v(doc_idx1, i)
            sims.append((i, sim))
        sims.sort(key=lambda x: x[1], reverse=True)
        return sims

    def _build_vocabulary_with_filtering(self):
        """
        Построение словаря с фильтрацией редких и частых слов.
        """
        if not self.words:
            raise ValueError("Корпус пуст")
        
        word_doc_freq = Counter()
        for doc in self.words:
            word_doc_freq.update(set(doc))
        
        num_docs = len(self.words)
        
        filtered_words = set()
        for word, doc_freq in word_doc_freq.items():
            if (doc_freq >= self.min_df and 
                doc_freq <= self.max_df * num_docs):
                filtered_words.add(word)
        
        if not filtered_words:
            raise ValueError("После фильтрации словарь пуст")
        
        self.word_to_idx = {word: idx for idx, word in enumerate(filtered_words)}
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}

    def _build_tfidf(self):
        vocab_size = len(self.word_to_idx)
        num_docs = len(self.words)
        
        tf_matrix = np.zeros((num_docs, vocab_size))
        for i, doc in enumerate(self.words):
            word_counts = Counter(doc)
            for word, count in word_counts.items():
                if word in self.word_to_idx:
                    tf_matrix[i, self.word_to_idx[word]] = np.log(1 + count)
        
        df = np.sum(tf_matrix > 0, axis=0)
        self.idf = np.log((num_docs + 1) / (df + 1)) + 1
        
        self.tfidf_matrix = tf_matrix * self.idf
        
        norms = np.linalg.norm(self.tfidf_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1
        self.tfidf_matrix = self.tfidf_matrix / norms

    def _apply_svd(self):
        if self.tfidf_matrix.size == 0:
            raise ValueError("TF-IDF матрица пуста")
        
        max_possible_k = min(self.tfidf_matrix.shape)
        if self.k is None:
            U, Sigma, Vt = svd(self.tfidf_matrix, full_matrices=False)
            total_variance = np.sum(Sigma ** 2)
            cumulative_variance = np.cumsum(Sigma ** 2) / total_variance
            self.k = np.argmax(cumulative_variance >= 0.9) + 1
            self.k = min(self.k, max_possible_k)
            print(f"Автоматически выбрано k={self.k} тем")
        
        if self.k > max_possible_k:
            print(f"k={self.k} больше максимального ({max_possible_k}), устанавливаем k={max_possible_k}")
            self.k = max_possible_k
        
        try:
            U, Sigma, Vt = svd(self.tfidf_matrix, full_matrices=False)
        except Exception as e:
            raise ValueError(f"Ошибка в SVD: {str(e)}")
        
        U_k = U[:, :self.k]
        Sigma_k = Sigma[:self.k]
        Vt_k = Vt[:self.k, :]
        
        self.doc_vectors = U_k * Sigma_k
        self.word_vectors = Vt_k.T
        self.sigma = Sigma_k
        self.Vt_k = Vt_k

    async def _preprocess_documents(self):
        tokenizer = TextTokenizerEnhanced()
        self.words = []
        for doc in self.documents:
            normalized_tokens = await self.normalizer.normalize(tokenizer.tokenize_with_positions(doc))
            normalized_words = [token.value for token in normalized_tokens if not token.is_emoji]
            if normalized_words:
                self.words.append(normalized_words)
        if not self.words:
            raise ValueError("Корпус после нормализации пуст")

    def _train_word2vec(self, vector_size=300, window=5, min_count=1):
        """
        Обучение Word2Vec для корпуса документов.
        """
        if not self.words:
            raise ValueError("Модель Word2Vec не обучена. Сначала вызовите fit().")
 
        model = Word2Vec(
            sentences=self.words,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            workers=multiprocessing.cpu_count(),
            sg=1
        )

        self.w2v_model = model.wv

    def _get_document_vector_word2vec(self, doc_idx: int):
        """
        Возвращает усреднённый вектор документа по обученной или предобученной модели Word2Vec.
        """
        if self.w2v_model is None:
            raise ValueError("Модель Word2Vec не обучена. Сначала вызовите fit().")

        kv = self.w2v_model

        words = [w.lower() for w in self.words[doc_idx] if isinstance(w, str)]
        found_words = [w for w in words if w in kv.key_to_index]

        if not found_words:
            return np.zeros(kv.vector_size)

        vectors = np.array([kv[w] for w in found_words])
        avg_vector = np.mean(vectors, axis=0)

        return avg_vector