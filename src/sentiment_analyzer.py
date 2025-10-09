import asyncio
import math
import os
from typing import Dict, List, Set
from dataclasses import dataclass
from tokenizer import Token


@dataclass
class SentimentResult:
    score: float
    sentiment: str
    word_count: int


class SentimentAnalyzer:
    def __init__(self):
        self._lexicon: Dict[str, float] = {}
        self._booster_words: Dict[str, float] = self._load_booster_words()
        self._negation_words: Set[str] = self._load_negation_words()
        self._load_tonal_lexicon()
    
    async def analyze(self, text: str) -> SentimentResult:
        from tokenizer import TextTokenizerEnhanced
        from normalizer import RussianNormalizer
        
        normalizer = RussianNormalizer()
        tokens = TextTokenizerEnhanced().tokenize_with_positions(text)
        
        emoji_tokens = [token for token in tokens if token.is_emoji]
        word_tokens = [token for token in tokens if not token.is_emoji]
        
        normalized_words = await normalizer.normalize(word_tokens)
        
        all_tokens = []
        
        all_tokens.extend(normalized_words)
        all_tokens.extend(emoji_tokens)
        
        score, word_count = self._apply_vader_rules_enhanced(all_tokens)
        
        normalized_score = self._normalize_vader_score(score, word_count)
        
        return SentimentResult(
            score=normalized_score,
            sentiment=self._get_sentiment_label(normalized_score),
            word_count=word_count
        )

    def _apply_vader_rules_enhanced(self, tokens: List[Token]) -> tuple[float, int]:
        """Применяет VADER правила к списку Token объектов"""
        score = 0.0
        word_count = 0
        processed_tokens = [token.value for token in tokens]
        
        for i, token in enumerate(tokens):
            token_value = token.value
            if token_value in self._lexicon:
                
                word_score = self._lexicon[token_value]
                
                if not token.is_emoji:
                    word_score = self._apply_negation_rules_enhanced(tokens, i, word_score)
                
                word_score = self._apply_booster_rules_enhanced(tokens, i, word_score)
                
                if not token.is_emoji:
                    word_score = self._apply_caps_rules(token_value, word_score)
                
                word_score = self._apply_emoji_punctuation_rules(token, word_score)
                
                word_score = self._apply_modifier_rules_enhanced(tokens, i, word_score)
                
                word_score = self._apply_contrast_rules_enhanced(tokens, i, word_score)
                
                score += word_score
                word_count += 1
        
        score = self._apply_global_emoji_rules(tokens, score)
        
        return score, word_count

    def _apply_negation_rules_enhanced(self, tokens: List[Token], index: int, score: float) -> float:
        """Правила отрицания для Token объектов"""
        for j in range(max(0, index - 3), index):
            prev_token = tokens[j]
            if not prev_token.is_emoji and prev_token.value in self._negation_words:
                distance = index - j
                negation_strength = 1.0 - (distance - 1) * 0.1
                return score * -0.74 * negation_strength
        
        return score

    def _apply_booster_rules_enhanced(self, tokens: List[Token], index: int, score: float) -> float:
        """Правила усилителей для Token объектов"""
        if index > 0:
            prev_token = tokens[index - 1]
            if not prev_token.is_emoji and prev_token.value in self._booster_words:
                boost_strength = self._booster_words[prev_token.value]
                return score * boost_strength
        
        if index < len(tokens) - 1:
            next_token = tokens[index + 1]
            if not next_token.is_emoji and next_token.value in self._intensifiers:
                return score * 1.1
        
        return score

    def _apply_emoji_punctuation_rules(self, token: Token, score: float) -> float:
        """Обработка пунктуации в эмодзи и смайликах"""
        if token.is_emoji:
            return score
        else:
            if "!" in token.original_value:
                exclamation_count = token.original_value.count("!")
                return score * (1.0 + exclamation_count * 0.1)
            elif "?" in token.original_value:
                question_count = token.original_value.count("?")
                return score * (1.0 - question_count * 0.05)
        
        return score

    def _apply_modifier_rules_enhanced(self, tokens: List[Token], index: int, score: float) -> float:
        """Правила модификаторов для Token объектов"""
        modifiers = {
            "вроде": 0.7, "как бы": 0.6, "типа": 0.7, "почти": 0.8,
            "слегка": 0.8, "немного": 0.8, "отчасти": 0.9, "частично": 0.9,
            "не совсем": 0.5, "в некоторой степени": 0.8
        }
        
        for j in range(max(0, index - 2), index):
            prev_token = tokens[j]
            if not prev_token.is_emoji and prev_token.value in modifiers:
                return score * modifiers[prev_token.value]
        
        return score

    def _apply_contrast_rules_enhanced(self, tokens: List[Token], index: int, score: float) -> float:
        """Правила контраста для Token объектов"""
        contrast_words = {"но", "однако", "тем не менее", "впрочем"}
        
        for j in range(max(0, index - 5), index):
            prev_token = tokens[j]
            if not prev_token.is_emoji and prev_token.value in contrast_words:
                return score * 1.3
        
        return score

    def _apply_global_emoji_rules(self, tokens: List[Token], score: float) -> float:
        """Глобальные правила для эмодзи и пунктуации"""
        positive_emojis = 0
        negative_emojis = 0
        
        for token in tokens:
            if token.is_emoji:
                emoji_score = self._lexicon.get(token.value, 0)
                if emoji_score > 0:
                    positive_emojis += 1
                elif emoji_score < 0:
                    negative_emojis += 1
        
        if positive_emojis > 1:
            score *= (1.0 + positive_emojis * 0.05)
        if negative_emojis > 1:
            score *= (1.0 - negative_emojis * 0.05)
        
        total_exclamations = sum(token.original_value.count("!") for token in tokens)
        total_questions = sum(token.original_value.count("?") for token in tokens)
        
        if total_exclamations > 0:
            exclamation_boost = 1.0 + (min(total_exclamations, 10) * 0.05)
            score *= exclamation_boost
        
        if total_questions > 3:
            question_penalty = 1.0 - (min(total_questions - 3, 5) * 0.03)
            score *= question_penalty
        
        return score

    def _apply_caps_rules(self, token_value: str, score: float) -> float:
        """Правила CAPS LOCK"""
        if token_value.isupper() and len(token_value) > 1:
            return score * 1.2
        return score

    def _normalize_vader_score(self, score: float, word_count: int) -> float:
        """Нормализует оценку по формуле VADER"""
        if word_count == 0:
            return 0.0
        
        normalized = score / math.sqrt(score + 15)
        
        if word_count > 0:
            normalized = normalized / math.sqrt(1 + word_count * 0.001)
        
        return max(-4.0, min(4.0, normalized))
    
    @property
    def _intensifiers(self) -> Set[str]:
        return {
            "же", "ведь", "вот", "прямо", "просто", "действительно",
            "именно", "точно", "ровно"
        }


    def _load_tonal_lexicon(self) -> None:
        """Загружает тональный словарь из файла в формате VADER"""
        try:
            lexicon_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "vader_lexicon.txt")
            
            if not os.path.exists(lexicon_path):
                print(f"Файл словаря не найден: {lexicon_path}")
                return
            
            with open(lexicon_path, 'r', encoding='utf-8') as file:
                for line in file:
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        word = parts[0].strip()
                        
                        score_str = parts[1].replace(',', '.').strip()
                        
                        try:
                            score = float(score_str)
                            self._lexicon[word] = score
                        except ValueError:
                            continue
            
            print(f"Загружено {len(self._lexicon)} слов с тональностью")
        
        except Exception as ex:
            print(f"Ошибка при чтении файла словаря: {ex}")
            """Загружает тональный словарь из файла"""
            try:
                lexicon_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "vader_lexicon.txt")
                
                if not os.path.exists(lexicon_path):
                    print(f"Файл словаря не найден: {lexicon_path}")
                    return
                
                with open(lexicon_path, 'r', encoding='utf-8') as file:
                    for line in file:
                        line = line.strip()
                        if not line:
                            continue
                        
                        columns = line.split('\t')
                        if len(columns) >= 2:
                            word = columns[0].strip()
                            try:
                                score = float(columns[1])
                                self._lexicon[word] = score
                            except ValueError:
                                continue
                
                print(f"Загружено {len(self._lexicon)} слов с тональностью")
                
            except Exception as ex:
                print(f"Ошибка при чтении файла словаря: {ex}")
        
    
    def _get_sentiment_label(self, score: float) -> str:
        match score:
            case x if x >= 0.05:
                return "Позитивная тональность"
            case x if x <= -0.05:
                return "Негативная тональность"
            case _:
                return "Нейтральная"
    
    def _load_booster_words(self) -> Dict[str, float]:
        return {
            "прекрасно": 4.0,
            "идеально": 4.0,
            "восхитительно": 4.0,
            "блестяще": 4.0,
            "великолепно": 4.0,
            "потрясающе": 3.0,
            "изумительно": 3.0,
            "невероятно": 3.0,
            "очень": 2.0,
            "чрезвычайно": 2.0,
            "исключительно": 2.0,
            "необычайно": 2.0,
            "довольно": 1.0,
            "весьма": 1.0,
            "достаточно": 1.0,
            "абсолютно": 0.0,
            "совершенно": 0.0,
            "полностью": 0.0,
            "целиком": 0.0,
            "прямо": 0.0,
            "просто": 0.0,
            "прямо-таки": 0.0,
            "даже": 0.0,
            "вот": 0.0,
            "же": 0.0,
            "слишком": -1.0,
            "чрезмерно": -1.0,
            "катастрофически": -2.0,
            "критически": -2.0,
            "ужасно": -2.0,
            "кошмарно": -2.0,
            "отвратительно": -4.0,
            "омерзительно": -4.0,
            "ужасающе": -3.0,
            "невыносимо": -3.0,
            "нестерпимо": -3.0
        }
    
    def _load_negation_words(self) -> Set[str]:
        return {"не", "нет", "ни", "никогда", "нисколько", "никак",
            "ничуть", "ничего", "никуда", "нигде", "никто", "ничто"}