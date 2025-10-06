import re
from typing import List, ClassVar
from dataclasses import dataclass

@dataclass(frozen=True)
class Token:
    value: str
    original_value: str
    is_emoji: bool 

class TextTokenizerEnhanced:
    # Единый комбинированный паттерн
    comprehensive: ClassVar[re.Pattern] = re.compile(
        r'(?:[:;8=][\-\^]?[)D(PO3]+)|'  # Смайлики ASCII
        r'(?:[\U0001F300-\U0001F5FF])|'  # Символы и пиктограммы
        r'(?:[\U0001F600-\U0001F64F])|'  # Emoji эмоции
        r'(?:[\U0001F680-\U0001F6FF])|'  # Транспорт и символы
        r'(?:[\U0001F1E0-\U0001F1FF])|'  # Флаги
        r'(?:[\u2600-\u26FF])|'          # Разные символы
        r'(?:[\u2700-\u27BF])|'          # Dingbats
        r'(\b[а-яё]+\b[!?.,;]*)',        # Слова с пунктуацией
        flags=re.IGNORECASE
    )

    @staticmethod
    def tokenize_with_positions(text: str) -> List[Token]:
        tokens: List[Token] = []
        
        for match in TextTokenizerEnhanced.comprehensive.finditer(text):
            if match.group(1):
                token_value = match.group(1)
                clean_word = re.sub(r'[!?.,;]+$', '', token_value)
                if clean_word:  
                    tokens.append(Token(clean_word.lower(), token_value, is_emoji=False))
            else:
                emoji_value = match.group()
                if emoji_value: 
                    tokens.append(Token(emoji_value, emoji_value, is_emoji=True))

        tokens = [token for token in tokens if token.value]
        return tokens