import asyncio
import os
import sys
from typing import Optional, Callable, Awaitable

# Добавляем текущую директорию в путь для импорта модулей
sys.path.append(os.path.dirname(__file__))

class Program:
    def __init__(self):
        self.exit_keyword = "exit"
    
    async def main(self):
        from sentiment_analyzer import SentimentAnalyzer
        
        user_input = ""
        sentiment_analyzer = SentimentAnalyzer()
        
        print("=== Анализатор тональности текста ===")
        print("Для выхода из программы наберите 'exit'.")
        
        while user_input.lower() != self.exit_keyword:
            await self._with_exception_handling(self._process_iteration, sentiment_analyzer)
    
    async def _process_iteration(self, analyzer):
        print("\n" + "="*50)
        print("Выберите режим работы:")
        print("1 - Анализ тональности из файла")
        print("2 - Анализ тональности из консоли")
        print("exit - Выход из программы")
        
        user_input = input("Ваш выбор: ").strip()
        
        if user_input.lower() == self.exit_keyword:
            print("Программа завершена. До свидания!")
            exit(0)
        
        text = await self._get_text_for_analysis(user_input)
        
        if text is None:
            print("Неверный входной формат. Пожалуйста, выберите 1, 2 или exit.")
            return
        
        print("Анализируем текст...")
        sentiment_result = await analyzer.analyze(text)
        
        print(f"\nРезультат анализа:")
        print(f"Тональность: {sentiment_result.sentiment}")
        print(f"Оценка: {sentiment_result.score:.3f}")
        print(f"Количество слов, учтенных в анализе: {sentiment_result.word_count}")
    
    async def _get_text_for_analysis(self, user_input: str) -> Optional[str]:
        if user_input == "1":
            return await self._get_text_from_file()
        elif user_input == "2":
            return self._get_text_from_console()
        else:
            return None
    
    async def _get_text_from_file(self) -> Optional[str]:
        print("Введите полный путь к файлу:")
        file_path = input().strip()
        
        if not file_path:
            print("Путь к файлу не может быть пустым.")
            return None
        
        if not os.path.exists(file_path):
            print(f"Файл не найден: {file_path}")
            return None
        
        try:
            encodings = ['utf-8', 'cp1251', 'windows-1251', 'koi8-r']
            
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as file:
                        lines = file.readlines()
                    
                    non_empty_lines = [line.strip() for line in lines if line.strip()]
                    text = " ".join(non_empty_lines)
                    
                    if text:
                        print(f"Файл прочитан успешно ({len(text)} символов)")
                        return text
                        
                except UnicodeDecodeError:
                    continue
            
            print("Не удалось прочитать файл. Возможно, неподдерживаемая кодировка.")
            return None
            
        except Exception as e:
            print(f"Ошибка чтения файла: {e}")
            return None
    
    def _get_text_from_console(self) -> Optional[str]:
        print("Введите текст для анализа:")
        text = input().strip()
        
        if not text:
            print("Текст не может быть пустым.")
            return None
        
        return text
    
    async def _with_exception_handling(self, async_action: Callable, *args):
        try:
            await async_action(*args)
        except Exception as ex:
            print(f"Произошла ошибка: {ex}")


if __name__ == "__main__":
    program = Program()
    try:
        asyncio.run(program.main())
    except KeyboardInterrupt:
        print("\n\nПрограмма прервана. До свидания!")
    except Exception as e:
        print(f"Критическая ошибка: {e}")