import asyncio
import os
import sys
import argparse
from typing import Optional, Callable, Awaitable, List
from latent_semantic_analyzer import LatentSemanticAnalyzer
from sentiment_analyzer import SentimentAnalyzer

sys.path.append(os.path.dirname(__file__))

class Program:
    async def main(self, modes: list[str], input_file: Optional[str] = None, input_string: Optional[str] = None,
                   input_files2: Optional[List[str]] = None, input_strings2: Optional[List[str]] = None, 
                   compare_mode: str = "tdidf", num_topics: int = 2):
        """
        :param modes: Список режимов анализа ('sentiment', 'lsa' или оба).
        :param input_file: Путь к первому файлу (если передан -f).
        :param input_string: Первая входная строка (если передан -i).
        :param input_files2: Список путей к дополнительным файлам (если передан -f2).
        :param input_strings2: Список дополнительных входных строк (если передан -i2).
        :param compare_mode: Режим сранвения (только для LSA).
        :param num_topics: Количество тем для LSA анализа.
        :param top_similar: Флаг для вывода топ схожих документов.
        :param top_n: Количество топ схожих документов.
        """
        valid_modes = {'sentiment', 'lsa'}
        if not modes or not all(mode in valid_modes for mode in modes):
            raise ValueError("Режим должен быть 'sentiment', 'lsa' или их комбинация (например, 'sentiment,lsa')")

        documents = []
        doc_names = []

        if input_file:
            text = await self._get_text_from_file(input_file)
            if text is not None:
                documents.append(text)
                doc_names.append(os.path.basename(input_file))
            else:
                print(f"Предупреждение: не удалось загрузить файл {input_file}")

        if input_string:
            documents.append(input_string)
            doc_names.append("input_1")

        if input_files2:
            for f in input_files2:
                text = await self._get_text_from_file(f)
                if text is not None:
                    documents.append(text)
                    doc_names.append(os.path.basename(f))
                else:
                    print(f"Предупреждение: не удалось загрузить файл {f}")

        if input_strings2:
            for i, s in enumerate(input_strings2, start=2):
                if s:
                    documents.append(s)
                    doc_names.append(f"input_{i}")

        if not documents:
            raise ValueError("Не удалось получить тексты для анализа. Укажите хотя бы один ввод с -f, -i, -f2 или -i2")

        print(f"\nАнализируем документы...")

        if 'sentiment' in modes:
            sentiment_analyzer = SentimentAnalyzer()
            for idx, text in enumerate(documents, 1):
                print(f"\n=== Результат анализа тональности (для документа {doc_names[idx-1]}) ===")
                sentiment_result = await sentiment_analyzer.analyze(text)
                print(f"Тональность: {sentiment_result.sentiment}")
                print(f"Оценка: {sentiment_result.score:.3f}")
                print(f"Количество слов, учтенных в анализе: {sentiment_result.word_count}")

        if 'lsa' in modes:
            print(f"\n=== Результат анализа LSA (число тем: {num_topics}) ===")
            lsa = LatentSemanticAnalyzer(documents, num_top_words=5, k=num_topics)
            await lsa.fit()
            lsa.print_results()

            if compare_mode:
                if len(documents) < 2:
                    print("\nДля сравнения требуется хотя бы два документа")
                elif lsa.doc_vectors is not None:
                    try:
                        top_docs = lsa.document_similarity(compare_mode, 0)
                        if (top_docs):
                            print(f"\nСхожесть документа {doc_names[0]} с:")
                            for rank, (idx, sim) in enumerate(top_docs, 1):
                                name = doc_names[idx] if idx < len(doc_names) else f"документ {idx+1}"
                                print(f"{rank}. {name} (схожесть: {sim:.3f})")
                        else:
                            print("\nНевозможно вывести схожесть документа с другими, так как анализатор не воспринял ни одного документа для сравнения. Возможно, в документе отсутствуют русские слова и символы, либо документ состоит из необрабатываемых символов.")
                    except Exception as e:
                        print(f"\nОшибка при поиске похожих документов: {e}")
                else:
                    print("\nНе удалось вычислить векторы документов для сравнения")

    async def _get_text_from_file(self, file_path: str) -> Optional[str]:
        """
        Чтение текста из файла с поддержкой разных кодировок.

        :param file_path: Путь к файлу.
        :return: Текст или None при ошибке.
        """
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

    async def _with_exception_handling(self, async_action: Callable, *args):
        try:
            await async_action(*args)
        except Exception as ex:
            print(f"Произошла ошибка: {ex}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Анализатор текста: тональность или LSA")
    parser.add_argument('--mode', type=str, required=True, help="Режим анализа: 'sentiment', 'lsa' или 'sentiment,lsa'")
    parser.add_argument('-f', '--file', type=str, help="Путь к первому файлу с текстом")
    parser.add_argument('-i', '--input', type=str, help="Первая входная строка для анализа")
    parser.add_argument('--compare', type=str, help="Параметр для вывода схожести документов (доступные алгоритмы tdidf/w2v)")
    parser.add_argument('-f2', '--file2', type=str, nargs='*', help="Пути к дополнительным файлам с текстом")
    parser.add_argument('-i2', '--input2', type=str, nargs='*', help="Дополнительные входные строки для анализа")
    parser.add_argument('--num-topics', type=int, default=2, help="Количество тем для LSA анализа (по умолчанию 2)")

    args = parser.parse_args()
    modes = args.mode.split(',')

    program = Program()
    try:
        asyncio.run(program.main(modes, args.file, args.input, args.file2, args.input2,
                                 args.compare, args.num_topics))
    except RuntimeError as e:
        if "cannot be called from a running event loop" in str(e):
            loop = asyncio.get_event_loop()
            loop.run_until_complete(program.main(modes, args.file, args.input, args.file2, args.input2,
                                                 args.compare, args.num_topics))
        else:
            raise e
    except Exception as e:
        print(f"Критическая ошибка: {e}")