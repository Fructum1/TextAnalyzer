import asyncio
import json
import subprocess
import re
import platform
import zipfile
import tarfile
from typing import Dict, List, Optional
from dataclasses import dataclass
import sys
import os
import urllib.request
import shutil


@dataclass
class Result:
    original_word: str
    lemma: Optional[str] = None
    grammar_info: Optional[str] = None
    part_of_speech: Optional[str] = None


class MyStemProcessor:
    def __init__(self, mystem_path: Optional[str] = None):
        self.mystem_path = mystem_path
        if not self.mystem_path:
            self.mystem_path = self._find_or_download_mystem()
    
    def _download_mystem(self) -> str:
        system = platform.system().lower()
        arch = platform.machine().lower()
        
        if system == "windows":
            url = "https://download.cdn.yandex.net/mystem/mystem-3.1-win-64bit.zip"
            archive_name = "mystem.zip"
            executable_name = "mystem.exe"
        elif system == "darwin":
            if "arm" in arch:
                url = "https://download.cdn.yandex.net/mystem/mystem-3.1-macosx-11-arm64.tar.gz"
            else:  # Intel Mac
                url = "https://download.cdn.yandex.net/mystem/mystem-3.1-macosx-10.12.tar.gz"
            archive_name = "mystem.tar.gz"
            executable_name = "mystem"
        else:
            url = "https://download.cdn.yandex.net/mystem/mystem-3.1-linux-64bit.tar.gz"
            archive_name = "mystem.tar.gz"
            executable_name = "mystem"
        
        current_dir = os.path.dirname(os.path.realpath(__file__))
        mystem_dir = os.path.join(current_dir, "mystem_bin")
        os.makedirs(mystem_dir, exist_ok=True)
        
        archive_path = os.path.join(mystem_dir, archive_name)
        executable_path = os.path.join(mystem_dir, executable_name)
        
        print(f"Загрузка MyStem из {url}...")
        try:
            urllib.request.urlretrieve(url, archive_path)
            print("Загрузка выполнена успешно.")
        except Exception as e:
            raise Exception(f"Ошибка загрузки MyStem: {e}")
        
        try:
            if archive_path.endswith('.zip'):
                with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                    zip_ref.extractall(mystem_dir)
            else:
                with tarfile.open(archive_path, 'r:gz') as tar_ref:
                    tar_ref.extractall(mystem_dir)
            print("Распаковка выполнена успешно.")
        except Exception as e:
            raise Exception(f"Ошибка при распаковки MyStem: {e}")
        
        # Удаляем архив
        try:
            os.remove(archive_path)
        except:
            pass
        
        if system != "windows":
            try:
                os.chmod(executable_path, 0o755)
            except Exception as e:
                print(f"MyStem невозможно сделать исполняемым: {e}")
        
        # Проверяем, что файл существует и доступен
        if not os.path.exists(executable_path):
            # Ищем распакованный файл
            for file in os.listdir(mystem_dir):
                if file.lower().startswith('mystem') and not file.endswith('.exe' if system == 'windows' else '.dll'):
                    candidate = os.path.join(mystem_dir, file)
                    if os.path.isfile(candidate):
                        executable_path = candidate
                        break
        
        if not os.path.exists(executable_path):
            raise FileNotFoundError(f"MyStem не найден после распаковки по пути {mystem_dir}")
        
        return executable_path
    
    def _find_or_download_mystem(self) -> str:
        existing_path = self._find_mystem()
        if existing_path:
            return existing_path
        
        print("MyStem не найден. Скачивание...")
        return self._download_mystem()
    
    async def analyze_text(self, text: str) -> Dict[str, Result]:
        results = {}
        
        try:
            if not self.mystem_path:
                raise FileNotFoundError("MyStem executable not found")
            
            process = await asyncio.create_subprocess_exec(
                self.mystem_path,
                "-nig", "--format", "json",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate(
                text.encode('utf-8')
            )
            
            try:
                await asyncio.wait_for(process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
            
            if stderr:
                print(f"MyStem errors: {stderr.decode('utf-8', errors='ignore')}")
            
            output = stdout.decode('utf-8', errors='ignore')
            return self.parse_my_stem_json_output(output)
            
        except Exception as ex:
            print(f"MyStem processing error: {ex}")
            return results
    
    def parse_my_stem_json_output(self, json_output: str) -> Dict[str, Result]:
        results = {}
        
        if not json_output:
            return results
        
        try:
            lines = [line.strip() for line in json_output.split('\n') if line.strip()]
            
            for line in lines:
                try:
                    result = self.parse_my_stem_line(line)
                    if result and result.original_word:
                        if result.lemma:
                            result.lemma = self.fix_my_stem_encoding(result.lemma)
                        results[result.original_word.lower()] = result
                except Exception as ex:
                    print(f"Error parsing MyStem line: {ex}")
                    
        except Exception as ex:
            print(f"Error parsing MyStem output: {ex}")
        
        return results
    
    def parse_my_stem_line(self, json_line: str) -> Optional[Result]:
        if not json_line or ('"text"' not in json_line and '"analysis"' not in json_line):
            return None
        
        try:
            data = json.loads(json_line)
        except json.JSONDecodeError:
            return None
        
        original_word = data.get("text", "")
        
        if not original_word:
            return None
        
        analysis_list = data.get("analysis", [])
        if not analysis_list:
            return Result(original_word=original_word)
        
        first_analysis = analysis_list[0]
        lemma = first_analysis.get("lex", "")
        grammar_info = first_analysis.get("gr", "")
        
        part_of_speech = self.extract_part_of_speech(grammar_info)
        
        return Result(
            original_word=original_word,
            lemma=lemma,
            grammar_info=grammar_info,
            part_of_speech=part_of_speech
        )
    
    def fix_my_stem_encoding(self, text: str) -> str:
        if not text:
            return text
            
        if self.is_valid_russian_text(text):
            return text
        
        encodings_to_try = ['windows-1251', 'cp866', 'koi8-r']
        
        for encoding_name in encodings_to_try:
            try:
                encoded_bytes = text.encode('iso-8859-1')
                decoded = encoded_bytes.decode(encoding_name)
                
                if self.is_valid_russian_text(decoded):
                    return decoded
            except (UnicodeEncodeError, UnicodeDecodeError):
                continue
        
        return text
    
    @staticmethod
    def is_valid_russian_text(text: str) -> bool:
        """Проверяет, является ли текст валидным русским текстом"""
        russian_pattern = re.compile(r'^[а-яёА-ЯЁ\s\-]+$')
        return bool(russian_pattern.match(text))
    
    @staticmethod
    def extract_part_of_speech(grammar_info: str) -> str:
        if not grammar_info:
            return "UNKN"
        
        parts = grammar_info.split(',')
        if not parts:
            return "UNKN"
        
        pos_map = {
            "S": "NOUN",      # Существительное
            "A": "ADJ",       # Прилагательное  
            "V": "VERB",      # Глагол
            "ADV": "ADV",     # Наречие
            "PR": "PREP",     # Предлог
            "CONJ": "CONJ",   # Союз
            "PART": "PART",   # Частица
            "SPRO": "PRON",   # Местоимение
            "NUM": "NUM",     # Числительное
        }
        
        first_part = parts[0]
        return pos_map.get(first_part, "UNKN")
    
    def _find_mystem(self) -> Optional[str]:
        system = platform.system().lower()
        current_dir = os.path.dirname(os.path.realpath(__file__))
        
        possible_paths = []
        
        if system == "windows":
            possible_paths.extend([
                "mystem.exe",
                "./mystem.exe",
                os.path.join(current_dir, "mystem.exe"),
            ])
        else:
            possible_paths.extend([
                "mystem",
                "./mystem", 
                "/usr/bin/mystem",
                "/usr/local/bin/mystem",
                os.path.join(current_dir, "mystem"),
            ])
        
        mystem_bin_dir = os.path.join(current_dir, "mystem_bin")
        if os.path.exists(mystem_bin_dir):
            for file in os.listdir(mystem_bin_dir):
                file_path = os.path.join(mystem_bin_dir, file)
                if os.path.isfile(file_path):
                    if system == "windows" and file.endswith('.exe'):
                        possible_paths.append(file_path)
                    elif system != "windows" and not file.endswith('.exe'):
                        possible_paths.append(file_path)
        
        for path in possible_paths:
            if os.path.exists(path):
                if system == "windows" and not path.endswith('.exe'):
                    continue

                if system != "windows" and not os.access(path, os.X_OK):
                    try:
                        os.chmod(path, 0o755)
                    except:
                        continue
                return path
        
        return None