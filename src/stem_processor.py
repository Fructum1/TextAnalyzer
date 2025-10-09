import asyncio
import json
import subprocess
import re
from typing import Dict, List, Optional
from dataclasses import dataclass
import sys
import os


@dataclass
class Result:
    original_word: str
    lemma: Optional[str] = None
    grammar_info: Optional[str] = None
    part_of_speech: Optional[str] = None


class MyStemProcessor:
    def __init__(self, mystem_path: Optional[str] = None):
        self.mystem_path = mystem_path or self._find_mystem()
    
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
            
            # Отправляем текст в mystem
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
        """Ищет исполняемый файл mystem в системе"""
        possible_paths = [
            "mystem",
            "mystem.exe",
            "./mystem",
            "./mystem.exe",
            "/usr/bin/mystem",
            "/usr/local/bin/mystem",
        ]
        
        current_dir = os.path.dirname(os.path.realpath(__file__))
        possible_paths.append(os.path.join(current_dir, "mystem"))
        possible_paths.append(os.path.join(current_dir, "mystem.exe"))

        for path in possible_paths:
            if os.path.exists(path) and os.access(path, os.X_OK):
                return path
        
        return None