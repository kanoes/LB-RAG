"""
翻訳機能モジュール
多言語対応のための翻訳機能を提供する
"""

import logging
from typing import Optional, List, Dict, Any
from enum import Enum

from deep_translator import GoogleTranslator
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException

from ..config import config

# 言語検出の結果を一定にするため
DetectorFactory.seed = 0

logger = logging.getLogger(__name__)


class LanguageCode(Enum):
    """サポートする言語コード"""
    JAPANESE = "ja"
    ENGLISH = "en"
    CHINESE = "zh"
    KOREAN = "ko"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    PORTUGUESE = "pt"
    RUSSIAN = "ru"
    ITALIAN = "it"


class TranslationService:
    """翻訳サービスクラス"""
    
    def __init__(self):
        """初期化"""
        self.supported_languages = {
            LanguageCode.JAPANESE.value: "日本語",
            LanguageCode.ENGLISH.value: "英語",
            LanguageCode.CHINESE.value: "中国語",
            LanguageCode.KOREAN.value: "韓国語",
            LanguageCode.SPANISH.value: "スペイン語",
            LanguageCode.FRENCH.value: "フランス語",
            LanguageCode.GERMAN.value: "ドイツ語",
            LanguageCode.PORTUGUESE.value: "ポルトガル語",
            LanguageCode.RUSSIAN.value: "ロシア語",
            LanguageCode.ITALIAN.value: "イタリア語"
        }
    
    def detect_language(self, text: str) -> Optional[str]:
        """
        テキストの言語を検出する
        
        Args:
            text: 検出対象のテキスト
            
        Returns:
            Optional[str]: 検出された言語コード（検出できない場合はNone）
        """
        try:
            if not text or len(text.strip()) < 3:
                return None
                
            detected_lang = detect(text)
            logger.info(f"言語検出結果: {detected_lang}")
            return detected_lang
            
        except LangDetectException as e:
            logger.warning(f"言語検出に失敗しました: {e}")
            return None
        except Exception as e:
            logger.error(f"言語検出中にエラーが発生しました: {e}")
            return None
    
    def translate_text(self, text: str, target_lang: str, source_lang: str = "auto") -> Optional[str]:
        """
        テキストを翻訳する
        
        Args:
            text: 翻訳対象のテキスト
            target_lang: 翻訳先の言語コード
            source_lang: 翻訳元の言語コード（autoで自動検出）
            
        Returns:
            Optional[str]: 翻訳されたテキスト（失敗時はNone）
        """
        try:
            if not text or not text.strip():
                return text
            
            # 同じ言語の場合は翻訳不要
            if source_lang == target_lang:
                return text
            
            # 言語を検出して同じ場合は翻訳不要
            if source_lang == "auto":
                detected_lang = self.detect_language(text)
                if detected_lang == target_lang:
                    return text
            
            # GoogleTranslatorを使用して翻訳
            translator = GoogleTranslator(source=source_lang, target=target_lang)
            translated = translator.translate(text)
            
            logger.info(f"翻訳完了: {source_lang} -> {target_lang}")
            return translated
            
        except Exception as e:
            logger.error(f"翻訳中にエラーが発生しました: {e}")
            return None
    
    def translate_query_to_english(self, query: str) -> str:
        """
        クエリを英語に翻訳する
        
        Args:
            query: 翻訳対象のクエリ
            
        Returns:
            str: 英語に翻訳されたクエリ（失敗時は元のクエリ）
        """
        try:
            # 英語かどうかを判定
            detected_lang = self.detect_language(query)
            if detected_lang == LanguageCode.ENGLISH.value:
                return query
            
            # 英語に翻訳
            translated = self.translate_text(query, LanguageCode.ENGLISH.value, detected_lang)
            if translated:
                logger.info(f"クエリを英語に翻訳しました: '{query}' -> '{translated}'")
                return translated
            else:
                logger.warning("クエリの翻訳に失敗しました。元のクエリを使用します。")
                return query
                
        except Exception as e:
            logger.error(f"クエリ翻訳中にエラーが発生しました: {e}")
            return query
    
    def translate_to_user_language(self, text: str, user_lang: str) -> str:
        """
        テキストをユーザーの言語に翻訳する
        
        Args:
            text: 翻訳対象のテキスト
            user_lang: ユーザーの言語コード
            
        Returns:
            str: ユーザーの言語に翻訳されたテキスト（失敗時は元のテキスト）
        """
        try:
            # 同じ言語の場合は翻訳不要
            detected_lang = self.detect_language(text)
            if detected_lang == user_lang:
                return text
            
            # ユーザーの言語に翻訳
            translated = self.translate_text(text, user_lang, detected_lang)
            if translated:
                logger.info(f"テキストをユーザー言語に翻訳しました: {detected_lang} -> {user_lang}")
                return translated
            else:
                logger.warning("テキストの翻訳に失敗しました。元のテキストを使用します。")
                return text
                
        except Exception as e:
            logger.error(f"ユーザー言語翻訳中にエラーが発生しました: {e}")
            return text
    
    def translate_documents_to_language(self, documents: List[Dict[str, Any]], target_lang: str) -> List[Dict[str, Any]]:
        """
        ドキュメントリストを指定言語に翻訳する
        
        Args:
            documents: 翻訳対象のドキュメント（page_contentフィールドを含む）
            target_lang: 翻訳先の言語コード
            
        Returns:
            List[Dict[str, Any]]: 翻訳されたドキュメント
        """
        translated_docs = []
        
        for doc in documents:
            try:
                # ドキュメントの内容を翻訳
                content = doc.get("page_content", "")
                if content:
                    translated_content = self.translate_text(content, target_lang)
                    if translated_content:
                        doc_copy = doc.copy()
                        doc_copy["page_content"] = translated_content
                        translated_docs.append(doc_copy)
                    else:
                        # 翻訳に失敗した場合は元のドキュメントを使用
                        translated_docs.append(doc)
                else:
                    translated_docs.append(doc)
                    
            except Exception as e:
                logger.error(f"ドキュメント翻訳中にエラーが発生しました: {e}")
                translated_docs.append(doc)
        
        logger.info(f"{len(translated_docs)}件のドキュメントを{target_lang}に翻訳しました")
        return translated_docs
    
    def get_supported_languages(self) -> Dict[str, str]:
        """
        サポートする言語の一覧を取得する
        
        Returns:
            Dict[str, str]: 言語コードと言語名のマッピング
        """
        return self.supported_languages.copy()
    
    def is_supported_language(self, lang_code: str) -> bool:
        """
        指定された言語がサポートされているかチェックする
        
        Args:
            lang_code: 言語コード
            
        Returns:
            bool: サポートされている場合True
        """
        return lang_code in self.supported_languages
    
    def get_language_name(self, lang_code: str) -> str:
        """
        言語コードから言語名を取得する
        
        Args:
            lang_code: 言語コード
            
        Returns:
            str: 言語名（見つからない場合は言語コードをそのまま返す）
        """
        return self.supported_languages.get(lang_code, lang_code) 