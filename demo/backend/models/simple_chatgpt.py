"""
シンプルなChatGPT問答ボット
OpenAI APIを直接使用した基本的な問答機能を提供する
"""

import logging
from typing import Optional, Dict, Any

from openai import OpenAI
from langdetect import detect

from ..config import config
from ..utils.translation import TranslationService

logger = logging.getLogger(__name__)


class SimpleChatGPTBot:
    """シンプルなChatGPT問答ボット"""
    
    def __init__(self):
        """初期化"""
        self.client = None
        self.translation_service = TranslationService()
        self._initialize_client()
    
    def _initialize_client(self):
        """OpenAIクライアントを初期化"""
        try:
            if not config.validate_openai_key():
                raise ValueError("OpenAI APIキーが設定されていません")
            
            self.client = OpenAI(api_key=config.openai_api_key)
            logger.info("OpenAIクライアントの初期化が完了しました")
            
        except Exception as e:
            logger.error(f"OpenAIクライアントの初期化中にエラーが発生しました: {e}")
            raise
    
    def generate_response(self, question: str, **kwargs) -> Dict[str, Any]:
        """
        質問に対する回答を生成する
        
        Args:
            question: ユーザーの質問
            **kwargs: 追加パラメータ
            
        Returns:
            Dict[str, Any]: 回答結果
        """
        try:
            if not self.client:
                return {
                    "success": False,
                    "answer": "OpenAI APIが利用できません。APIキーを確認してください。",
                    "error": "API not available"
                }
            
            if not question or not question.strip():
                return {
                    "success": False,
                    "answer": "質問を入力してください。",
                    "error": "Empty question"
                }
            
            # 質問の言語を検出
            detected_lang = self.translation_service.detect_language(question)
            logger.info(f"検出された言語: {detected_lang}")
            
            # システムプロンプトを作成（言語に応じて）
            system_prompt = self._create_system_prompt(detected_lang)
            
            # ChatGPT APIを呼び出し
            response = self.client.chat.completions.create(
                model=config.openai_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question}
                ],
                temperature=config.openai_temperature,
                max_tokens=config.openai_max_tokens
            )
            
            answer = response.choices[0].message.content
            
            return {
                "success": True,
                "answer": answer,
                "detected_language": detected_lang,
                "model": config.openai_model,
                "tokens_used": response.usage.total_tokens if response.usage else 0
            }
            
        except Exception as e:
            logger.error(f"回答生成中にエラーが発生しました: {e}")
            return {
                "success": False,
                "answer": f"申し訳ございません。エラーが発生しました: {str(e)}",
                "error": str(e)
            }
    
    def _create_system_prompt(self, detected_lang: Optional[str] = None) -> str:
        """
        システムプロンプトを作成する
        
        Args:
            detected_lang: 検出された言語コード
            
        Returns:
            str: システムプロンプト
        """
        base_prompt = """あなたは親切で知識豊富なアシスタントです。
ユーザーの質問に対して、正確で分かりやすい回答を提供してください。
質問の言語で回答してください。"""
        
        if detected_lang:
            lang_name = self.translation_service.get_language_name(detected_lang)
            if detected_lang == "ja":
                base_prompt += f"\n\n質問は{lang_name}で書かれています。{lang_name}で回答してください。"
            elif detected_lang == "en":
                base_prompt += f"\n\nThe question is written in {lang_name}. Please respond in {lang_name}."
            elif detected_lang == "zh":
                base_prompt += f"\n\n问题是用{lang_name}写的。请用{lang_name}回答。"
            elif detected_lang == "ko":
                base_prompt += f"\n\n질문은 {lang_name}로 작성되었습니다. {lang_name}로 답변해주세요."
            else:
                base_prompt += f"\n\nThe question is in {lang_name}. Please respond in the same language."
        
        return base_prompt
    
    def get_bot_info(self) -> Dict[str, Any]:
        """
        ボットの情報を取得する
        
        Returns:
            Dict[str, Any]: ボット情報
        """
        return {
            "name": "Simple ChatGPT Bot",
            "description": "OpenAI ChatGPT APIを直接使用するシンプルな問答ボット",
            "version": "1.0",
            "model": config.openai_model,
            "features": [
                "多言語対応",
                "直接的なChatGPT問答",
                "言語検出機能"
            ],
            "api_available": config.validate_openai_key()
        }
    
    def is_available(self) -> bool:
        """
        ボットが利用可能かチェックする
        
        Returns:
            bool: 利用可能な場合True
        """
        return self.client is not None and config.validate_openai_key() 