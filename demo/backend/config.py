"""
設定管理モジュール
アプリケーション全体の設定を管理する
"""

import os
from typing import Optional
from pydantic import BaseModel
from dotenv import load_dotenv

# 環境変数を読み込み
load_dotenv()


class Config(BaseModel):
    """アプリケーション設定クラス"""
    
    # OpenAI API設定
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    openai_model: str = "gpt-4o-mini"
    openai_temperature: float = 0.5
    openai_max_tokens: int = 4096
    
    # 埋め込みモデル設定
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # ベクトルストア設定
    vector_store_persist_directory: str = "./data/vector_store"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    # RAG検索設定
    similarity_threshold: float = 0.5
    max_results: int = 3
    
    # 翻訳設定
    translation_service: str = "google"  # google, deepl
    supported_languages: list = ["ja", "en", "zh", "ko", "es", "fr", "de"]
    
    # UI設定
    page_title: str = "多言語RAG問答システム"
    page_icon: str = "🤖"
    
    # ログ設定
    log_level: str = "INFO"
    
    def validate_openai_key(self) -> bool:
        """OpenAI APIキーの有効性をチェック"""
        return self.openai_api_key is not None and len(self.openai_api_key) > 0


# グローバル設定インスタンス
config = Config() 