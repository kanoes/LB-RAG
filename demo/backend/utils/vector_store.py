"""
ベクトルストア管理モジュール
ドキュメントの埋め込みとベクトルストアの管理を行う
"""

import logging
from typing import List, Optional, Dict, Any
from pathlib import Path

from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

from ..config import config

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """ベクトルストア管理クラス"""
    
    def __init__(self, collection_name: str = "documents"):
        """
        初期化
        
        Args:
            collection_name: コレクション名
        """
        self.collection_name = collection_name
        self.persist_directory = Path(config.vector_store_persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # 埋め込みモデルを初期化
        self.embeddings = self._initialize_embeddings()
        
        # ベクトルストアを初期化
        self.vector_store = None
        self._load_or_create_vector_store()
    
    def _initialize_embeddings(self):
        """埋め込みモデルを初期化"""
        try:
            # OpenAI埋め込みを優先的に使用
            if config.validate_openai_key():
                logger.info("OpenAI埋め込みモデルを使用します")
                return OpenAIEmbeddings(
                    openai_api_key=config.openai_api_key,
                    model="text-embedding-ada-002"
                )
            else:
                logger.info("HuggingFace埋め込みモデルを使用します")
                return HuggingFaceEmbeddings(
                    model_name=config.embedding_model,
                    model_kwargs={'device': 'cpu'}
                )
        except Exception as e:
            logger.error(f"埋め込みモデルの初期化中にエラーが発生しました: {e}")
            # フォールバック: HuggingFaceEmbeddingsを使用
            return HuggingFaceEmbeddings(
                model_name=config.embedding_model,
                model_kwargs={'device': 'cpu'}
            )
    
    def _load_or_create_vector_store(self):
        """ベクトルストアを読み込むか作成する"""
        try:
            # 既存のベクトルストアが存在するかチェック
            if self.persist_directory.exists() and any(self.persist_directory.iterdir()):
                logger.info("既存のベクトルストアを読み込みます")
                self.vector_store = Chroma(
                    collection_name=self.collection_name,
                    embedding_function=self.embeddings,
                    persist_directory=str(self.persist_directory)
                )
            else:
                logger.info("新しいベクトルストアを作成します")
                self.vector_store = Chroma(
                    collection_name=self.collection_name,
                    embedding_function=self.embeddings,
                    persist_directory=str(self.persist_directory)
                )
        except Exception as e:
            logger.error(f"ベクトルストアの初期化中にエラーが発生しました: {e}")
            raise
    
    def add_documents(self, documents: List[Document]) -> bool:
        """
        ドキュメントをベクトルストアに追加する
        
        Args:
            documents: 追加するドキュメント
            
        Returns:
            bool: 成功したかどうか
        """
        try:
            if not documents:
                logger.warning("追加するドキュメントがありません")
                return False
            
            # ドキュメントをベクトルストアに追加
            self.vector_store.add_documents(documents)
            
            # 永続化
            self.vector_store.persist()
            
            logger.info(f"{len(documents)}件のドキュメントをベクトルストアに追加しました")
            return True
            
        except Exception as e:
            logger.error(f"ドキュメント追加中にエラーが発生しました: {e}")
            return False
    
    def search_similar(self, query: str, k: int = 5) -> List[Document]:
        """
        類似ドキュメントを検索する
        
        Args:
            query: 検索クエリ
            k: 取得する文書数
            
        Returns:
            List[Document]: 類似ドキュメント
        """
        try:
            if not self.vector_store:
                logger.error("ベクトルストアが初期化されていません")
                return []
            
            # 類似性検索を実行
            docs = self.vector_store.similarity_search(query, k=k)
            
            logger.info(f"クエリ '{query}' に対して{len(docs)}件の類似ドキュメントを取得しました")
            return docs
            
        except Exception as e:
            logger.error(f"検索中にエラーが発生しました: {e}")
            return []
    
    def search_with_scores(self, query: str, k: int = 5) -> List[tuple]:
        """
        スコア付きで類似ドキュメントを検索する
        
        Args:
            query: 検索クエリ
            k: 取得する文書数
            
        Returns:
            List[tuple]: (Document, score)のタプルリスト
        """
        try:
            if not self.vector_store:
                logger.error("ベクトルストアが初期化されていません")
                return []
            
            # スコア付き類似性検索を実行
            docs_with_scores = self.vector_store.similarity_search_with_score(query, k=k)
            
            logger.info(f"クエリ '{query}' に対して{len(docs_with_scores)}件の類似ドキュメントを取得しました")
            return docs_with_scores
            
        except Exception as e:
            logger.error(f"スコア付き検索中にエラーが発生しました: {e}")
            return []
    
    def search_with_threshold(self, query: str, similarity_threshold: float = 0.7, max_results: int = 3) -> List[tuple]:
        """
        閾値を使用してフィルタリングした類似ドキュメントを検索する
        
        Args:
            query: 検索クエリ
            similarity_threshold: 類似度の閾値（この値以上のもののみ返す）
            max_results: 最大取得件数（0-3）
            
        Returns:
            List[tuple]: (Document, score)のタプルリスト（閾値以上のもののみ）
        """
        try:
            if not self.vector_store:
                logger.error("ベクトルストアが初期化されていません")
                return []
            
            # より多めに検索して閾値でフィルタリング
            search_k = min(max_results * 2, 10)  # 最大10件で検索
            docs_with_scores = self.vector_store.similarity_search_with_score(query, k=search_k)
            
            # 閾値でフィルタリング（Chromaでは距離が小さいほど類似度が高い）
            # similarity_threshold を距離に変換（1 - similarity_threshold）
            distance_threshold = 1.0 - similarity_threshold
            
            filtered_docs = []
            for doc, distance in docs_with_scores:
                if distance <= distance_threshold:
                    similarity = 1.0 - distance  # 距離を類似度に変換
                    filtered_docs.append((doc, similarity))
            
            # 最大結果数でカット
            filtered_docs = filtered_docs[:max_results]
            
            logger.info(f"クエリ '{query}' で閾値 {similarity_threshold} 以上の類似ドキュメントを{len(filtered_docs)}件取得しました")
            return filtered_docs
            
        except Exception as e:
            logger.error(f"閾値付き検索中にエラーが発生しました: {e}")
            return []
    
    def get_retriever(self, k: int = 5):
        """
        リトリーバーを取得する
        
        Args:
            k: 取得する文書数
            
        Returns:
            VectorStoreRetriever: リトリーバー
        """
        try:
            if not self.vector_store:
                logger.error("ベクトルストアが初期化されていません")
                return None
            
            return self.vector_store.as_retriever(search_kwargs={"k": k})
            
        except Exception as e:
            logger.error(f"リトリーバー取得中にエラーが発生しました: {e}")
            return None
    
    def delete_documents_by_filename(self, filename: str) -> bool:
        """
        指定されたファイル名のドキュメントを削除する
        
        Args:
            filename: 削除するファイル名
            
        Returns:
            bool: 成功したかどうか
        """
        try:
            if not self.vector_store:
                logger.error("ベクトルストアが初期化されていません")
                return False
            
            # ファイル名でフィルタリングしてドキュメントを削除
            collection = self.vector_store._collection
            
            # ファイル名に基づいてドキュメントIDを取得
            results = collection.get(
                where={"filename": filename}
            )
            
            if results['ids']:
                # 該当するドキュメントを削除
                collection.delete(ids=results['ids'])
                
                # 永続化
                self.vector_store.persist()
                
                logger.info(f"ファイル '{filename}' に関連する{len(results['ids'])}件のドキュメントを削除しました")
                return True
            else:
                logger.warning(f"ファイル '{filename}' に関連するドキュメントが見つかりません")
                return False
                
        except Exception as e:
            logger.error(f"ファイル削除中にエラーが発生しました: {e}")
            return False
    
    def get_stored_filenames(self) -> List[str]:
        """
        保存されているファイル名のリストを取得する
        
        Returns:
            List[str]: ファイル名のリスト
        """
        try:
            if not self.vector_store:
                logger.error("ベクトルストアが初期化されていません")
                return []
            
            collection = self.vector_store._collection
            results = collection.get(
                include=["metadatas"]
            )
            
            # ファイル名を抽出（重複を排除）
            filenames = set()
            for metadata in results['metadatas']:
                if metadata and 'filename' in metadata:
                    filenames.add(metadata['filename'])
            
            return sorted(list(filenames))
            
        except Exception as e:
            logger.error(f"ファイル名取得中にエラーが発生しました: {e}")
            return []
    
    def get_file_document_count(self, filename: str) -> int:
        """
        指定されたファイルのドキュメント数を取得する
        
        Args:
            filename: ファイル名
            
        Returns:
            int: ドキュメント数
        """
        try:
            if not self.vector_store:
                return 0
            
            collection = self.vector_store._collection
            results = collection.get(
                where={"filename": filename}
            )
            
            return len(results['ids']) if results['ids'] else 0
            
        except Exception as e:
            logger.error(f"ファイル '{filename}' のドキュメント数取得中にエラーが発生しました: {e}")
            return 0
    
    def clear_vector_store(self):
        """ベクトルストアをクリアする"""
        try:
            if self.vector_store:
                # コレクションを削除
                self.vector_store.delete_collection()
                
                # 新しいベクトルストアを作成
                self._load_or_create_vector_store()
                
                logger.info("ベクトルストアをクリアしました")
                return True
        except Exception as e:
            logger.error(f"ベクトルストアクリア中にエラーが発生しました: {e}")
            return False
    
    def get_collection_count(self) -> int:
        """コレクション内のドキュメント数を取得"""
        try:
            if not self.vector_store:
                return 0
            
            # Chromaのコレクションからドキュメント数を取得
            collection = self.vector_store._collection
            return collection.count()
            
        except Exception as e:
            logger.error(f"ドキュメント数取得中にエラーが発生しました: {e}")
            return 0 