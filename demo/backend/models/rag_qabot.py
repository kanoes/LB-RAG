"""
RAGベースのQAボット
検索拡張生成（RAG）を使用した高度な問答機能を提供する
"""

import logging
from typing import List, Dict, Any, Optional

from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.llms import HuggingFacePipeline

from ..config import config
from ..utils.document_processor import DocumentProcessor
from ..utils.vector_store import VectorStoreManager
from ..utils.translation import TranslationService

logger = logging.getLogger(__name__)


class RAGQABot:
    """RAGベースのQAボット"""
    
    def __init__(self, collection_name: str = "rag_documents"):
        """
        初期化
        
        Args:
            collection_name: ベクトルストアのコレクション名
        """
        self.collection_name = collection_name
        self.document_processor = DocumentProcessor()
        self.vector_store_manager = VectorStoreManager(collection_name)
        self.translation_service = TranslationService()
        self.llm = None
        self.qa_chain = None
        self._initialize_llm()
        self._initialize_qa_chain()
    
    def _initialize_llm(self):
        """LLMを初期化"""
        try:
            # OpenAIを優先的に使用
            if config.validate_openai_key():
                logger.info("OpenAI ChatGPTモデルを使用します")
                self.llm = ChatOpenAI(
                    openai_api_key=config.openai_api_key,
                    model_name=config.openai_model,
                    temperature=config.openai_temperature,
                    max_tokens=config.openai_max_tokens
                )
            else:
                logger.warning("OpenAI APIキーが設定されていません。HuggingFaceモデルを使用します")
                # HuggingFaceのフォールバック（実装は簡略化）
                self.llm = None
                
        except Exception as e:
            logger.error(f"LLM初期化中にエラーが発生しました: {e}")
            self.llm = None
    
    def _initialize_qa_chain(self):
        """QAチェーンを初期化"""
        try:
            if not self.llm:
                logger.warning("LLMが初期化されていません。QAチェーンを作成できません")
                return
            
            # プロンプトテンプレートを作成
            prompt_template = """ユーザの質問に答えてください。必要である場合、参考資料を参照してください。

参考資料: {context}

ユーザ質問: {question}"""
            
            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
            
            # リトリーバーを取得（ドキュメントがない場合はNone）
            retriever = self.vector_store_manager.get_retriever(k=5)
            
            if retriever:
                # RetrievalQAチェーンを作成
                self.qa_chain = RetrievalQA.from_chain_type(
                    llm=self.llm,
                    chain_type="stuff",
                    retriever=retriever,
                    chain_type_kwargs={"prompt": prompt},
                    return_source_documents=True
                )
                logger.info("QAチェーンの初期化が完了しました")
            else:
                # ドキュメントがない場合もQAチェーンを作成（空のリトリーバー使用）
                logger.info("ドキュメントなしでQAチェーンを初期化します")
                self.qa_chain = "no_documents_chain"  # 特別なマーカー
                
        except Exception as e:
            logger.error(f"QAチェーン初期化中にエラーが発生しました: {e}")
            self.qa_chain = None
    
    def load_documents_from_json(self, json_data: List[Dict[str, Any]]) -> bool:
        """
        JSONデータからドキュメントを読み込む
        
        Args:
            json_data: JSON形式のドキュメントデータ
            
        Returns:
            bool: 成功したかどうか
        """
        try:
            # JSONデータを処理
            documents = self.document_processor.process_json_documents(json_data)
            if not documents:
                logger.warning("処理可能なドキュメントがありません")
                return False
            
            # ドキュメントを分割
            split_documents = self.document_processor.split_documents(documents)
            
            # ドキュメントを検証
            valid_documents = self.document_processor.validate_documents(split_documents)
            
            # ベクトルストアに追加
            success = self.vector_store_manager.add_documents(valid_documents)
            
            if success:
                # QAチェーンを再初期化
                self._initialize_qa_chain()
                logger.info(f"{len(valid_documents)}件のドキュメントを読み込みました")
                return True
            else:
                logger.error("ベクトルストアへの追加に失敗しました")
                return False
                
        except Exception as e:
            logger.error(f"ドキュメント読み込み中にエラーが発生しました: {e}")
            return False
    
    def load_documents_from_file(self, file_path: str) -> bool:
        """
        ファイルからドキュメントを読み込む
        
        Args:
            file_path: JSONファイルのパス
            
        Returns:
            bool: 成功したかどうか
        """
        try:
            # ファイルからJSONデータを読み込む
            json_data = self.document_processor.load_json_from_file(file_path)
            if not json_data:
                logger.warning("ファイルからデータを読み込めませんでした")
                return False
            
            # ドキュメントを読み込む
            return self.load_documents_from_json(json_data)
            
        except Exception as e:
            logger.error(f"ファイル読み込み中にエラーが発生しました: {e}")
            return False
    
    def load_documents_from_uploaded_file(self, file_content: bytes, filename: str) -> bool:
        """
        アップロードされたファイルからドキュメントを読み込む
        
        Args:
            file_content: ファイルの内容（バイナリデータ）
            filename: ファイル名
            
        Returns:
            bool: 成功したかどうか
        """
        try:
            # アップロードされたファイルを処理
            documents = self.document_processor.process_uploaded_file(file_content, filename)
            if not documents:
                logger.warning("処理可能なドキュメントがありません")
                return False
            
            # ドキュメントを分割
            split_documents = self.document_processor.split_documents(documents)
            
            # ドキュメントを検証
            valid_documents = self.document_processor.validate_documents(split_documents)
            
            # ベクトルストアに追加
            success = self.vector_store_manager.add_documents(valid_documents)
            
            if success:
                # QAチェーンを再初期化
                self._initialize_qa_chain()
                logger.info(f"{len(valid_documents)}件のドキュメントを読み込みました")
                return True
            else:
                logger.error("ベクトルストアへの追加に失敗しました")
                return False
                
        except Exception as e:
            logger.error(f"ドキュメント読み込み中にエラーが発生しました: {e}")
            return False
    
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
            if not self.qa_chain:
                return {
                    "success": False,
                    "answer": "RAGシステムが初期化されていません。",
                    "error": "RAG system not initialized"
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
            
            # ドキュメントがない場合の処理
            if self.qa_chain == "no_documents_chain":
                # 直接LLMを呼び出し、参考資料なしで回答
                prompt_text = f"""ユーザの質問に答えてください。必要である場合、参考資料を参照してください。

参考資料: 参考資料なし

ユーザ質問: {question}"""
                
                response = self.llm.invoke(prompt_text)
                answer = response.content if hasattr(response, 'content') else str(response)
                
                return {
                    "success": True,
                    "answer": answer,
                    "detected_language": detected_lang,
                    "sources": [],
                    "model": config.openai_model,
                    "document_count": 0
                }
            
            # 通常のRAG処理：閾値を使用してドキュメントを検索
            similarity_threshold = kwargs.get("similarity_threshold", config.similarity_threshold)
            max_results = kwargs.get("max_results", config.max_results)
            
            # 閾値を使用してドキュメントを検索
            docs_with_scores = self.vector_store_manager.search_with_threshold(
                question, similarity_threshold, max_results
            )
            
            # 検索結果からcontextを構築
            if docs_with_scores:
                context_parts = []
                for doc, score in docs_with_scores:
                    context_parts.append(doc.page_content)
                context = "\n\n".join(context_parts)
            else:
                context = "参考資料なし"
            
            # プロンプトを直接構築して回答を生成
            prompt_text = f"""ユーザの質問に答えてください。必要である場合、参考資料を参照してください。

参考資料: {context}

ユーザ質問: {question}"""
            
            response = self.llm.invoke(prompt_text)
            answer = response.content if hasattr(response, 'content') else str(response)
            
            # ソースドキュメントの情報を抽出
            sources = []
            for doc, score in docs_with_scores:
                sources.append({
                    "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    "metadata": doc.metadata,
                    "similarity_score": score
                })
            
            return {
                "success": True,
                "answer": answer,
                "detected_language": detected_lang,
                "sources": sources,
                "model": config.openai_model,
                "document_count": len(docs_with_scores),
                "similarity_threshold": similarity_threshold
            }
            
        except Exception as e:
            logger.error(f"回答生成中にエラーが発生しました: {e}")
            return {
                "success": False,
                "answer": f"申し訳ございません。エラーが発生しました: {str(e)}",
                "error": str(e)
            }
    
    def search_documents(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        ドキュメントを検索する
        
        Args:
            query: 検索クエリ
            k: 取得する文書数
            
        Returns:
            List[Dict[str, Any]]: 検索結果
        """
        try:
            docs_with_scores = self.vector_store_manager.search_with_scores(query, k)
            
            results = []
            for doc, score in docs_with_scores:
                results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "similarity_score": score
                })
            
            return results
            
        except Exception as e:
            logger.error(f"ドキュメント検索中にエラーが発生しました: {e}")
            return []
    
    def delete_file_documents(self, filename: str) -> bool:
        """
        指定されたファイルのドキュメントを削除する
        
        Args:
            filename: 削除するファイル名
            
        Returns:
            bool: 成功したかどうか
        """
        try:
            success = self.vector_store_manager.delete_documents_by_filename(filename)
            
            if success:
                # QAチェーンを再初期化
                self._initialize_qa_chain()
                logger.info(f"ファイル '{filename}' のドキュメントを削除しました")
                return True
            else:
                logger.error(f"ファイル '{filename}' の削除に失敗しました")
                return False
                
        except Exception as e:
            logger.error(f"ファイル削除中にエラーが発生しました: {e}")
            return False
    
    def get_stored_files(self) -> List[Dict[str, Any]]:
        """
        保存されているファイルのリストを取得する
        
        Returns:
            List[Dict[str, Any]]: ファイル情報のリスト
        """
        try:
            filenames = self.vector_store_manager.get_stored_filenames()
            
            files_info = []
            for filename in filenames:
                doc_count = self.vector_store_manager.get_file_document_count(filename)
                files_info.append({
                    "filename": filename,
                    "document_count": doc_count,
                    "file_type": filename.split('.')[-1].lower() if '.' in filename else 'unknown'
                })
            
            return files_info
            
        except Exception as e:
            logger.error(f"ファイル一覧取得中にエラーが発生しました: {e}")
            return []
    
    def clear_documents(self) -> bool:
        """
        ドキュメントをクリアする
        
        Returns:
            bool: 成功したかどうか
        """
        try:
            success = self.vector_store_manager.clear_vector_store()
            if success:
                self.qa_chain = None
                logger.info("ドキュメントをクリアしました")
            return success
            
        except Exception as e:
            logger.error(f"ドキュメントクリア中にエラーが発生しました: {e}")
            return False
    
    def get_document_count(self) -> int:
        """
        ドキュメント数を取得する
        
        Returns:
            int: ドキュメント数
        """
        return self.vector_store_manager.get_collection_count()
    
    def get_bot_info(self) -> Dict[str, Any]:
        """
        ボットの情報を取得する
        
        Returns:
            Dict[str, Any]: ボット情報
        """
        return {
            "name": "RAG QA Bot",
            "description": "検索拡張生成（RAG）を使用した高度な問答ボット",
            "version": "1.0",
            "model": config.openai_model,
            "features": [
                "ドキュメント検索",
                "コンテキスト活用回答",
                "ソース情報提供",
                "多言語対応"
            ],
            "document_count": self.get_document_count(),
            "api_available": config.validate_openai_key(),
            "rag_available": self.qa_chain is not None
        }
    
    def has_documents(self) -> bool:
        """
        知識库にドキュメントが存在するかチェックする
        
        Returns:
            bool: ドキュメントが存在する場合True
        """
        return self.get_document_count() > 0
    
    def is_available(self) -> bool:
        """
        ボットが利用可能かチェックする
        
        Returns:
            bool: 利用可能な場合True
        """
        return self.qa_chain is not None and config.validate_openai_key() 