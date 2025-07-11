"""
多言語RAGボット
翻訳機能を組み込んだ多言語対応の高度なRAGベースQAボット
"""

import logging
from typing import List, Dict, Any, Optional

from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from ..config import config
from ..utils.document_processor import DocumentProcessor
from ..utils.vector_store import VectorStoreManager
from ..utils.translation import TranslationService, LanguageCode

logger = logging.getLogger(__name__)


class MultilingualRAGBot:
    """多言語RAGボット"""
    
    def __init__(self, collection_name: str = "multilingual_rag_documents"):
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
            if config.validate_openai_key():
                logger.info("多言語対応OpenAI ChatGPTモデルを使用します")
                self.llm = ChatOpenAI(
                    openai_api_key=config.openai_api_key,
                    model_name=config.openai_model,
                    temperature=config.openai_temperature,
                    max_tokens=config.openai_max_tokens
                )
            else:
                logger.warning("OpenAI APIキーが設定されていません")
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
            
            # 多言語対応プロンプトテンプレート
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
                logger.info("多言語QAチェーンの初期化が完了しました")
            else:
                # ドキュメントがない場合もQAチェーンを作成（空のリトリーバー使用）
                logger.info("ドキュメントなしで多言語QAチェーンを初期化します")
                self.qa_chain = "no_documents_chain"  # 特別なマーカー
                
        except Exception as e:
            logger.error(f"QAチェーン初期化中にエラーが発生しました: {e}")
            self.qa_chain = None
    
    def load_documents_from_json(self, json_data: List[Dict[str, Any]], 
                               translate_to_english: bool = True) -> bool:
        """
        JSONデータからドキュメントを読み込む
        
        Args:
            json_data: JSON形式のドキュメントデータ
            translate_to_english: 英語に翻訳するかどうか
            
        Returns:
            bool: 成功したかどうか
        """
        try:
            # JSONデータを処理
            documents = self.document_processor.process_json_documents(json_data)
            if not documents:
                logger.warning("処理可能なドキュメントがありません")
                return False
            
            # 多言語ドキュメントを処理
            if translate_to_english:
                documents = self._translate_documents_to_english(documents)
            
            # ドキュメントを分割
            split_documents = self.document_processor.split_documents(documents)
            
            # ドキュメントを検証
            valid_documents = self.document_processor.validate_documents(split_documents)
            
            # ベクトルストアに追加
            success = self.vector_store_manager.add_documents(valid_documents)
            
            if success:
                # QAチェーンを再初期化
                self._initialize_qa_chain()
                logger.info(f"{len(valid_documents)}件の多言語ドキュメントを読み込みました")
                return True
            else:
                logger.error("ベクトルストアへの追加に失敗しました")
                return False
                
        except Exception as e:
            logger.error(f"ドキュメント読み込み中にエラーが発生しました: {e}")
            return False
    
    def _translate_documents_to_english(self, documents: List[Document]) -> List[Document]:
        """
        ドキュメントを英語に翻訳する
        
        Args:
            documents: 翻訳対象のドキュメント
            
        Returns:
            List[Document]: 翻訳されたドキュメント
        """
        translated_documents = []
        
        for doc in documents:
            try:
                # 言語を検出
                detected_lang = self.translation_service.detect_language(doc.page_content)
                
                # 英語でない場合は翻訳
                if detected_lang != LanguageCode.ENGLISH.value:
                    translated_content = self.translation_service.translate_text(
                        doc.page_content, 
                        LanguageCode.ENGLISH.value, 
                        detected_lang
                    )
                    
                    if translated_content:
                        # 翻訳されたドキュメントを作成
                        translated_doc = Document(
                            page_content=translated_content,
                            metadata={
                                **doc.metadata,
                                "original_language": detected_lang,
                                "translated_to": LanguageCode.ENGLISH.value
                            }
                        )
                        translated_documents.append(translated_doc)
                        logger.info(f"ドキュメントを{detected_lang}から英語に翻訳しました")
                    else:
                        # 翻訳に失敗した場合は元のドキュメントを使用
                        translated_documents.append(doc)
                else:
                    # 既に英語の場合はそのまま使用
                    translated_documents.append(doc)
                    
            except Exception as e:
                logger.error(f"ドキュメント翻訳中にエラーが発生しました: {e}")
                translated_documents.append(doc)
        
        return translated_documents
    
    def load_documents_from_uploaded_file(self, file_content: bytes, filename: str, 
                                        translate_to_english: bool = True) -> bool:
        """
        アップロードされたファイルからドキュメントを読み込む
        
        Args:
            file_content: ファイルの内容（バイナリデータ）
            filename: ファイル名
            translate_to_english: 英語に翻訳するかどうか
            
        Returns:
            bool: 成功したかどうか
        """
        try:
            # アップロードされたファイルを処理
            documents = self.document_processor.process_uploaded_file(file_content, filename)
            if not documents:
                logger.warning("処理可能なドキュメントがありません")
                return False
            
            # 多言語ドキュメントを処理
            if translate_to_english:
                documents = self._translate_documents_to_english(documents)
            
            # ドキュメントを分割
            split_documents = self.document_processor.split_documents(documents)
            
            # ドキュメントを検証
            valid_documents = self.document_processor.validate_documents(split_documents)
            
            # ベクトルストアに追加
            success = self.vector_store_manager.add_documents(valid_documents)
            
            if success:
                # QAチェーンを再初期化
                self._initialize_qa_chain()
                logger.info(f"{len(valid_documents)}件の多言語ドキュメントを読み込みました")
                return True
            else:
                logger.error("ベクトルストアへの追加に失敗しました")
                return False
                
        except Exception as e:
            logger.error(f"ドキュメント読み込み中にエラーが発生しました: {e}")
            return False
    
    def generate_response(self, question: str, **kwargs) -> Dict[str, Any]:
        """
        質問に対する回答を生成する（多言語対応）
        
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
                    "answer": "多言語RAGシステムが初期化されていません。",
                    "error": "Multilingual RAG system not initialized"
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
                    "search_mode": "no_documents",
                    "model": config.openai_model,
                    "document_count": 0
                }
            
            # 多言語検索モードの選択
            search_mode = kwargs.get("search_mode", "dual")  # dual, translate_query, direct
            
            if search_mode == "dual":
                # デュアル検索：元の質問と翻訳した質問の両方で検索
                result = self._dual_search_and_generate(question, detected_lang, **kwargs)
            elif search_mode == "translate_query":
                # 翻訳検索：質問を英語に翻訳してから検索
                result = self._translate_query_and_generate(question, detected_lang, **kwargs)
            else:
                # 直接検索：質問をそのまま使用
                result = self._direct_search_and_generate(question, detected_lang, **kwargs)
            
            return result
            
        except Exception as e:
            logger.error(f"回答生成中にエラーが発生しました: {e}")
            return {
                "success": False,
                "answer": f"申し訳ございません。エラーが発生しました: {str(e)}",
                "error": str(e)
            }
    
    def _dual_search_and_generate(self, question: str, detected_lang: str, **kwargs) -> Dict[str, Any]:
        """デュアル検索を実行して回答を生成"""
        try:
            # 閾値を使用してドキュメントを検索
            similarity_threshold = kwargs.get("similarity_threshold", config.similarity_threshold)
            max_results = kwargs.get("max_results", config.max_results)
            
            # 元の質問で検索
            original_docs = self.vector_store_manager.search_with_threshold(
                question, similarity_threshold, max_results
            )
            
            # 質問を英語に翻訳して検索
            translated_question = self.translation_service.translate_query_to_english(question)
            translated_docs = self.vector_store_manager.search_with_threshold(
                translated_question, similarity_threshold, max_results
            )
            
            # 結果を統合
            combined_sources = []
            combined_sources.extend(original_docs)
            combined_sources.extend(translated_docs)
            
            # 重複を除去（スコアが高い方を優先）
            unique_sources = []
            seen_contents = set()
            for doc, score in combined_sources:
                if doc.page_content not in seen_contents:
                    unique_sources.append((doc, score))
                    seen_contents.add(doc.page_content)
            
            # スコアでソートして最大結果数でカット
            unique_sources.sort(key=lambda x: x[1], reverse=True)
            unique_sources = unique_sources[:max_results]
            
            # 検索結果からcontextを構築
            if unique_sources:
                context_parts = []
                for doc, score in unique_sources:
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
            for doc, score in unique_sources:
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
                "search_mode": "dual",
                "translated_question": translated_question,
                "model": config.openai_model,
                "document_count": len(unique_sources),
                "similarity_threshold": similarity_threshold
            }
            
        except Exception as e:
            logger.error(f"デュアル検索中にエラーが発生しました: {e}")
            return self._direct_search_and_generate(question, detected_lang, **kwargs)
    
    def _translate_query_and_generate(self, question: str, detected_lang: str, **kwargs) -> Dict[str, Any]:
        """翻訳検索を実行して回答を生成"""
        try:
            # 閾値を使用してドキュメントを検索
            similarity_threshold = kwargs.get("similarity_threshold", config.similarity_threshold)
            max_results = kwargs.get("max_results", config.max_results)
            
            # 質問を英語に翻訳
            translated_question = self.translation_service.translate_query_to_english(question)
            
            # 英語で検索
            docs_with_scores = self.vector_store_manager.search_with_threshold(
                translated_question, similarity_threshold, max_results
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
                "search_mode": "translate_query",
                "translated_question": translated_question,
                "model": config.openai_model,
                "document_count": len(docs_with_scores),
                "similarity_threshold": similarity_threshold
            }
            
        except Exception as e:
            logger.error(f"翻訳検索中にエラーが発生しました: {e}")
            return self._direct_search_and_generate(question, detected_lang, **kwargs)
    
    def _direct_search_and_generate(self, question: str, detected_lang: str, **kwargs) -> Dict[str, Any]:
        """直接検索を実行して回答を生成"""
        try:
            # 閾値を使用してドキュメントを検索
            similarity_threshold = kwargs.get("similarity_threshold", config.similarity_threshold)
            max_results = kwargs.get("max_results", config.max_results)
            
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
                "search_mode": "direct",
                "model": config.openai_model,
                "document_count": len(docs_with_scores),
                "similarity_threshold": similarity_threshold
            }
            
        except Exception as e:
            logger.error(f"直接検索中にエラーが発生しました: {e}")
            return {
                "success": False,
                "answer": f"検索中にエラーが発生しました: {str(e)}",
                "error": str(e)
            }
    
    def search_documents(self, query: str, k: int = 5, use_translation: bool = True) -> List[Dict[str, Any]]:
        """
        多言語ドキュメント検索
        
        Args:
            query: 検索クエリ
            k: 取得する文書数
            use_translation: 翻訳を使用するかどうか
            
        Returns:
            List[Dict[str, Any]]: 検索結果
        """
        try:
            results = []
            
            # 直接検索
            direct_results = self.vector_store_manager.search_with_scores(query, k)
            
            if use_translation:
                # 翻訳検索
                translated_query = self.translation_service.translate_query_to_english(query)
                if translated_query != query:
                    translated_results = self.vector_store_manager.search_with_scores(translated_query, k)
                    # 結果を統合
                    all_results = direct_results + translated_results
                else:
                    all_results = direct_results
            else:
                all_results = direct_results
            
            # 重複を除去してスコア順にソート
            unique_results = {}
            for doc, score in all_results:
                content_hash = hash(doc.page_content)
                if content_hash not in unique_results or unique_results[content_hash][1] > score:
                    unique_results[content_hash] = (doc, score)
            
            # 上位k件を取得
            sorted_results = sorted(unique_results.values(), key=lambda x: x[1])[:k]
            
            for doc, score in sorted_results:
                results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "similarity_score": score
                })
            
            return results
            
        except Exception as e:
            logger.error(f"多言語ドキュメント検索中にエラーが発生しました: {e}")
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
                logger.info(f"多言語ファイル '{filename}' のドキュメントを削除しました")
                return True
            else:
                logger.error(f"多言語ファイル '{filename}' の削除に失敗しました")
                return False
                
        except Exception as e:
            logger.error(f"多言語ファイル削除中にエラーが発生しました: {e}")
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
            logger.error(f"多言語ファイル一覧取得中にエラーが発生しました: {e}")
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
                logger.info("多言語ドキュメントをクリアしました")
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
    
    def get_supported_languages(self) -> Dict[str, str]:
        """
        サポートされている言語を取得する
        
        Returns:
            Dict[str, str]: 言語コードと言語名のマッピング
        """
        return self.translation_service.get_supported_languages()
    
    def get_bot_info(self) -> Dict[str, Any]:
        """
        ボットの情報を取得する
        
        Returns:
            Dict[str, Any]: ボット情報
        """
        return {
            "name": "Multilingual RAG Bot",
            "description": "多言語対応検索拡張生成（RAG）ボット",
            "version": "1.0",
            "model": config.openai_model,
            "features": [
                "多言語検索",
                "デュアル検索（元言語+翻訳）",
                "質問言語での回答",
                "言語自動検出",
                "ソース情報提供"
            ],
            "supported_languages": list(self.get_supported_languages().keys()),
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