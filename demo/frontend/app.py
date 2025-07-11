"""
多言語RAG問答システム - Streamlit UI
3つのQAボットを統合したWebアプリケーション
"""

import streamlit as st
import json
import logging
from typing import Dict, Any, List
import sys
import os
from pathlib import Path

# バックエンドモジュールをインポートパスに追加
backend_root = Path(__file__).resolve().parents[1]   # demo/ を指す
sys.path.insert(0, str(backend_root))               # demo/ をパスに追加

from backend.config import config
from backend.models.simple_chatgpt import SimpleChatGPTBot
from backend.models.rag_qabot import RAGQABot
from backend.models.multilingual_rag import MultilingualRAGBot
from backend.utils.translation import TranslationService

# ロギング設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QABotApp:
    """QAボットアプリケーション"""
    
    def __init__(self):
        """初期化"""
        self.setup_page_config()
        self.initialize_session_state()
        self.translation_service = TranslationService()
    
    def setup_page_config(self):
        """ページ設定"""
        st.set_page_config(
            page_title=config.page_title,
            page_icon=config.page_icon,
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    def initialize_session_state(self):
        """セッション状態の初期化"""
        if 'current_bot' not in st.session_state:
            st.session_state.current_bot = "simple_chatgpt"
        
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = {
                "simple_chatgpt": [],
                "rag_qabot": [],
                "multilingual_rag": []
            }
        
        if 'bots' not in st.session_state:
            st.session_state.bots = {}
        
        if 'documents_loaded' not in st.session_state:
            st.session_state.documents_loaded = {
                "rag_qabot": False,
                "multilingual_rag": False
            }
    
    def initialize_bots(self):
        """ボットの初期化"""
        try:
            # Simple ChatGPT Bot
            if 'simple_chatgpt' not in st.session_state.bots:
                st.session_state.bots['simple_chatgpt'] = SimpleChatGPTBot()
            
            # RAG QA Bot
            if 'rag_qabot' not in st.session_state.bots:
                st.session_state.bots['rag_qabot'] = RAGQABot()
            
            # Multilingual RAG Bot
            if 'multilingual_rag' not in st.session_state.bots:
                st.session_state.bots['multilingual_rag'] = MultilingualRAGBot()
            
            # 文書読み込み状態を実際の知識库状態と同期
            self._sync_document_states()
                
        except Exception as e:
            st.error(f"ボットの初期化中にエラーが発生しました: {e}")
            logger.error(f"ボット初期化エラー: {e}")
    
    def _sync_document_states(self):
        """文書読み込み状態を実際の知識库状態と同期"""
        try:
            for bot_type in ["rag_qabot", "multilingual_rag"]:
                if bot_type in st.session_state.bots:
                    bot = st.session_state.bots[bot_type]
                    has_docs = bot.has_documents()
                    st.session_state.documents_loaded[bot_type] = has_docs
                    logger.info(f"{bot_type}の文書状態を同期しました: {has_docs}")
        except Exception as e:
            logger.error(f"文書状態同期中にエラーが発生しました: {e}")
    
    def render_sidebar(self):
        """サイドバーの描画"""
        with st.sidebar:
            st.title("QAボット選択")
            
            # ボット選択
            bot_options = {
                "simple_chatgpt": "1. シンプルChatGPT",
                "rag_qabot": "2. RAGベースQA",
                "multilingual_rag": "3. 多言語RAG"
            }
            
            selected_bot = st.selectbox(
                "使用するボットを選択してください：",
                options=list(bot_options.keys()),
                format_func=lambda x: bot_options[x],
                index=list(bot_options.keys()).index(st.session_state.current_bot)
            )
            
            if selected_bot != st.session_state.current_bot:
                st.session_state.current_bot = selected_bot
                st.rerun()
            
            st.divider()
            
            # ボット情報表示
            self.render_bot_info()
            
            st.divider()
            
            # 設定セクション
            self.render_settings()
            
            st.divider()
            
            # ドキュメント管理（RAGボットの場合）
            if st.session_state.current_bot in ["rag_qabot", "multilingual_rag"]:
                self.render_document_management()
    
    def render_bot_info(self):
        """ボット情報の表示"""
        st.subheader("📊 ボット情報")
        
        current_bot = st.session_state.current_bot
        
        if current_bot in st.session_state.bots:
            bot = st.session_state.bots[current_bot]
            bot_info = bot.get_bot_info()
            
            st.write(f"**名前:** {bot_info['name']}")
            st.write(f"**説明:** {bot_info['description']}")
            st.write(f"**バージョン:** {bot_info['version']}")
            
            if bot_info.get('api_available', False):
                st.success("✅ API利用可能")
            else:
                st.error("❌ API利用不可（APIキーを確認してください）")
            
            # 機能一覧
            if 'features' in bot_info:
                st.write("**機能:**")
                for feature in bot_info['features']:
                    st.write(f"• {feature}")
            
            # RAGボット特有の情報
            if current_bot in ["rag_qabot", "multilingual_rag"]:
                if 'document_count' in bot_info:
                    st.write(f"**ドキュメント数:** {bot_info['document_count']}")
                
                if bot_info.get('rag_available', False):
                    st.success("✅ RAGシステム準備完了")
                else:
                    st.warning("⚠️ RAGシステム未準備（ドキュメントを読み込んでください）")
    
    def render_settings(self):
        """設定セクション"""
        st.subheader("⚙️ 設定")
        
        # 多言語RAGボットの場合の検索モード設定
        if st.session_state.current_bot == "multilingual_rag":
            search_mode = st.selectbox(
                "検索モード：",
                options=["dual", "translate_query", "direct"],
                format_func=lambda x: {
                    "dual": "デュアル検索（推奨）",
                    "translate_query": "翻訳検索",
                    "direct": "直接検索"
                }[x],
                key="search_mode"
            )
            
            st.write("**検索モード説明:**")
            st.write("• **デュアル検索**: 元の質問と翻訳した質問の両方で検索")
            st.write("• **翻訳検索**: 質問を英語に翻訳してから検索")
            st.write("• **直接検索**: 質問をそのまま使用して検索")
        
        # API設定状態表示
        if config.validate_openai_key():
            st.success("✅ OpenAI APIキー設定済み")
        else:
            st.error("❌ OpenAI APIキーが設定されていません")
            st.write("環境変数 `OPENAI_API_KEY` を設定してください")
    
    def render_document_management(self):
        """ドキュメント管理セクション"""
        st.subheader("📄 ドキュメント管理")
        
        current_bot = st.session_state.current_bot
        
        # RAGボットのみドキュメント管理を表示
        if current_bot not in ["rag_qabot", "multilingual_rag"]:
            st.info("このボットではドキュメント管理は必要ありません")
            return
        
        # ドキュメント読み込み状態
        if st.session_state.documents_loaded.get(current_bot, False):
            st.success("✅ ドキュメント読み込み済み")
        else:
            st.warning("⚠️ ドキュメント未読み込み")
        
        # サポートされているファイル形式を表示
        bot = st.session_state.bots.get(current_bot)
        if bot and hasattr(bot, 'document_processor'):
            supported_formats = bot.document_processor.get_supported_formats()
            st.info(f"サポートされるファイル形式: {', '.join(supported_formats)}")
        
        # ファイルアップロード
        st.write("**知識库文件をアップロードしてください:**")
        uploaded_files = st.file_uploader(
            "ファイルを選択（複数選択可）",
            type=['txt', 'pdf', 'docx', 'json'],
            key=f"file_upload_{current_bot}",
            help="txt、pdf、docx、jsonファイルをアップロードできます（複数選択可）",
            accept_multiple_files=True
        )
        
        # 多言語RAGボットの場合は翻訳オプションを表示
        translate_to_english = True
        if current_bot == "multilingual_rag":
            translate_to_english = st.checkbox(
                "ドキュメントを英語に翻訳して処理する",
                value=True,
                key=f"translate_option_{current_bot}",
                help="チェックすると、非英語ドキュメントを英語に翻訳してから処理します"
            )
        
        # ファイル処理ボタンを中央に配置
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            if st.button("📚 ファイルを処理", key=f"process_files_{current_bot}", use_container_width=True):
                if uploaded_files:
                    self.process_uploaded_files(uploaded_files, current_bot, translate_to_english)
                else:
                    st.warning("⚠️ ファイルをアップロードしてください")
        
        # 知識库文件リスト表示
        st.divider()
        self.render_knowledge_files(current_bot)
    
    def process_uploaded_files(self, uploaded_files, bot_type: str, translate_to_english: bool = True):
        """アップロードされた複数ファイルの処理"""
        try:
            if not uploaded_files:
                st.error("ファイルをアップロードしてください")
                return
            
            # ボットを取得
            bot = st.session_state.bots[bot_type]
            
            success_count = 0
            fail_count = 0
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, uploaded_file in enumerate(uploaded_files):
                # ファイルの内容を読み込み
                file_content = uploaded_file.read()
                filename = uploaded_file.name
                
                # 進捗更新
                progress = (i + 1) / len(uploaded_files)
                progress_bar.progress(progress)
                status_text.text(f"処理中: {filename} ({i + 1}/{len(uploaded_files)})")
                
                try:
                    # ファイル形式に応じて処理
                    if bot_type == "multilingual_rag":
                        success = bot.load_documents_from_uploaded_file(
                            file_content, filename, translate_to_english
                        )
                    else:
                        success = bot.load_documents_from_uploaded_file(file_content, filename)
                    
                    if success:
                        success_count += 1
                        logger.info(f"ファイル '{filename}' の処理に成功しました")
                    else:
                        fail_count += 1
                        logger.error(f"ファイル '{filename}' の処理に失敗しました")
                        
                except Exception as e:
                    fail_count += 1
                    logger.error(f"ファイル '{filename}' の処理中にエラーが発生しました: {e}")
            
            # 処理完了
            progress_bar.empty()
            status_text.empty()
            
            if success_count > 0:
                st.session_state.documents_loaded[bot_type] = True
                st.success(f"✅ {success_count}件のファイルを正常に処理しました")
                if fail_count > 0:
                    st.warning(f"⚠️ {fail_count}件のファイルの処理に失敗しました")
                st.rerun()
            else:
                st.error(f"❌ すべてのファイルの処理に失敗しました")
                
        except Exception as e:
            st.error(f"❌ ファイル処理中にエラーが発生しました: {e}")
            logger.error(f"ファイル処理エラー: {e}")
    
    def render_knowledge_files(self, bot_type: str):
        """知識库文件リストの表示"""
        try:
            st.subheader("📋 知識库文件一覧")
            
            # ボットを取得
            bot = st.session_state.bots.get(bot_type)
            if not bot:
                st.error("ボットが初期化されていません")
                return
            
            # 保存されているファイル名を直接取得
            stored_filenames = bot.vector_store_manager.get_stored_filenames()
            
            if not stored_filenames:
                st.info("知識库に文件がありません")
                return
            
            # ファイルごとに表示
            for i, filename in enumerate(stored_filenames):
                # ファイル情報を表示
                col1, col2 = st.columns([4, 1])
                
                with col1:
                    st.write(f"**{filename}**")
                
                with col2:
                    # 削除ボタン
                    if st.button("🗑️", key=f"delete_{bot_type}_{i}", help="削除"):
                        self.delete_knowledge_file(filename, bot_type)
                
                st.divider()
            
            # 統計情報
            total_files = len(stored_filenames)
            
            st.info(f"**合計:** {total_files}ファイル")
            
        except Exception as e:
            st.error(f"❌ ファイル一覧表示中にエラーが発生しました: {e}")
            logger.error(f"ファイル一覧表示エラー: {e}")
    
    def delete_knowledge_file(self, filename: str, bot_type: str):
        """知識库文件の削除"""
        try:
            # ボットを取得
            bot = st.session_state.bots[bot_type]
            
            with st.spinner(f"ファイル '{filename}' を削除中..."):
                success = bot.delete_file_documents(filename)
            
            if success:
                st.success(f"✅ ファイル '{filename}' を削除しました")
                
                # 残りのファイルがあるかチェック
                remaining_files = bot.get_stored_files()
                if not remaining_files:
                    st.session_state.documents_loaded[bot_type] = False
                
                st.rerun()
            else:
                st.error(f"❌ ファイル '{filename}' の削除に失敗しました")
                
        except Exception as e:
            st.error(f"❌ ファイル削除中にエラーが発生しました: {e}")
            logger.error(f"ファイル削除エラー: {e}")
    

    def render_main_content(self):
        """メインコンテンツの描画"""
        st.title("RAGデモアプリ")
        
        current_bot = st.session_state.current_bot
        
        # ボットの説明
        bot_descriptions = {
            "simple_chatgpt": "シンプルなChatGPT問答ボット：OpenAI APIを直接使用した基本的な質問応答",
            "rag_qabot": "RAGベースQAボット：ドキュメントを検索して文脈に基づいた回答を生成",
            "multilingual_rag": "多言語RAGボット：翻訳機能を活用した高度な多言語対応問答システム"
        }
        
        st.info(f"現在のボット: **{bot_descriptions[current_bot]}**")
        
        # チャット履歴の表示
        self.render_chat_history()
        
        # 質問入力フォーム
        self.render_question_form()
    
    def render_chat_history(self):
        """チャット履歴の表示"""
        current_bot = st.session_state.current_bot
        chat_history = st.session_state.chat_history[current_bot]
        
        if chat_history:
            st.subheader("💬 チャット履歴")
            
            # 最新の質問と回答を表示
            if chat_history:
                latest_question, latest_response = chat_history[-1]
                
                with st.container():
                    st.write(f"**最新の質問:** {latest_question}")
                    
                    if latest_response['success']:
                        st.write(f"**回答:** {latest_response['answer']}")
                        
                        # 追加情報の表示
                        if 'detected_language' in latest_response:
                            detected_lang = latest_response['detected_language']
                            lang_name = self.translation_service.get_language_name(detected_lang)
                            st.caption(f"検出言語: {lang_name}")
                        
                        # RAGボットの場合はソース情報を表示
                        if current_bot in ["rag_qabot", "multilingual_rag"] and 'sources' in latest_response:
                            with st.expander("📚 参考ソース"):
                                for j, source in enumerate(latest_response['sources']):
                                    st.write(f"**ソース {j+1}:**")
                                    st.write(source['content'])
                                    if 'similarity_score' in source:
                                        st.caption(f"類似度: {source['similarity_score']:.3f}")
                                    if source['metadata']:
                                        st.caption(f"メタデータ: {source['metadata']}")
                        
                        # RAGボットの場合は設定情報を表示
                        if current_bot in ["rag_qabot", "multilingual_rag"]:
                            settings_info = []
                            if 'similarity_threshold' in latest_response:
                                settings_info.append(f"類似度閾値: {latest_response['similarity_threshold']}")
                            if 'document_count' in latest_response:
                                settings_info.append(f"検索結果: {latest_response['document_count']}件")
                            if 'search_mode' in latest_response:
                                settings_info.append(f"検索モード: {latest_response['search_mode']}")
                            
                            if settings_info:
                                st.caption(" | ".join(settings_info))
                    else:
                        st.error(f"**エラー:** {latest_response['answer']}")
                    
                    st.divider()
            
            # 履歴が2件以上ある場合は、展開ボタンを表示
            if len(chat_history) > 1:
                with st.expander(f"📋 履歴を展開 ({len(chat_history) - 1}件の過去の質問)"):
                    # 最新以外の履歴を時系列順で表示
                    for i, (question, response) in enumerate(chat_history[:-1]):  # 最新を除く
                        with st.container():
                            st.write(f"**質問 {i+1}:** {question}")
                            
                            if response['success']:
                                st.write(f"**回答:** {response['answer']}")
                                
                                # 追加情報の表示
                                if 'detected_language' in response:
                                    detected_lang = response['detected_language']
                                    lang_name = self.translation_service.get_language_name(detected_lang)
                                    st.caption(f"検出言語: {lang_name}")
                                
                                # RAGボットの場合はソース情報を表示
                                if current_bot in ["rag_qabot", "multilingual_rag"] and 'sources' in response:
                                    with st.expander("📚 参考ソース", expanded=False):
                                        for j, source in enumerate(response['sources']):
                                            st.write(f"**ソース {j+1}:**")
                                            st.write(source['content'])
                                            if 'similarity_score' in source:
                                                st.caption(f"類似度: {source['similarity_score']:.3f}")
                                            if source['metadata']:
                                                st.caption(f"メタデータ: {source['metadata']}")
                                
                                # RAGボットの場合は設定情報を表示
                                if current_bot in ["rag_qabot", "multilingual_rag"]:
                                    settings_info = []
                                    if 'similarity_threshold' in response:
                                        settings_info.append(f"類似度閾値: {response['similarity_threshold']}")
                                    if 'document_count' in response:
                                        settings_info.append(f"検索結果: {response['document_count']}件")
                                    if 'search_mode' in response:
                                        settings_info.append(f"検索モード: {response['search_mode']}")
                                    
                                    if settings_info:
                                        st.caption(" | ".join(settings_info))
                            else:
                                st.error(f"**エラー:** {response['answer']}")
                            
                            st.divider()
    
    def render_question_form(self):
        """質問入力フォーム"""
        st.subheader("❓ 質問を入力してください")
        
        current_bot = st.session_state.current_bot
        
        # 質問入力
        with st.form(key=f"question_form_{current_bot}"):
            question = st.text_area(
                "質問:",
                height=100,
                placeholder="ここに質問を入力してください...",
                key=f"question_input_{current_bot}"
            )
            
            submitted = st.form_submit_button("🚀 質問を送信")
            
            if submitted and question.strip():
                self.process_question(question.strip())
    
    def process_question(self, question: str):
        """質問の処理"""
        current_bot = st.session_state.current_bot
        
        try:
            # ボットが初期化されているかチェック
            if current_bot not in st.session_state.bots:
                st.error("ボットが初期化されていません")
                return
            
            bot = st.session_state.bots[current_bot]
            
            # RAGボットの場合、ドキュメント状態を表示（警告ではなく情報として）
            if current_bot in ["rag_qabot", "multilingual_rag"]:
                if not st.session_state.documents_loaded.get(current_bot, False):
                    st.info("💡 ドキュメントが読み込まれていない場合、一般的な知識で回答します。より具体的な回答が必要な場合は、サイドバーからドキュメントを読み込んでください。")
            
            # 質問を処理
            with st.spinner("回答を生成中..."):
                # RAGボットの場合は設定を取得
                kwargs = {}
                if current_bot in ["rag_qabot", "multilingual_rag"]:
                    kwargs['similarity_threshold'] = config.similarity_threshold
                    kwargs['max_results'] = config.max_results
                
                # 多言語RAGボットの場合は検索モードも取得
                if current_bot == "multilingual_rag":
                    kwargs['search_mode'] = st.session_state.get('search_mode', 'dual')
                
                response = bot.generate_response(question, **kwargs)
            
            # チャット履歴に追加
            st.session_state.chat_history[current_bot].append((question, response))
            
            # 成功時の処理
            if response['success']:
                st.success("✅ 回答が生成されました")
                st.rerun()
            else:
                st.error(f"❌ エラー: {response['answer']}")
                
        except Exception as e:
            st.error(f"❌ 質問処理中にエラーが発生しました: {e}")
            logger.error(f"質問処理エラー: {e}")
    
    def run(self):
        """アプリケーションの実行"""
        try:
            # ボットの初期化
            self.initialize_bots()
            
            # サイドバーの描画
            self.render_sidebar()
            
            # メインコンテンツの描画
            self.render_main_content()
            
            # フッター
            st.markdown("---")
            st.markdown("💡 **使用方法:**")
            st.markdown("1. 左側のサイドバーから使用したいボットを選択")
            st.markdown("2. RAGボットの場合は、ドキュメントを読み込み")
            st.markdown("3. 質問を入力して送信")
            st.markdown("4. 回答とソース情報を確認")
            
        except Exception as e:
            st.error(f"アプリケーション実行中にエラーが発生しました: {e}")
            logger.error(f"アプリケーション実行エラー: {e}")


def main():
    """メイン関数"""
    app = QABotApp()
    app.run()


if __name__ == "__main__":
    main() 