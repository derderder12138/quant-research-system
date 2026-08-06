"""
后台分析引擎 — 线程池异步执行，session_state 跨页面共享进度。
允许用户在分析进行中自由切换页面，分析完成后自动通知。
"""

import threading
import time
from typing import List, Dict, Any, Callable, Optional
from concurrent.futures import ThreadPoolExecutor


class BackendEngine:
    """
    全局单例后台引擎 — 管理批量分析任务的启动、进度跟踪与结果回调。
    通过 streamlit.session_state 实现跨页面状态共享。
    """

    def __init__(self):
        self._executor: Optional[ThreadPoolExecutor] = None
        self._lock = threading.Lock()

    # ---- session_state 操作（由外部注入 st.session_state）----
    @staticmethod
    def init_session(st_session) -> None:
        """在 app 启动时调用一次，初始化 session_state 中的引擎字段。"""
        defaults = {
            "analysis_running": False,
            "analysis_total": 0,
            "analysis_done": 0,
            "analysis_current": "",
            "analysis_results": [],
            "analysis_start_time": None,
            "analysis_log": [],
        }
        for k, v in defaults.items():
            if k not in st_session:
                st_session[k] = v

    @staticmethod
    def set_progress(st_session, done: int, total: int, current: str, log_msg: str = "") -> None:
        """更新 session_state 中的进度信息。"""
        st_session["analysis_done"] = done
        st_session["analysis_total"] = total
        st_session["analysis_current"] = current
        if log_msg:
            st_session["analysis_log"].append(f"[{time.strftime('%H:%M:%S')}] {log_msg}")

    @staticmethod
    def mark_complete(st_session, results: List[Dict]) -> None:
        st_session["analysis_running"] = False
        st_session["analysis_results"] = results
        st_session["analysis_log"].append(f"[{time.strftime('%H:%M:%S')}] ====== 分析完成 ======")

    @staticmethod
    def mark_failed(st_session, error: str) -> None:
        st_session["analysis_running"] = False
        st_session["analysis_log"].append(f"[{time.strftime('%H:%M:%S')}] 分析异常: {error}")

    # ---- 实际执行 ----
    def run_batch(
        self,
        tickers: List[str],
        max_workers: int,
        runner_fn: Callable[[str], Dict[str, Any]],
        save_fn: Callable[[str, Dict[str, Any]], Any],
        st_session,
    ) -> None:
        """
        在后台线程中启动批量分析。

        Args:
            tickers: 股票代码列表
            max_workers: 线程池大小
            runner_fn: 单支股票分析函数 ticker -> final_state dict
            save_fn: 结果保存函数 db_path, final_state -> row_id
            st_session: streamlit.session_state
        """
        if st_session.get("analysis_running"):
            return  # 已经在运行

        st_session["analysis_running"] = True
        st_session["analysis_total"] = len(tickers)
        st_session["analysis_done"] = 0
        st_session["analysis_current"] = ""
        st_session["analysis_results"] = []
        st_session["analysis_start_time"] = time.time()
        st_session["analysis_log"] = [f"[{time.strftime('%H:%M:%S')}] 启动分析: {len(tickers)} 支 × {max_workers} 线程"]

        def _worker():
            results = []
            total = len(tickers)
            try:
                with ThreadPoolExecutor(max_workers=max_workers) as pool:
                    futures = {pool.submit(runner_fn, t): t for t in tickers}
                    for i, future in enumerate(futures):
                        ticker = futures[future]
                        try:
                            final = future.result()
                        except Exception as e:
                            final = {"ticker": ticker, "data_fetch_success": False, "error_message": str(e)}
                        try:
                            save_fn(final)
                        except Exception:
                            pass
                        results.append(final)
                        self.set_progress(st_session, i + 1, total, ticker, f"{ticker}: {'OK' if final.get('data_fetch_success') else 'FAIL'}")
                self.mark_complete(st_session, results)
            except Exception as e:
                self.mark_failed(st_session, str(e))

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()


# 全局单例
engine = BackendEngine()
