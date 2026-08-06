"""
批量调度引擎 —— 多线程并发分析、进度显示、结构化日志、故障隔离、结果持久化。
"""

import os
import sys
import logging
import re
import yaml
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

# tqdm 用于终端进度条
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from graph_builder import build_graph
from graph_types import StockAgentState
from data_fetcher import configure_fetcher
from agents import init_llm
from database import init_db, save_result, get_summary


def _setup_logging(log_level: str, log_file: str) -> logging.Logger:
    """配置双通道日志：控制台 + 文件。"""
    logger = logging.getLogger("stock_agent")
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    logger.handlers.clear()

    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 文件 handler
    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # 控制台 handler（INFO 级别，避免干扰 tqdm）
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.WARNING)  # 控制台只显示 WARNING+
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger


def _parse_watchlist(file_path: str) -> List[str]:
    """
    解析股票池文件。每行一个代码，支持 # 注释和空行。
    返回去重、去空白后的 ticker 列表。
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"股票池文件不存在: {file_path}")

    tickers = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            # 去掉注释和空白
            line = line.split("#")[0].strip()
            if line:
                tickers.append(line)

    # 去重保持顺序
    seen = set()
    unique = []
    for t in tickers:
        if t not in seen:
            seen.add(t)
            unique.append(t)

    if not unique:
        raise ValueError(f"股票池为空: {file_path}")

    return unique


def _make_initial_state(ticker: str) -> StockAgentState:
    """为单个 ticker 构造初始状态字典。"""
    return {
        "ticker": ticker,
        "data_fetch_success": False,
        "error_message": "",
        "raw_history_data": {},
        "technical_analysis": "",
        "fundamental_analysis": "",
        "final_report": ""
    }


class BatchRunner:
    """批量量化分析调度器。"""

    def __init__(self, config_path: str = "config.yaml"):
        """
        Args:
            config_path: YAML 配置文件路径
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.logger = _setup_logging(
            self.config.get("logging", {}).get("level", "INFO"),
            self.config.get("logging", {}).get("file", "logs/batch_run.log")
        )

        # 解析股票池
        watchlist_file = self.config.get("watchlist", {}).get("file", "watchlist.txt")
        self.tickers = _parse_watchlist(watchlist_file)
        self.logger.info(f"加载股票池: {len(self.tickers)} 支 — {self.tickers}")

        # 运行时参数
        runtime = self.config.get("runtime", {})
        self.max_workers = runtime.get("max_workers", 5)
        self.retry_attempts = runtime.get("retry_attempts", 3)
        self.retry_delay = runtime.get("retry_delay", 2)

        # 数据库路径
        self.db_path = self.config.get("database", {}).get("path", "data/analysis.db")

    def _load_config(self) -> Dict[str, Any]:
        """加载 YAML 配置文件。"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"配置文件不存在: {self.config_path}")

        with open(self.config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        if config is None:
            raise ValueError(f"配置文件为空: {self.config_path}")
        return config

    def _init_runtime(self) -> None:
        """初始化全局运行时：LLM、数据获取配置、数据库。"""
        # LLM
        llm_cfg = self.config.get("llm", {})
        api_key = os.getenv(llm_cfg.get("api_key_env", "OPENAI_API_KEY"))
        api_base = os.getenv(llm_cfg.get("api_base_env", "OPENAI_API_BASE"))

        if not api_key:
            raise RuntimeError(
                f"未找到 API Key 环境变量: {llm_cfg.get('api_key_env', 'OPENAI_API_KEY')}"
            )

        init_llm(
            model=llm_cfg.get("model", "deepseek-chat"),
            api_key=api_key,
            api_base=api_base or "https://api.deepseek.com/v1"
        )
        self.logger.info(f"LLM 初始化: model={llm_cfg.get('model', 'deepseek-chat')}")

        # 数据获取配置
        configure_fetcher(
            retry_attempts=self.retry_attempts,
            retry_delay=self.retry_delay
        )

        # 数据库
        init_db(self.db_path)
        self.logger.info(f"数据库就绪: {self.db_path}")

    def _run_single_stock(self, ticker: str) -> Dict[str, Any]:
        """
        对单支股票执行完整的 LangGraph 分析管线。
        任何异常都会被捕获并返回错误状态，绝不向上传播。

        Args:
            ticker: A 股代码

        Returns:
            最终状态字典（无论成功或失败）
        """
        start_time = datetime.now()
        self.logger.info(f"[{ticker}] 开始分析...")

        try:
            initial_state = _make_initial_state(ticker)
            graph = build_graph()
            final_state = graph.invoke(initial_state)

            elapsed = (datetime.now() - start_time).total_seconds()
            self.logger.info(
                f"[{ticker}] 分析完成 — "
                f"数据={'OK' if final_state['data_fetch_success'] else 'FAIL'}, "
                f"耗时={elapsed:.1f}s"
            )
            return final_state

        except Exception as e:
            elapsed = (datetime.now() - start_time).total_seconds()
            self.logger.error(f"[{ticker}] 分析异常 — {e} (耗时 {elapsed:.1f}s)")
            return {
                "ticker": ticker,
                "data_fetch_success": False,
                "error_message": f"管线异常: {str(e)}",
                "raw_history_data": {},
                "technical_analysis": "",
                "fundamental_analysis": "",
                "final_report": ""
            }

    def run_batch(self) -> Dict[str, Any]:
        """
        主入口：批量执行所有股票分析。

        Returns:
            汇总统计字典
        """
        print("=" * 60)
        print("🚀 启动工业级批量量化投研系统")
        print(f"   股票池: {len(self.tickers)} 支")
        print(f"   并发数: {self.max_workers} 线程")
        print(f"   数据库: {self.db_path}")
        print("=" * 60)

        # 初始化运行时
        self._init_runtime()

        # 编译图（全局复用一个实例）
        self.logger.info("编译 LangGraph 计算图...")

        # 批量执行
        results: List[Dict[str, Any]] = []
        has_tqdm = tqdm is not None

        desc = "分析进度"
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_ticker = {
                executor.submit(self._run_single_stock, t): t
                for t in self.tickers
            }

            # 使用 tqdm 或简单计数器显示进度
            if has_tqdm:
                pbar = tqdm(
                    as_completed(future_to_ticker),
                    total=len(self.tickers),
                    desc=desc,
                    unit="支",
                    ncols=80
                )
            else:
                pbar = as_completed(future_to_ticker)

            completed = 0
            for future in pbar:
                ticker = future_to_ticker[future]
                try:
                    final_state = future.result()
                except Exception as e:
                    # 极端兜底：future.result() 本身抛异常
                    self.logger.error(f"[{ticker}] 线程级异常: {e}")
                    final_state = {
                        "ticker": ticker,
                        "data_fetch_success": False,
                        "error_message": f"线程异常: {str(e)}",
                        "raw_history_data": {},
                        "technical_analysis": "",
                        "fundamental_analysis": "",
                        "final_report": ""
                    }

                # 持久化
                try:
                    row_id = save_result(self.db_path, final_state)
                    self.logger.info(f"[{ticker}] 已写入数据库 (row={row_id})")
                except Exception as e:
                    self.logger.error(f"[{ticker}] 数据库写入失败: {e}")

                results.append(final_state)
                completed += 1

                # 更新进度条后缀
                success = final_state.get("data_fetch_success", False)
                status_icon = "✓" if success else "✗"
                if has_tqdm and isinstance(pbar, tqdm):
                    pbar.set_postfix_str(f"{status_icon} {ticker}")

        # 汇总统计
        summary = self._print_summary(results)

        # 数据库汇总
        db_summary = get_summary(self.db_path)
        if db_summary["total"] > 0:
            print(f"\n📊 数据库汇总 (全部历史):")
            print(f"   总记录: {db_summary['total']} | 成功: {db_summary['success']} | 失败: {db_summary['failed']}")
            if db_summary["ratings"]:
                print(f"   评级分布: {db_summary['ratings']}")

        print(f"\n📝 详细日志: {self.config.get('logging', {}).get('file', 'logs/batch_run.log')}")

        return summary

    def _print_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """打印批量运行汇总报告。"""
        total = len(results)
        success = sum(1 for r in results if r.get("data_fetch_success"))
        failed = total - success

        print("\n" + "=" * 60)
        print("📊 批量运行汇总报告")
        print("=" * 60)
        print(f"   总股票数: {total}")
        print(f"   数据成功: {success}  ({success/total*100:.1f}%)" if total > 0 else "   数据成功: 0")
        print(f"   数据失败: {failed}  ({failed/total*100:.1f}%)" if total > 0 else "   数据失败: 0")

        # 列出失败的股票
        if failed > 0:
            print(f"\n❌ 数据获取失败的股票:")
            for r in results:
                if not r.get("data_fetch_success"):
                    err = r.get("error_message", "未知错误")
                    print(f"   - {r['ticker']}: {err[:80]}")

        # 列出成功的股票及其评级
        if success > 0:
            print(f"\n✅ 分析完成的股票:")
            # 尝试提取评级
            rating_pat = re.compile(r"\[(积极建仓|谨慎持有|观望等待|减仓回避)\]")
            for r in results:
                if r.get("data_fetch_success"):
                    ticker = r["ticker"]
                    report = r.get("final_report", "")
                    match = rating_pat.search(report)
                    rating = match.group(1) if match else "未识别"
                    print(f"   - {ticker}: {rating}")

        return {
            "total": total,
            "success": success,
            "failed": failed,
            "timestamp": datetime.now().isoformat()
        }
