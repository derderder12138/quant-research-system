"""
部署前健康检查 — 推送前运行，一项失败就拒绝推送。
用法: python health_check.py
"""

import sys, traceback

FAILED = 0
PASSED = 0


def check(name, fn):
    global PASSED, FAILED
    try:
        result = fn()
        if result is False:
            FAILED += 1
            print(f"  ❌ {name} — 断言失败")
        else:
            PASSED += 1
            print(f"  ✅ {name}")
    except Exception as e:
        FAILED += 1
        print(f"  💥 {name} — {type(e).__name__}: {str(e)[:100]}")


# ============================================
print("1. 模块导入链")
# ============================================
modules = [
    "database", "graph_builder", "agents", "data_fetcher",
    "real_time", "stock_universe", "charts", "risk_metrics",
    "strategy", "strategy_custom", "fundamental_data",
    "industry", "signals", "backend_engine",
]
for m in modules:
    check(f"import {m}", lambda m=m: __import__(m))

# ============================================
print("\n2. 数据库核心")
# ============================================
import database as db
check("verify_user", lambda: db.verify_user("health_check", "pass123"))
check("save_result", lambda: db.save_result("health_check", {
    "ticker": "600519", "data_fetch_success": True, "error_message": "",
    "technical_analysis": "OK", "fundamental_analysis": "OK", "final_report": "OK"
}) > 0)
check("get_results", lambda: len(db.get_results("health_check", limit=1)) > 0)
check("get_summary", lambda: db.get_summary("health_check")["total"] > 0)
check("get_watchlist", lambda: isinstance(db.get_watchlist("health_check"), list))
check("save_note", lambda: not db.save_note("health_check", "600519", "test") and True)
check("get_note", lambda: isinstance(db.get_note("health_check", "600519"), str))

# ============================================
print("\n3. 实时行情")
# ============================================
import real_time as rt
check("get_index_quotes", lambda: len(rt.get_index_quotes()) >= 2)
check("get_realtime_quotes", lambda: any(q.get("price", 0) > 0 for q in rt.get_realtime_quotes(["600519", "000001"])))
check("validate_tickers", lambda: len(rt.validate_tickers(["600519", "999999"])[0]) == 1)
check("get_quotes_batched", lambda: len(rt.get_quotes_batched(["600519", "000001", "300750"])) >= 2)

# ============================================
print("\n4. 全市场搜索")
# ============================================
import stock_universe as su
try:
    su.refresh_universe(force=False)
except Exception:
    pass  # 云端可能不可用，允许降级
check("search_stocks", lambda: su.search_stocks("茅台")[0]["code"] == "600519")
check("get_universe_stats", lambda: su.get_universe_stats()["total"] >= 5000)
check("get_by_board", lambda: len(su.get_by_board("创业板", 10)) > 0)

# ============================================
print("\n5. 业务模块（不联网）")
# ============================================
import graph_builder as gb
check("build_graph", lambda: gb.build_graph() is not None)

import risk_metrics as rm
import pandas as pd, numpy as np
dummy = pd.DataFrame({"close": np.random.randn(100).cumsum() + 100,
                      "open": np.random.randn(100).cumsum() + 100,
                      "high": np.random.randn(100).cumsum() + 102,
                      "low": np.random.randn(100).cumsum() + 98,
                      "volume": np.random.randint(1000, 10000, 100)})
check("calculate_metrics", lambda: "error" not in rm.calculate_metrics(dummy))

# ============================================
print("\n6. app.py 语法")
# ============================================
import py_compile
def _compile_ok():
    try: py_compile.compile("app.py", doraise=True); return True
    except py_compile.PyCompileError: return False
check("app.py 编译", _compile_ok)

# ============================================
print(f"\n{'='*40}")
print(f"  通过: {PASSED}  |  失败: {FAILED}")
print(f"{'='*40}")
if FAILED > 0:
    print("❌ 健康检查未通过，禁止推送！请修复后重试。")
    sys.exit(1)
else:
    print("✅ 全部通过，可以安全推送。")
    sys.exit(0)
