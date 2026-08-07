"""
量化投研系统 — 专业前端。
多用户 · 持仓独立 · 异步批量分析 · 策略优化 · 智能风控。
"""

import os, sys, time, re
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed

import streamlit as st
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    for key in ["OPENAI_API_KEY", "OPENAI_API_BASE"]:
        if hasattr(st, "secrets") and key in st.secrets:  # type: ignore[reportOperatorIssue]
            os.environ[key] = str(st.secrets[key])  # type: ignore[reportArgumentType]
except Exception:
    from dotenv import load_dotenv; load_dotenv()

from database import (  # noqa: E402
    verify_user, get_summary, get_results, save_result, extract_rating,
    get_watchlist, add_to_watchlist, remove_from_watchlist, get_watchlist_names,
)
from graph_builder import build_graph  # noqa: E402
from graph_types import StockAgentState  # noqa: E402
from agents import init_llm  # noqa: E402
from data_fetcher import configure_fetcher  # noqa: E402
from real_time import get_realtime_quotes, get_index_quotes, validate_tickers, get_top_active  # noqa: E402
from stock_universe import search_stocks, refresh_universe, get_universe_stats, get_by_board  # noqa: E402
from charts import build_kline_chart, build_return_distribution, TIMEFRAME_DAYS  # noqa: E402
from risk_metrics import calculate_metrics, metrics_summary  # noqa: E402
from strategy import build_strategy_chart, optimize_ma_pairs  # noqa: E402

refresh_universe(force=False)
st.set_page_config(page_title="量化投研系统", page_icon="📈", layout="wide")

st.markdown("""<style>
    html,body{font-family:"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif}
    .idx-card{background:#f5f6f9;border-radius:8px;padding:10px 14px;text-align:center}
    .metric-chip{background:#f0f2f5;border-radius:6px;padding:6px 10px;font-size:0.8rem;text-align:center}
    .metric-chip .v{font-weight:700;font-size:1rem}.metric-chip .l{color:#888;font-size:0.65rem}
    [data-testid="stSidebar"]{background:linear-gradient(180deg,#1a1a2e 0%,#16213e 100%)}
    [data-testid="stSidebar"] *{color:#e0e0e0!important}
    [data-testid="stSidebar"] button{background:#0f3460!important;color:#e0e0e0!important;border:1px solid #1a1a4e!important}
    /* 统一分析报告字体：LLM输出的##/###标题不再放大 */
    [data-testid="stExpander"] h1,[data-testid="stExpander"] h2,[data-testid="stExpander"] h3,
    .stMarkdown h1,.stMarkdown h2,.stMarkdown h3{font-size:1rem!important;font-weight:700!important}
    [data-testid="stExpander"] p,.stMarkdown p{font-size:0.9rem!important}
</style>""", unsafe_allow_html=True)

# 报告文本清洗：把 ### 标题转成加粗文本，统一字号
def _clean_report(text: str) -> str:
    if not text: return "*无数据*"
    import re
    # ## Title → **Title**
    text = re.sub(r'^#{2,4}\s+', '**', text, flags=re.MULTILINE)
    # 如果开头变成了 **，补上换行
    text = re.sub(r'\*\*([^*]+)$', r'**\n\1', text, flags=re.MULTILINE)
    return text.replace("\n", "\n\n")

# ============================================
# 登录
# ============================================
if "user" not in st.session_state: st.session_state["user"] = None
if st.session_state["user"] is None:
    st.title("🔐 量化投研系统")
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        u = st.text_input("用户名", placeholder="输入用户名")
        p = st.text_input("密码", type="password", placeholder="输入密码")
        if st.button("登录 / 注册", use_container_width=True):
            if u and p:
                if verify_user(u, p): st.session_state["user"] = u; st.rerun()
                else: st.error("密码错误")
            else: st.error("用户名和密码不能为空")
        st.caption("首次输入自动注册 · 每人独立数据")
    st.stop()

USER = st.session_state["user"]

# ---- 异步分析状态 ----
for k in ["batch_tickers","batch_i","batch_total","batch_results","batch_running"]:
    if k not in st.session_state: st.session_state[k] = [] if k.endswith("s") or k.endswith("l") else (0 if k.endswith("i") or k.endswith("l") else False)

# ---- 工具函数（必须在全局批量处理之前定义）----
@st.cache_data(ttl=15)
def _q(t): return get_realtime_quotes(list(t))
@st.cache_data(ttl=20)
def _idx(): return get_index_quotes()
@st.cache_data(ttl=20)
def _active(n): return get_top_active(n)
def _cc(v):
    try: x=float(str(v).replace("%","").replace("+",""));return "color:#e53935;font-weight:700" if x>0 else "color:#43a047;font-weight:700" if x<0 else ""
    except: return ""
def _init_rt():
    if "rt" not in st.session_state:
        init_llm(model=os.getenv("LLM_MODEL","deepseek-chat"),api_key=os.getenv("OPENAI_API_KEY",""),api_base=os.getenv("OPENAI_API_BASE","https://api.deepseek.com/v1"))
        configure_fetcher(3,2);st.session_state["rt"]=True
def _run_one(t):
    g=build_graph()
    s:StockAgentState={"ticker":t,"data_fetch_success":False,"error_message":"","raw_history_data":{},"technical_analysis":"","fundamental_analysis":"","final_report":""}
    try:return g.invoke(s)  # type: ignore[reportAttributeAccessIssue]
    except Exception as e:return{**s,"error_message":str(e)}

# ============================================
# 侧边栏
# ============================================
indices=_idx()
with st.sidebar:
    st.markdown(f"### 📈 {USER}")
    if st.button("🔒 退出",use_container_width=True):st.session_state["user"]=None;st.rerun()
    if indices:
        for idx in indices:
            c="#e53935" if idx["change"]>0 else "#43a047" if idx["change"]<0 else "#aaa"
            a="▲" if idx["change"]>0 else "▼" if idx["change"]<0 else "—"
            st.markdown(f"""<div style='background:#ffffff10;border-radius:8px;padding:8px 10px;margin:2px 0'>
                <small style='color:#999'>{idx['name']}</small><br>
                <b style='color:#fff'>{idx['price']:.2f}</b>
                <span style='color:{c};font-weight:600'> {a}{idx['change']:+.2f}({idx['change_pct']:+.2f}%)</span></div>""",unsafe_allow_html=True)
    st.divider()

    # 上次分析结果摘要
    if st.session_state.get("batch_results"):
        ok=sum(1 for r in st.session_state["batch_results"] if r.get("data_fetch_success"))
        st.success(f"✅ 上次分析: {ok}/{len(st.session_state['batch_results'])} 成功")

    page=st.radio("",["🏠 市场概览","📈 个股深度","📐 策略回测","🔍 全市场搜索","⭐ 持仓管理","🚀 批量分析","📋 历史报告","🛡️ 智能风控"],label_visibility="collapsed")
    st.divider()
    s=get_summary(USER);st.metric("我的报告",s["total"])
    if s["total"]>0:st.metric("成功率",f"{s['success']/s['total']*100:.0f}%")

def _bar():
    if indices:
        cols=st.columns(len(indices))
        for i,idx in enumerate(indices):
            with cols[i]:
                c="#e53935" if idx["change"]>0 else "#43a047" if idx["change"]<0 else "#888"
                a="▲" if idx["change"]>0 else "▼"
                st.markdown(f"""<div class='idx-card'><small style='color:#888'>{idx['name']}</small><br><b>{idx['price']:.2f}</b><br><span style='color:{c};font-weight:600'>{a}{idx['change']:+.2f}({idx['change_pct']:+.2f}%)</span></div>""",unsafe_allow_html=True)

# ============================================
# 市场概览
# ============================================
if page.startswith("🏠"):
    st.title("市场概览");st.caption(f"👤 {USER} | {time.strftime('%H:%M:%S')}")
    _bar();st.divider()
    ca,cb=st.columns([2,1])
    with ca:
        st.subheader("🔥 成交量活跃榜 Top20")
        active=_active(30)
        if active:
            rows=[{"代码":q["code"],"名称":q["name"],"现价":f"{q['price']:.2f}","涨跌%":f"{q['change_pct']:+.2f}%","量(手)":f"{q.get('volume',0)/100:,.0f}"} for q in active[:20]]
            st.dataframe(pd.DataFrame(rows).style.map(_cc,subset=["涨跌%"]),hide_index=True,height=680)
    with cb:
        st.subheader("📋 我的持仓")
        wc=get_watchlist(USER) or ["600519","000858","600036","300750","000001"]
        qs=_q(tuple(wc[:12]))
        for q in qs:
            if q.get("name"):
                ch=q.get("change",0);c="#e53935" if ch>0 else "#43a047" if ch<0 else "#888"
                st.markdown(f"""<div style='background:#f6f8fa;border-radius:6px;padding:6px 8px;margin:2px 0;display:flex;justify-content:space-between'>
                    <div><b>{q['code']}</b><br><small>{q['name']}</small></div>
                    <div style='text-align:right'><b>{q['price']:.2f}</b><br><small style='color:{c};font-weight:600'>{ch:+.2f}({q['change_pct']:+.2f}%)</small></div></div>""",unsafe_allow_html=True)
        sm=get_summary(USER)
        st.divider();st.metric("我的报告",sm["total"])
        if sm["total"]>0:st.metric("成功率",f"{sm['success']/sm['total']*100:.0f}%")
        if sm.get("ratings"):
            for r,cnt in sorted(sm["ratings"].items(),key=lambda x:-x[1]):st.caption(f"  {r}: {cnt}")

# ============================================
# 个股深度
# ============================================
elif page.startswith("📈"):
    st.title("个股深度分析")
    ticker=st.text_input("股票代码",value="600519",max_chars=6,placeholder="600519",key="detail_ticker")
    if ticker and ticker.isdigit() and len(ticker)==6:
        info_q=_q((ticker,));name=info_q[0]["name"] if info_q and info_q[0].get("name") else ""
        st.subheader(f"{ticker} {name}")
        tf=st.radio("时间",list(TIMEFRAME_DAYS.keys()),horizontal=True,index=3,key="tf")
        with st.spinner("加载K线..."):
            fig,df=build_kline_chart(ticker,name,tf)
        if fig:st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":True,"displaylogo":False})
        if df is not None and not df.empty:
            st.divider();st.subheader("📊 量化指标")
            m=calculate_metrics(df)
            if "error" not in m:
                cm=[("最新价",f"{m['current_price']:.2f}"),("年化收益",f"{m['annual_return']:.2f}%"),("年化波动",f"{m['annual_volatility']:.2f}%"),("夏普",f"{m['sharpe_ratio']:.2f}"),("最大回撤",f"{m['max_drawdown']:.2f}%"),("盈亏比",f"{m['profit_factor']:.2f}"),("胜率",f"{m['win_rate']:.1f}%"),("VaR95",f"{m['var_95']:.2f}%"),("52周高",f"{m['price_52w_high']:.2f}"),("52周低",f"{m['price_52w_low']:.2f}"),("距高点",f"{m['pct_from_high']:.2f}%"),("趋势",m['trend'])]
                for i in range(0,len(cm),6):
                    cols=st.columns(6)
                    for j,(l,v) in enumerate(cm[i:i+6]):
                        with cols[j]:st.markdown(f"<div class='metric-chip'><div class='v'>{v}</div><div class='l'>{l}</div></div>",unsafe_allow_html=True)
        st.divider();st.subheader("🤖 AI 投研")
        if st.button("▶️ 生成报告",type="primary"):
            _init_rt()
            with st.spinner("分析中..."):
                r=_run_one(ticker);save_result(USER,r)
            if r.get("data_fetch_success"):
                t1,t2,t3=st.tabs(["技术面","基本面","CIO"])
                with t1:st.markdown(_clean_report(r.get("technical_analysis") or ""))
                with t2:st.markdown(_clean_report(r.get("fundamental_analysis") or ""))
                with t3:
                    rt=r.get("rating")or extract_rating(r.get("final_report",""))
                    if rt:st.markdown(f"### {rt}")
                    st.divider();st.markdown(_clean_report(r.get("final_report") or ""))
            else:st.error(f"失败: {r.get('error_message','')[:200]}")

# ============================================
# 策略回测
# ============================================
elif page.startswith("📐"):
    st.title("📐 双均线策略引擎")
    mode=st.radio("模式",["单策略 (25日×500日)","参数优化 (扫描400组)"],horizontal=True)
    ticker=st.text_input("股票代码",value="600519",max_chars=6,key="stk")
    if ticker and ticker.isdigit() and len(ticker)==6:
        info_q=_q((ticker,));name=info_q[0]["name"] if info_q and info_q[0].get("name") else ""
        if "单策略" in mode:
            if st.button("🔍 运行回测",type="primary"):
                with st.spinner("加载数据..."):
                    sfig,sr=build_strategy_chart(ticker,name)
                if sfig:
                    st.plotly_chart(sfig,use_container_width=True,config={"displaylogo":False})
                    c1,c2,c3,c4,c5=st.columns(5)
                    c1.metric("策略收益",f"{sr['total_return']:.2f}%")
                    c2.metric("买入持有",f"{sr['bh_return']:.2f}%")
                    c3.metric("最大回撤",f"{sr['max_drawdown']:.2f}%")
                    c4.metric("胜率",f"{sr['win_rate']:.1f}%")
                    c5.metric("交易次数",sr['total_trades'])
                    st.info(f"**{sr['current_status']}**")
        else:
            st.caption("20×20=400 组MA组合，按夏普排名")
            if st.button("🚀 参数优化",type="primary"):
                with st.spinner("扫描中(~30秒)..."):
                    ofig,odf=optimize_ma_pairs(ticker,name)
                if ofig is not None:
                    st.plotly_chart(ofig,use_container_width=True,config={"displaylogo":False})
                    top=odf.head(50)[["short","long","total_return","bh_return","excess","sharpe","max_drawdown","win_rate","total_trades"]]
                    top.columns=["短线","长线","策略%","持有%","超额%","夏普","回撤%","胜率%","交易"]
                    st.dataframe(top,hide_index=True,height=400)
                    best=odf.iloc[0];st.success(f"🏆 MA{int(best['short'])}×MA{int(best['long'])} 夏普:{best['sharpe']:.2f} 超额:{best['excess']:.2f}%")

# ============================================
# 全市场搜索 / 持仓管理 / 历史报告（精简但完整）
# ============================================
elif page.startswith("🔍"):
    st.title("全市场搜索");univ=get_universe_stats();st.caption(f"A股 {univ['total']:,} 支")
    qry=st.text_input("搜索",placeholder="茅台/宁德/600519")
    bf=st.selectbox("板块",["全部"]+sorted(univ.get("boards",{}).keys()))
    if qry:
        results=search_stocks(qry,60)
        if bf!="全部":results=[r for r in results if r.get("board")==bf]
        if results:
            for i in range(0,len(results),5):
                cols=st.columns(5)
                for j,r in enumerate(results[i:i+5]):
                    with cols[j]:
                        st.markdown(f"<div style='background:#f6f8fa;border-radius:8px;padding:10px;text-align:center'><b>{r['code']}</b><br>{r['name']}<br><small>{r['market']}·{r['board']}</small></div>",unsafe_allow_html=True)
                        if st.button("➕",key=f"s_{r['code']}"):add_to_watchlist(USER,"默认池",[r["code"]]);st.toast(f"已加入 {r['code']}")
    else:
        tabs=st.tabs(list(univ.get("boards",{}).keys()))
        for tab,board in zip(tabs,univ.get("boards",{}).keys()):
            with tab:
                for i in range(0,min(len(get_by_board(board,24)),24),6):
                    cols=st.columns(6)
                    for j,s in enumerate(get_by_board(board,24)[i:i+6]):
                        with cols[j]:st.code(s["code"],language=None);st.caption(s["name"])
                        if st.button("➕",key=f"b_{s['code']}"):add_to_watchlist(USER,"默认池",[s["code"]]);st.toast(f"已加入 {s['code']}")

elif page.startswith("⭐"):
    st.title("持仓管理")
    pns=get_watchlist_names(USER) or ["默认池"];al=st.selectbox("池",pns)
    codes=get_watchlist(USER,al)
    ca,cm=st.columns([3,1])
    with ca:
        if codes:
            qs=_q(tuple(codes));valid=[q for q in qs if q.get("price",0)>0]
            st.caption(f"持仓 **{len(codes)}** 支 | 有效 **{len(valid)}**")
            if valid:
                rows=[{"代码":q["code"],"名称":q["name"],"现价":f"{q['price']:.2f}","涨跌%":f"{q['change_pct']:+.2f}%","量(手)":f"{q.get('volume',0)/100:,.0f}"} for q in valid]
                st.dataframe(pd.DataFrame(rows).style.map(_cc,subset=["涨跌%"]),hide_index=True,height=400)
        else:st.info("空。去「全市场搜索」添加。")
    with cm:
        batch=st.text_area("批量导入",placeholder="600519,000858")
        if st.button("📥 导入",use_container_width=True) and batch:
            new=[c.strip() for c in batch.replace(" ",",").split(",") if c.strip()]
            v,iv=validate_tickers(new)
            if v:add_to_watchlist(USER,al,v);st.toast(f"+{len(v)}支");st.rerun()
            if iv:st.error(f"无效: {', '.join(iv)}")
        if codes:
            rm=st.multiselect("移除",codes)
            if st.button("🗑️ 移除",use_container_width=True,disabled=not rm):remove_from_watchlist(USER,al,rm);st.rerun()

# ============================================
# 🚀 批量分析（一次执行，无闪烁，实时进度）
# ============================================
elif page.startswith("🚀"):
    st.title("批量量化分析")
    _init_rt()

    # ---- 配置 ----
    col_in, col_cfg = st.columns([2, 1])
    with col_in:
        src = st.radio("标的来源", ["我的持仓池", "实时活跃榜 Top20", "手动输入"], horizontal=True)
        if "持仓" in src:
            pn = get_watchlist_names(USER) or ["默认池"]; sp = st.selectbox("池", pn)
            tickers = get_watchlist(USER, sp)
            if tickers: v, iv = validate_tickers(tickers); tickers = v
            st.success(f"有效: **{len(tickers)}** 支" if tickers else "池为空")
        elif "活跃" in src:
            tickers = [q["code"] for q in _active(20)]
            st.info(f"{len(tickers)} 支")
        else:
            raw = st.text_area("代码（每行一个）", "600519\n000858\n300750\n000001")
            tickers = [t.strip() for t in raw.split("\n") if t.strip()]
            v, iv = validate_tickers(tickers); tickers = v
    with col_cfg:
        st.caption(f"待分析: **{len(tickers) if tickers else 0}** 支")
        st.caption("每支约 15-30 秒")
        st.caption("分析期间请勿切换页面")

    st.divider()

    if st.button("▶️ 开始批量分析", type="primary", use_container_width=True, disabled=not bool(tickers)):
        total = len(tickers)
        results = []
        pbar = st.progress(0, text="准备中...")
        status_area = st.empty()

        success_count = 0
        for idx, t in enumerate(tickers):
            # 更新进度条（不触发完整页面刷新）
            pbar.progress((idx) / total, text=f"正在分析: {t} ({idx+1}/{total})")

            # 执行分析
            final = _run_one(t)
            save_result(USER, final)
            results.append(final)

            if final.get("data_fetch_success"):
                success_count += 1

            # 实时显示刚完成的结果
            icon = "✅" if final.get("data_fetch_success") else "❌"
            err = final.get("error_message", "")[:50]
            status_area.markdown(
                f"{icon} **{t}** "
                + (f"| {final.get('rating','')}" if final.get("data_fetch_success") else f"| {err}")
                + f"<br><small>{time.strftime('%H:%M:%S')}</small>",
                unsafe_allow_html=True,
            )

        pbar.progress(1.0, text=f"完成! {success_count}/{total} 成功")

        # 汇总
        st.divider()
        c1, c2, c3 = st.columns(3)
        c1.metric("成功", success_count)
        c2.metric("失败", total - success_count)
        c3.metric("合计", total)

        # 失败详情
        failed = [r for r in results if not r.get("data_fetch_success")]
        if failed:
            with st.expander(f"❌ {len(failed)} 支失败详情"):
                for r in failed:
                    st.warning(f"**{r['ticker']}**: {r.get('error_message', '未知错误')[:150]}")

        # 成功结果一览
        ok_results = [r for r in results if r.get("data_fetch_success")]
        if ok_results:
            with st.expander(f"✅ {len(ok_results)} 支分析结果"):
                for r in ok_results:
                    st.markdown(f"**{r['ticker']}** | 评级: {r.get('rating', extract_rating(r.get('final_report','')) or '—')}")
                    st.caption((r.get('final_report', '') or '')[:150] + "...")

        st.success("已自动存入历史报告，前往 📋 历史报告 查看完整内容。")

        # 保存到 session 供侧边栏显示
        st.session_state["batch_results"] = results
        st.session_state["batch_running"] = False

# ============================================
# 📋 历史报告
# ============================================
elif page.startswith("📋"):
    st.title("历史报告")

    # ---- 数据库自诊断 ----
    from database import _get_db_dir, _db_path
    dd = _get_db_dir(); df = _db_path(USER); fe = os.path.exists(df)
    with st.expander("🔧 数据库诊断", expanded=False):
        st.caption(f"目录: {dd}")
        st.caption(f"文件: {df}")
        st.caption(f"存在: {'是' if fe else '否'} | 大小: {os.path.getsize(df) if fe else 0} bytes")
        if fe:
            try:
                import sqlite3
                c = sqlite3.connect(df)
                cnt = c.execute("SELECT COUNT(*) FROM analysis_results").fetchone()[0]
                st.caption(f"记录数: {cnt}")
                if cnt > 0:
                    for r in c.execute("SELECT ticker,fetch_success,rating,created_at FROM analysis_results ORDER BY created_at DESC LIMIT 5").fetchall():
                        st.caption(f"  {r[0]} | {'OK' if r[1] else 'FAIL'} | {r[2] or '-'} | {r[3]}")
                c.close()
            except Exception as e:
                st.error(f"读取异常: {e}")

    all_r = get_results(USER, limit=500)
    if not all_r:
        st.info("暂无数据。请先运行「批量量化分析」或「个股深度分析」生成报告。")
    else:
        df_all=pd.DataFrame(all_r)
        if "fetch_success" in df_all.columns:df_all["状态"]=df_all["fetch_success"].apply(lambda x:"✅" if x else "❌")
        if "created_at" in df_all.columns:df_all["时间"]=df_all["created_at"].str[:19]
        c1,c2,c3=st.columns(3)
        with c1:st_=st.multiselect("代码",sorted(df_all["ticker"].dropna().unique()))
        with c2:sr=st.multiselect("评级",sorted(df_all["rating"].dropna().unique()))
        with c3:kw=st.text_input("关键词","")
        df_f=df_all.copy()
        if st_:df_f=df_f[df_f["ticker"].isin(st_)]
        if sr:df_f=df_f[df_f["rating"].isin(sr)]
        if kw:df_f=df_f[df_f["final_report"].fillna("").str.contains(kw,case=False)]
        st.caption(f"{len(df_f)}/{len(df_all)} 条")
        st.dataframe(df_f[["时间","ticker","状态","rating","technical_analysis","fundamental_analysis","final_report"]].rename(columns={"ticker":"代码","rating":"评级","technical_analysis":"技术面","fundamental_analysis":"基本面","final_report":"CIO决策"}),hide_index=True,height=400)
        st.divider();st.subheader("单股详情")
        detail=st.selectbox("代码",sorted(df_all["ticker"].dropna().unique()))
        if detail:
            recs=[r for r in all_r if r["ticker"]==detail]
            if recs:
                r=recs[0];t1,t2,t3=st.tabs(["技术面","基本面","CIO"])
                with t1:st.markdown(_clean_report(r.get("technical_analysis") or ""))
                with t2:st.markdown(_clean_report(r.get("fundamental_analysis") or ""))
                with t3:
                    rt=r.get("rating")or extract_rating(r.get("final_report",""))
                    if rt:st.markdown(f"### {rt}")
                    st.divider();st.markdown(_clean_report(r.get("final_report") or ""))

# ============================================
# 🛡️ 智能风控（新增）
# ============================================
else:
    st.title("🛡️ 智能风控决策辅助")
    st.caption("输入持仓信息，AI 量化分析师给出仓位管理、止损止盈、风险评估")

    col_a, col_b = st.columns(2)
    with col_a:
        ticker = st.text_input("股票代码", value="600519", max_chars=6, key="risk_ticker")
        entry_price = st.number_input("买入成本价", value=1300.0, step=0.01, format="%.2f")
        shares = st.number_input("持股数量（股）", value=100, step=100)
        risk_tolerance = st.select_slider("风险承受", ["保守","稳健","积极"], value="稳健")

    with col_b:
        if ticker and ticker.isdigit() and len(ticker) == 6:
            info_q = _q((ticker,))
            name = info_q[0]["name"] if info_q and info_q[0].get("name") else ""
            if name:
                # 获取当前价
                cur_price = info_q[0]["price"] if info_q[0].get("price", 0) > 0 else entry_price
                pnl = (cur_price - entry_price) * shares
                pnl_pct = (cur_price / entry_price - 1) * 100
                cost = entry_price * shares

                s_color = "#e53935" if pnl >= 0 else "#43a047"
                st.markdown(f"""
                <div style='background:#f6f8fa;border-radius:8px;padding:16px;margin:8px 0'>
                    <b>{ticker} {name}</b><br>
                    现价: <b>{cur_price:.2f}</b> | 成本: {entry_price:.2f}<br>
                    浮动盈亏: <b style='color:{s_color};font-size:1.2rem'>¥{pnl:+,.0f} ({pnl_pct:+.2f}%)</b><br>
                    持仓市值: ¥{cur_price*shares:,.0f}
                </div>""", unsafe_allow_html=True)

    st.divider()

    if st.button("🛡️ 生成风控报告", type="primary", use_container_width=True, disabled=not (ticker and ticker.isdigit())):
        _init_rt()
        with st.spinner("AI 风控分析师工作中..."):

            # 获取量化指标
            from charts import _fetch_history as _fh
            chart_df, _ = build_kline_chart(ticker, name, "1 年")
            risk_text = ""
            if chart_df is not None:
                # Re-fetch for metrics since chart_df might not have enough data
                pass
            try:
                # Direct fetch for risk metrics
                import datetime
                import requests as _r
                _o = _r.Session.__init__
                def _p(s, *a, **k): _o(s, *a, **k); s.trust_env = False
                _r.Session.__init__ = _p
                import akshare as ak
                prefix = "sh" if ticker.startswith(("60", "68")) else "sz"
                end = datetime.datetime.now().strftime("%Y%m%d")
                start = (datetime.datetime.now() - datetime.timedelta(days=800)).strftime("%Y%m%d")
                hist_df = ak.stock_zh_a_hist_tx(symbol=prefix + ticker, start_date=start, end_date=end, adjust="")
                if not hist_df.empty:
                    m = calculate_metrics(hist_df)
                    if "error" not in m:
                        risk_text = metrics_summary(m)
            except Exception:
                risk_text = "（量化指标暂时获取失败，基于行情数据推理）"

            cur_price = info_q[0]["price"] if info_q and info_q[0].get("price", 0) > 0 else entry_price

            prompt = f"""你是一位拥有20年经验的华尔街量化风控总监。请对以下持仓进行专业风险评估并给出操作建议。

【持仓信息】
- 股票: {ticker} {name}
- 成本价: {entry_price:.2f} 元
- 当前价: {cur_price:.2f} 元
- 持股数量: {shares} 股
- 持仓市值: ¥{cur_price*shares:,.0f}
- 浮动盈亏: {(cur_price/entry_price-1)*100:+.2f}%
- 风险偏好: {risk_tolerance}

{risk_text}

请输出（200字以内，极其理性、数据驱动、不带感情色彩）：

1. **风险等级**：用 [高风险/中风险/低风险] 标签
2. **止损位**：基于波动率计算的具体价格
3. **仓位建议**：建议持有/减仓/加仓及具体比例
4. **核心风险点**：当前最大的1-2个风险因素
5. **操作优先级**：接下来最该做的一件事"""

            llm = init_llm(
                model=os.getenv("LLM_MODEL", "deepseek-chat"),
                api_key=os.getenv("OPENAI_API_KEY", ""),
                api_base=os.getenv("OPENAI_API_BASE", "https://api.deepseek.com/v1"),
            )
            from langchain_core.messages import HumanMessage
            response = llm.invoke([HumanMessage(content=prompt)])
            st.markdown(response.content.replace("\n", "\n\n"))

            # 计算建议止损位（基于ATR/波动率）
            if "m" in dir() and "error" not in m:
                vol = m.get("volatility_30d", 30) / 100
                atr_stop = cur_price * (1 - 2 * vol / (252 ** 0.5))
                st.divider()
                c1, c2, c3 = st.columns(3)
                c1.metric("建议止损价", f"{atr_stop:.2f}")
                c2.metric("距止损", f"{(cur_price-atr_stop)/cur_price*100:.2f}%")
                c3.metric("30日波动率", f"{m['volatility_30d']:.1f}%")
