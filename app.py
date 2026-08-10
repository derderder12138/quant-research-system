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
    save_note, get_note, get_all_notes,
    save_position, get_position, get_all_positions,
    save_alert, get_alerts, delete_alert,
    init_virtual_account, get_virtual_cash, get_virtual_holdings, get_virtual_transactions, virtual_trade,
)
from graph_builder import build_graph  # noqa: E402
from graph_types import StockAgentState  # noqa: E402
from agents import init_llm  # noqa: E402
from data_fetcher import configure_fetcher  # noqa: E402
from real_time import get_realtime_quotes, get_index_quotes, validate_tickers, get_top_active, get_quotes_batched  # noqa: E402
from stock_universe import search_stocks, refresh_universe, get_universe_stats, get_by_board  # noqa: E402
from charts import build_kline_chart, build_return_distribution, TIMEFRAME_DAYS  # noqa: E402
from risk_metrics import calculate_metrics, metrics_summary  # noqa: E402
from strategy import build_strategy_chart, optimize_ma_pairs  # noqa: E402
from strategy_custom import build_custom_chart  # noqa: E402
from fundamental_data import get_single_fundamentals  # noqa: E402
from industry import classify_batch, get_industry_list  # noqa: E402
from signals import get_all_signals  # noqa: E402

try:
    refresh_universe(force=False)
except Exception:
    pass  # 云端网络受限时降级，依赖已有缓存或跳过
st.set_page_config(page_title="量化投研系统", page_icon="📈", layout="wide")

st.markdown("""<style>
    section[data-testid="stSidebar"] { background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%) !important; }
    section[data-testid="stSidebar"] * { color: #e0e0e0 !important; }
    section[data-testid="stSidebar"] button { background: #0f3460 !important; border: 1px solid #1a1a4e !important; }
    .idx-card { background: #f6f8fa; border-radius: 8px; padding: 10px 14px; text-align: center; }
    .metric-chip { background: #f0f2f5; border-radius: 6px; padding: 6px 10px; font-size: .8rem; text-align: center; }
    .metric-chip .v { font-weight: 700; font-size: 1rem; }
    .metric-chip .l { font-size: .65rem; color: #888; }
    [data-testid="stExpander"] h1, [data-testid="stExpander"] h2, [data-testid="stExpander"] h3 { font-size: 1rem !important; font-weight: 700 !important; }
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
    except Exception: return ""
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

    page=st.radio("",["🏠 市场概览","📈 个股深度","📐 策略回测","🔍 全市场搜索","🎯 条件选股","⭐ 持仓管理","💸 模拟交易","🚀 批量分析","📋 历史报告"],label_visibility="collapsed")
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
    _bar()

    # 市场广度
    col_br1,col_br2,col_br3,col_br4=st.columns(4)
    try:
        from real_time import get_realtime_quotes
        sample=['600519','000001','300750','000858','600036','601318','000725','002415','601012','600900',
                '601398','600276','688981','002594','000002','601668','002230','002371','300059','600030']
        qs=get_realtime_quotes(sample)
        up=sum(1 for q in qs if q.get('change',0)>0)
        down=sum(1 for q in qs if q.get('change',0)<0)
        total=len([q for q in qs if q.get('price',0)>0])
        col_br1.metric("📈 上涨家数",up)
        col_br2.metric("📉 下跌家数",down)
        col_br3.metric("📊 样本数",total)
        col_br4.metric("🔥 涨跌比",f"{up}:{down}" if total else "-")
    except Exception:
        pass

    # 板块轮动热力
    st.divider()
    st.subheader("🔥 板块轮动")
    univ = get_universe_stats()
    board_names = list(univ.get("boards", {}).keys())
    board_changes = {}
    with st.spinner("计算板块涨跌..."):
        for b in board_names:
            sample = get_by_board(b, limit=80)
            if sample:
                codes = [s["code"] for s in sample[:60]]
                try:
                    qs = get_quotes_batched(codes)
                    if qs:
                        avg_chg = sum(q["change_pct"] for q in qs.values() if q.get("change_pct", 0) != 0) / max(len(qs), 1)
                        board_changes[b] = round(avg_chg, 2)
                except Exception:
                    pass

    if board_changes:
        bc = sorted(board_changes.items(), key=lambda x: x[1], reverse=True)
        bcols = st.columns(len(bc))
        for i, (bname, bchg) in enumerate(bc):
            with bcols[i]:
                color = "#e53935" if bchg > 0 else "#43a047" if bchg < 0 else "#888"
                arrow = "🔥" if bchg > 0.5 else "▲" if bchg > 0 else "▼" if bchg < 0 else "—"
                st.markdown(f"""<div style='background:#f6f8fa;border-radius:8px;padding:10px 6px;text-align:center'>
                    <small style='color:#888'>{bname}</small><br>
                    <b style='color:{color};font-size:1.1rem'>{arrow} {bchg:+.2f}%</b></div>""", unsafe_allow_html=True)

    # 触发预警提醒
    alerts=get_alerts(USER)
    if alerts:
        triggered=[]
        for a in alerts:
            try:
                qq=_q((a["ticker"],))
                if qq and qq[0].get("price",0)>0:
                    cur=qq[0]["price"]
                    if (a["direction"]=="above" and cur>=a["price"]) or (a["direction"]=="below" and cur<=a["price"]):
                        triggered.append(f"{a['ticker']} {'涨破' if a['direction']=='above' else '跌破'} {a['price']:.2f} (现价{cur:.2f})")
            except Exception: pass
        if triggered:
            st.warning("🔔 触发预警: " + " | ".join(triggered))

    # 持仓盈亏总览
    positions=get_all_positions(USER)
    if positions:
        wl=get_watchlist(USER) or list(positions.keys())
        if wl:
            try:
                qs=_q(tuple(wl[:15]))
                pos_total_cost=0; pos_total_value=0
                for q in qs:
                    if q.get("code") in positions and q.get("price",0)>0:
                        p=positions[q["code"]]
                        pos_total_cost+=p["cost"]*p["shares"]
                        pos_total_value+=q["price"]*p["shares"]
                if pos_total_cost>0:
                    pos_pnl=pos_total_value-pos_total_cost
                    pos_pnl_pct=(pos_total_value/pos_total_cost-1)*100
                    pc1,pc2,pc3=st.columns(3)
                    pc1.metric("持仓总成本",f"{pos_total_cost:,.0f}元")
                    pc2.metric("持仓总市值",f"{pos_total_value:,.0f}元")
                    pc3.metric("持仓总盈亏",f"{pos_pnl:+,.0f}元 ({pos_pnl_pct:+.2f}%)")
            except Exception: pass

    st.divider()
    ca,cb=st.columns([2,1])
    with ca:
        st.subheader("🔥 成交量活跃榜 Top20")
        active=_active(30)
        if active:
            rows=[{"代码":str(q["code"]).zfill(6),"名称":q["name"],"现价":f"{q['price']:.2f}","涨跌%":f"{q['change_pct']:+.2f}%","量(手)":f"{q.get('volume',0)/100:,.0f}"} for q in active[:20]]
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

        # 个人笔记
        existing_note = get_note(USER, ticker)
        note = st.text_area("📝 我的笔记", value=existing_note, placeholder="记录看好理由、关键价位、风险提示...",
                            height=68, key=f"note_{ticker}")
        if note != existing_note:
            save_note(USER, ticker, note)
            st.toast("笔记已保存")

        # 基本面数据
        with st.spinner("加载基本面..."):
            fd = get_single_fundamentals(ticker)
        if fd and fd.get("pe", 0) > 0:
            st.divider(); st.subheader("📋 基本面速览")
            fc1, fc2, fc3, fc4, fc5, fc6 = st.columns(6)
            fc1.metric("市盈率(动)", f"{fd['pe']:.2f}")
            fc2.metric("总市值(亿)", f"{fd['market_cap']:,.0f}")
            fc3.metric("流通市值(亿)", f"{fd['circ_market_cap']:,.0f}")
            fc4.metric("换手率", f"{fd['turnover_rate']:.2f}%")
            fc5.metric("52周最高", f"{fd['high_52w']:.2f}")
            fc6.metric("52周最低", f"{fd['low_52w']:.2f}")
            fc7, fc8, fc9, fc10 = st.columns(4)
            fc7.metric("近一年涨幅", f"{fd['y1_change']:+.2f}%")
            fc8.metric("近半年涨幅", f"{fd['hy_change']:+.2f}%")
            fc9.metric("年初至今", f"{fd['ytd_change']:+.2f}%")
            fc10.metric("ROE", f"{fd['roe']:.2f}%")

        if df is not None and not df.empty:
            st.divider();st.subheader("📊 量化指标")
            m=calculate_metrics(df)
            if "error" not in m:
                cm=[("最新价",f"{m['current_price']:.2f}"),("年化收益",f"{m['annual_return']:.2f}%"),("年化波动",f"{m['annual_volatility']:.2f}%"),("夏普",f"{m['sharpe_ratio']:.2f}"),("最大回撤",f"{m['max_drawdown']:.2f}%"),("盈亏比",f"{m['profit_factor']:.2f}"),("胜率",f"{m['win_rate']:.1f}%"),("VaR95",f"{m['var_95']:.2f}%"),("52周高",f"{m['price_52w_high']:.2f}"),("52周低",f"{m['price_52w_low']:.2f}"),("距高点",f"{m['pct_from_high']:.2f}%"),("趋势",m['trend'])]
                for i in range(0,len(cm),6):
                    cols=st.columns(6)
                    for j,(l,v) in enumerate(cm[i:i+6]):
                        with cols[j]:st.markdown(f"<div class='metric-chip'><div class='v'>{v}</div><div class='l'>{l}</div></div>",unsafe_allow_html=True)
        # 交易信号
        st.divider();st.subheader("📡 交易信号")
        if st.button("🔍 分析交易信号", key="sig_btn"):
            with st.spinner("计算凯里公式 + 多条件信号..."):
                sigs=get_all_signals(ticker)
            if 'error' not in sigs:
                # 凯里公式
                k=sigs['kelly']
                st.markdown(f"**🎯 凯里公式仓位建议**")
                ck1,ck2,ck3=st.columns(3)
                ck1.metric("全凯里",f"{k.get('kelly_full',0):.1f}%")
                ck2.metric("半凯里(保守)",f"{k.get('kelly_half',0):.1f}%")
                ck3.metric("盈亏比",f"{k.get('b_ratio',0):.2f}")
                st.caption(k.get('interpretation',''))

                st.divider()
                # 各信号行
                signals_list=[
                    ("MA10黏着",sigs['ma10_sticky']),
                    ("MA25止损",sigs['ma25_stop']),
                    ("历史高点(前复权)",sigs['history_high']),
                    ("50%回撤位",sigs['retrace_50']),
                    ("突破回踩",sigs['double_pullback']),
                    ("布林缩口",sigs['bollinger_squeeze']),
                    ("量价关系",sigs['volume_divergence']),
                    ("跳空缺口",sigs['gap_break']),
                ]
                for sname,sd in signals_list:
                    sig_text=sd.get('signal','')
                    if '🟢' in sig_text: bg='#e8f5e9';border='#4caf50'
                    elif '🔴' in sig_text: bg='#fce4ec';border='#e53935'
                    elif '🟡' in sig_text: bg='#fff8e1';border='#ff9800'
                    else: bg='#f5f5f5';border='#ccc'
                    st.markdown(f"""<div style='background:{bg};border-left:3px solid {border};padding:8px 12px;margin:3px 0;border-radius:4px'>
                        <b>{sname}</b>: {sig_text}</div>""",unsafe_allow_html=True)

                st.divider()
                st.info(f"**综合判断**: {sigs['action']}")
            else:
                st.error(sigs['error'])

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
    mode=st.radio("模式",["自定义策略 (5×25金叉+MACD+周线)","单策略 (25日×500日)","参数优化 (扫描400组)"],horizontal=True)
    ticker=st.text_input("股票代码",value="600519",max_chars=6,key="stk")
    if ticker and ticker.isdigit() and len(ticker)==6:
        info_q=_q((ticker,));name=info_q[0]["name"] if info_q and info_q[0].get("name") else ""
        if "自定义策略" in mode:
            st.caption("买入：MA5金叉MA25 + MACD日/周/月黄线>0 + 价格>25周线")
            st.caption("卖出：跌穿MA10减半仓 | 跌穿MA25清仓")
            if st.button("🔍 运行自定义策略", type="primary", key="run_custom"):
                with st.spinner("计算日/周/月线 MACD..."):
                    cfig, cr = build_custom_chart(ticker, name)
                if cfig and cr:
                    st.plotly_chart(cfig, use_container_width=True, config={"displaylogo": False})

                    # 绩效
                    c1, c2, c3, c4, c5 = st.columns(5)
                    c1.metric("策略收益", f"{cr['total_return']:.2f}%")
                    c2.metric("买入持有", f"{cr['bh_return']:.2f}%")
                    c3.metric("最大回撤", f"{cr['max_drawdown']:.2f}%")
                    c4.metric("胜率", f"{cr['win_rate']:.1f}%")
                    c5.metric("买入次数", cr['buy_count'])

                    # 当前条件状态
                    st.divider()
                    st.subheader("📋 当前买入条件状态")
                    cond_cols = st.columns(5)
                    cond_labels = {
                        "MA5金叉MA25": "MA5↑MA25",
                        "MACD日线黄线>0": "MACD日>0",
                        "MACD周线黄线>0": "MACD周>0",
                        "MACD月线黄线>0": "MACD月>0",
                        "价格>25周线": "价>25周线",
                    }
                    for i, (key, met) in enumerate(cr["conditions"].items()):
                        with cond_cols[i]:
                            icon = "✅" if met else "❌"
                            st.metric(cond_labels.get(key, key), icon)

                    if cr["all_met"]:
                        st.success(f"🟢 全部买入条件满足！当前仓位: {'满仓' if cr['position']==1 else '半仓' if cr['position']==0.5 else '空仓'}")
                    else:
                        unmet = [k for k, v in cr["conditions"].items() if not v]
                        st.warning(f"🔴 {len(unmet)} 项未满足: {', '.join(unmet)} | 当前仓位: {'满仓' if cr['position']==1 else '半仓' if cr['position']==0.5 else '空仓'}")

                    # 价格位置
                    cm1, cm2, cm3, cm4 = st.columns(4)
                    cm1.metric("最新价", f"{cr['latest_price']:.2f}")
                    cm2.metric("MA10", f"{cr['latest_ma10']:.2f}")
                    cm3.metric("MA25", f"{cr['latest_ma25']:.2f}")
                    cm4.metric("MA125(25周)", f"{cr['latest_ma125']:.2f}")
                else:
                    st.warning("数据不足，该股票历史数据不够（需要 800+ 交易日）。")

        elif "单策略" in mode:
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
                # 缓存：同股票 30 分钟内不重复计算
                @st.cache_data(ttl=1800, show_spinner=False)
                def _cached_optimize(t):
                    return optimize_ma_pairs(t, "")
                with st.spinner("扫描中(~30秒)…"):
                    ofig, odf = _cached_optimize(ticker)
                if ofig is not None:
                    st.plotly_chart(ofig,use_container_width=True,config={"displaylogo":False})
                    top=odf.head(50)[["short","long","total_return","bh_return","excess","sharpe","max_drawdown","win_rate","total_trades"]]
                    top.columns=["短线","长线","策略%","持有%","超额%","夏普","回撤%","胜率%","交易"]
                    st.dataframe(top,hide_index=True,height=400)
                    best=odf.iloc[0];st.success(f"🏆 MA{int(best['short'])}×MA{int(best['long'])} 夏普:{best['sharpe']:.2f} 超额:{best['excess']:.2f}%")

# ============================================
# ============================================
# 🎯 条件选股
# ============================================
elif page.startswith("🎯"):
    st.title("🎯 智能选股")
    st.caption("多维度筛选 + 综合评分，找到最优潜力股")

    from fundamental_data import get_fundamentals
    univ = get_universe_stats()

    col1, col2, col3 = st.columns(3)
    with col1:
        pe_max = st.number_input("PE(动)上限", value=80, min_value=1, max_value=500, step=5, help="市盈率越低越便宜")
        trend_filter = st.selectbox("趋势", ["全部","上升(价>MA20)","下降(价<MA20)"], help="价格在20日均线上方=上升趋势")
    with col2:
        cap_min = st.number_input("市值下限(亿)", value=50, min_value=0, max_value=10000, step=10)
        ind_filter = st.selectbox("行业", ["全部"] + get_industry_list(), help="聚焦特定行业赛道")
    with col3:
        board_f = st.selectbox("板块", ["全部"] + sorted(univ.get("boards", {}).keys()))
        chg_dir = st.selectbox("今日涨跌", ["全部","上涨","下跌"], help="今日市场表现")

    st.divider()

    if st.button("🔍 智能筛选", type="primary", use_container_width=True):
        with st.spinner(f"扫描 {univ['total']:,} 支股票..."):
            # 取候选池
            if board_f == "全部":
                candidates = []
                for b in univ.get("boards", {}).keys():
                    candidates.extend(get_by_board(b, limit=300))
            else:
                candidates = get_by_board(board_f, limit=500)
            candidates = candidates[:500]

            codes = [c["code"] for c in candidates]
            fd_all = get_fundamentals(codes[:200])

            results = []
            for code in codes:
                fd = fd_all.get(code, {})
                if not fd or fd.get("pe", 0) <= 0:
                    continue
                pe = fd.get("pe", 99)
                cap = fd.get("market_cap", 0)
                chg = fd.get("change_pct", 0)
                roe = fd.get("roe", 0)
                y1 = fd.get("y1_change", 0)
                turnover = fd.get("turnover_rate", 0)

                # 筛选逻辑
                if pe > pe_max or cap < cap_min:
                    continue
                if chg_dir == "上涨" and chg <= 0:
                    continue
                if chg_dir == "下跌" and chg >= 0:
                    continue
                if ind_filter != "全部":
                    from industry import classify_stock
                    if classify_stock(code) != ind_filter:
                        continue
                if trend_filter == "上升(价>MA20)":
                    try:
                        from charts import _fetch_history
                        td = _fetch_history(code, 60)
                        if td.empty or len(td) < 25: continue
                        td["MA20"] = td["close"].rolling(20).mean()
                        if td["close"].iloc[-1] <= td["MA20"].iloc[-1]: continue
                    except: pass
                elif trend_filter == "下降(价<MA20)":
                    try:
                        from charts import _fetch_history
                        td = _fetch_history(code, 60)
                        if td.empty or len(td) < 25: continue
                        td["MA20"] = td["close"].rolling(20).mean()
                        if td["close"].iloc[-1] >= td["MA20"].iloc[-1]: continue
                    except: pass

                # 综合评分（0-100）
                score = 50
                if pe < 15: score += 15
                elif pe < 30: score += 8
                else: score -= 5
                if roe > 15: score += 12
                elif roe > 8: score += 5
                if y1 > 10: score += 10
                elif y1 > 0: score += 3
                else: score -= 8
                if chg > 0: score += 3
                if cap > 500: score += 5
                if turnover > 1: score += 3
                score = max(0, min(100, score))

                results.append({**fd, "code": code, "score": score})

            if results:
                results.sort(key=lambda x: -x["score"])
                st.success(f"筛选出 {len(results)} 支")

                # 按评分分组显示
                gold = [r for r in results if r["score"] >= 75][:20]
                silver = [r for r in results if 50 <= r["score"] < 75][:30]
                bronze = [r for r in results if r["score"] < 50][:30]

                if gold:
                    st.subheader("🥇 优质标的（评分≥75）")
                    gdf = pd.DataFrame([{"代码":str(r["code"]).zfill(6),"名称":r.get("name",""),"评分":r["score"],"PE":f"{r['pe']:.1f}","市值":f"{r['market_cap']:,.0f}亿","ROE%":f"{r.get('roe',0):.1f}","近1年%":f"{r['y1_change']:+.1f}","今日%":f"{r['change_pct']:+.2f}"} for r in gold])
                    st.dataframe(gdf, hide_index=True, height=250)

                if silver:
                    st.subheader("🥈 良好标的（评分50-74）")
                    sdf = pd.DataFrame([{"代码":str(r["code"]).zfill(6),"名称":r.get("name",""),"评分":r["score"],"PE":f"{r['pe']:.1f}"} for r in silver])
                    st.dataframe(sdf, hide_index=True, height=200)

                # 批量操作
                sel = st.multiselect("选择加入持仓", [str(r["code"]).zfill(6) for r in results[:100]], key="scr2")
                if st.button(f"📥 加入 ({len(sel)}支)", disabled=not sel):
                    add_to_watchlist(USER, "默认池", sel)
                    st.toast(f"已添加 {len(sel)} 支")
                    st.rerun()
            else:
                st.info("无匹配。放宽条件试试。")

# ============================================
# 📊 多股对比
# ============================================
elif page.startswith("📊"):
    st.title("📊 多股对比")
    st.caption("选中多支股票，同时展示走势、基本面、技术指标")

    # 快速选择：持仓中的股票
    wl_codes = get_watchlist(USER) or ["600519", "000858", "300750"]
    selected = st.multiselect("选择对比股票（建议 2-4 支）", wl_codes,
                              default=wl_codes[:3] if len(wl_codes) >= 3 else wl_codes,
                              max_selections=6, key="compare_select")

    if selected:
        tabs = st.tabs(["📈 走势叠加", "📋 基本面对比", "📊 指标对比"])

        # Tab 1: 走势叠加
        with tabs[0]:
            tf = st.radio("时间", list(TIMEFRAME_DAYS.keys()), horizontal=True, index=3, key="comp_tf")
            from charts import _fetch_history
            import plotly.graph_objects as go

            fig_c = go.Figure()
            colors = ["#e83939", "#33a3ff", "#ffb340", "#9b30ff", "#1aad19", "#ff9800"]
            for idx, t in enumerate(selected):
                df_c = _fetch_history(t, TIMEFRAME_DAYS.get(tf, 200))
                if df_c.empty:
                    continue
                df_c["norm"] = df_c["close"] / df_c["close"].iloc[0] * 100
                info_q = _q((t,))
                nm = info_q[0]["name"] if info_q and info_q[0].get("name") else t
                fig_c.add_trace(go.Scatter(x=df_c["date"], y=df_c["norm"], mode="lines",
                    name=f"{t} {nm}", line=dict(color=colors[idx % len(colors)], width=2)))

            fig_c.add_hline(y=100, line_dash="dash", line_color="#666", line_width=0.5)
            fig_c.update_layout(template="plotly_dark", height=500, hovermode="x unified",
                                plot_bgcolor="#1a1a1a", paper_bgcolor="#1a1a1a",
                                margin=dict(l=10, r=10, t=30, b=10),
                                yaxis_title="归一化价格 (基期=100)",
                                legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0))
            st.plotly_chart(fig_c, use_container_width=True, config={"displaylogo": False})

        # Tab 2: 基本面对比
        with tabs[1]:
            from fundamental_data import get_fundamentals
            fd_all = get_fundamentals(selected)
            if fd_all:
                comp_rows = []
                for code, fd in fd_all.items():
                    comp_rows.append({
                        "代码": code, "名称": fd.get("name", ""),
                        "现价": f"{fd['price']:.2f}",
                        "涨跌%": f"{fd['change_pct']:+.2f}%",
                        "PE(动)": f"{fd['pe']:.2f}",
                        "市值(亿)": f"{fd['market_cap']:,.0f}",
                        "换手%": f"{fd['turnover_rate']:.2f}",
                        "ROE%": f"{fd.get('roe', 0):.2f}",
                        "近1年%": f"{fd['y1_change']:+.2f}%",
                        "年初%": f"{fd['ytd_change']:+.2f}%",
                    })
                st.dataframe(pd.DataFrame(comp_rows).style.map(_cc, subset=["涨跌%", "近1年%", "年初%"]),
                           hide_index=True, height=300)

        # Tab 3: 技术指标对比
        with tabs[2]:
            from risk_metrics import calculate_metrics
            ind_rows = []
            for t in selected:
                from charts import _fetch_history
                df_i = _fetch_history(t, 400)
                if not df_i.empty:
                    m = calculate_metrics(df_i)
                    if "error" not in m:
                        info_q = _q((t,))
                        nm = info_q[0]["name"] if info_q and info_q[0].get("name") else t
                        ind_rows.append({
                            "代码": t, "名称": nm,
                            "年化收益%": f"{m['annual_return']:.2f}",
                            "年化波动%": f"{m['annual_volatility']:.2f}",
                            "夏普比率": f"{m['sharpe_ratio']:.2f}",
                            "最大回撤%": f"{m['max_drawdown']:.2f}",
                            "日胜率%": f"{m['win_rate']:.1f}",
                            "趋势": m['trend'],
                        })
            if ind_rows:
                st.dataframe(pd.DataFrame(ind_rows), hide_index=True, height=250)
            else:
                st.info("数据不足")

# 全市场搜索 / 持仓管理 / 历史报告（精简但完整）
# ============================================
elif page.startswith("🔍"):
    st.title("全市场搜索")
    univ = get_universe_stats()
    st.caption(f"A股 {univ['total']:,} 支 | 支持代码/名称/拼音搜索 | 指数ETF可搜")

    # 实时指数速览
    if indices:
        idx_cols = st.columns(len(indices))
        for i, idx in enumerate(indices):
            with idx_cols[i]:
                c = "#e53935" if idx["change"] > 0 else "#43a047" if idx["change"] < 0 else "#888"
                a = "▲" if idx["change"] > 0 else "▼"
                st.metric(idx["name"], f"{idx['price']:.2f}", f"{a}{idx['change']:+.2f}({idx['change_pct']:+.2f}%)")

    col_q, col_f1, col_f2, col_n = st.columns([2.5, 1.2, 1.2, 1])
    with col_q:
        qry = st.text_input("搜索", placeholder="茅台 / 宁德 / 600519 / 银行")
    with col_f1:
        bf = st.selectbox("板块", ["全部"] + sorted(univ.get("boards", {}).keys()))
    with col_f2:
        industries = ["全部"] + get_industry_list()
        ind_f = st.selectbox("行业", industries)
    with col_n:
        limit_n = st.selectbox("显示数量", [50, 100, 200, 500], index=1)

    if qry:
        results = search_stocks(qry, max(limit_n, 200))
        if bf != "全部":
            results = [r for r in results if r.get("board") == bf]
        if ind_f != "全部":
            codes_industry = classify_batch([r["code"] for r in results])
            results = [r for r in results if codes_industry.get(r["code"], "其他") == ind_f]
        results = results[:limit_n]

        if results:
            st.success(f"找到 {len(results)} 支匹配")

            # 获取全部搜索结果实时行情
            codes = [r["code"] for r in results]
            try:
                quotes = get_quotes_batched(codes)
            except Exception:
                quotes = {}

            # 构建表格行
            rows = []
            for r in results:
                q = quotes.get(r["code"], {})
                price = q.get("price", 0)
                chg = q.get("change_pct", 0)
                row = {
                    "代码": r["code"],
                    "名称": r["name"],
                    "行业": classify_batch([r["code"]]).get(r["code"], "其他"),
                    "板块": r["board"],
                    "市场": r["market"],
                }
                if price > 0:
                    row["现价"] = f"{price:.2f}"
                    row["涨跌%"] = f"{chg:+.2f}%"
                else:
                    row["现价"] = "—"
                    row["涨跌%"] = "—"
                rows.append(row)

            df = pd.DataFrame(rows)
            if "涨跌%" in df.columns:
                styled = df.style.map(_cc, subset=["涨跌%"])
                st.dataframe(styled, hide_index=True, height=min(600, 35*len(rows)+38),
                           column_config={"代码": st.column_config.TextColumn(width="small"),
                                         "名称": st.column_config.TextColumn(width="medium"),
                                         "板块": st.column_config.TextColumn(width="small"),
                                         "市场": st.column_config.TextColumn(width="small"),
                                         "现价": st.column_config.TextColumn(width="small"),
                                         "涨跌%": st.column_config.TextColumn(width="small")})
            else:
                st.dataframe(df, hide_index=True, height=min(600, 35*len(rows)+38),
                           column_config={"代码": st.column_config.TextColumn(width="small"),
                                         "名称": st.column_config.TextColumn(width="medium"),
                                         "板块": st.column_config.TextColumn(width="small"),
                                         "市场": st.column_config.TextColumn(width="small")})

            # 批量添加区域
            st.divider()
            col_sel, col_add = st.columns([3, 1])
            with col_sel:
                selected = st.multiselect("选择股票批量加入持仓", codes, key="bulk_add",
                                          placeholder="点击选择，可多选...")
            with col_add:
                if st.button(f"📥 批量加入 ({len(selected)}支)", use_container_width=True, disabled=not selected):
                    n = add_to_watchlist(USER, "默认池", selected)
                    st.toast(f"已添加 {n} 支")
                    st.rerun()
        else:
            st.warning(f"无匹配「{qry}」的股票。尝试其他关键词（如公司全称、拼音首字母）。")

    else:
        # 无搜索时按板块浏览——表格形式，每次显示更多
        st.divider()
        st.subheader("📂 按板块浏览")
        boards = list(univ.get("boards", {}).keys())
        tabs = st.tabs(boards)
        for tab, board in zip(tabs, boards):
            with tab:
                stocks = get_by_board(board, limit=200)
                if stocks:
                    # 获取全部板块股票实时行情
                    all_codes = [s["code"] for s in stocks]
                    try:
                        quotes = get_quotes_batched(all_codes)
                    except Exception:
                        quotes = {}

                    rows = []
                    for s in stocks:
                        q = quotes.get(s["code"], {})
                        price = q.get("price", 0)
                        chg = q.get("change_pct", 0)
                        row = {"代码": s["code"], "名称": s["name"]}
                        if price > 0:
                            row["现价"] = f"{price:.2f}"
                            row["涨跌%"] = f"{chg:+.2f}%"
                        else:
                            row["现价"] = "—"; row["涨跌%"] = "—"
                        rows.append(row)

                    df = pd.DataFrame(rows)
                    if "涨跌%" in df.columns:
                        st.dataframe(df.style.map(_cc, subset=["涨跌%"]), hide_index=True,
                                   height=min(500, 35*len(rows)+38),
                                   column_config={"代码": st.column_config.TextColumn(width="small"),
                                                 "名称": st.column_config.TextColumn(width="medium"),
                                                 "现价": st.column_config.TextColumn(width="small"),
                                                 "涨跌%": st.column_config.TextColumn(width="small")})
                    else:
                        st.dataframe(df, hide_index=True, height=min(500, 35*len(rows)+38))

                    # 批量操作
                    sel_codes = st.multiselect(f"选择 {board} 股票加入持仓", [s["code"] for s in stocks],
                                               key=f"sel_{board}", placeholder="可多选...")
                    if st.button(f"📥 加入持仓 ({len(sel_codes)}支)", key=f"add_{board}", disabled=not sel_codes):
                        add_to_watchlist(USER, "默认池", sel_codes)
                        st.toast(f"已添加 {len(sel_codes)} 支")
                        st.rerun()

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
                notes_map = get_all_notes(USER)
                pos_map = get_all_positions(USER)
                rows=[{"代码":str(q["code"]).zfill(6),"名称":q["name"],"现价":f"{q['price']:.2f}","成本":f"{pos_map.get(q['code'],{}).get('cost',0):.2f}" if pos_map.get(q['code'],{}).get('cost',0)>0 else "-","盈亏%":f"{(q['price']/pos_map[q['code']]['cost']-1)*100:+.2f}%" if pos_map.get(q['code'],{}).get('cost',0)>0 else "-","涨跌%":f"{q['change_pct']:+.2f}%","量(手)":f"{q.get('volume',0)/100:,.0f}","笔记":notes_map.get(q["code"],"")[:20]} for q in valid]
                total_pnl=sum((q['price']-pos_map.get(q['code'],{}).get('cost',0))*pos_map.get(q['code'],{}).get('shares',0) for q in valid if pos_map.get(q['code'],{}).get('cost',0)>0)
                st.caption(f"持仓 **{len(codes)}** 支 | 有效 **{len(valid)}** | 持仓浮动盈亏: {total_pnl:+,.0f}元" if total_pnl!=0 else f"持仓 **{len(codes)}** 支 | 有效 **{len(valid)}**")
                st.dataframe(pd.DataFrame(rows).style.map(_cc,subset=["涨跌%","盈亏%"]),hide_index=True,height=400,
                           column_config={"笔记": st.column_config.TextColumn(width="small"),"成本": st.column_config.TextColumn(width="small"),"盈亏%": st.column_config.TextColumn(width="small")})
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
# 💸 模拟交易
# ============================================
elif page.startswith("💸"):
    st.title("💸 模拟交易")
    init_virtual_account(USER, 100000)
    cash = get_virtual_cash(USER)
    holdings = get_virtual_holdings(USER)

    # 账户总览
    col_cash, col_value, col_total, col_pnl = st.columns(4)
    col_cash.metric("可用现金", f"{cash:,.0f}元")

    total_value = cash
    total_cost = 0.0
    if holdings:
        codes = [h["ticker"] for h in holdings]
        try:
            quotes = get_quotes_batched(codes)
            for h in holdings:
                q = quotes.get(h["ticker"], {})
                price = q.get("price", 0)
                if price > 0:
                    total_value += price * h["shares"]
                    total_cost += h["avg_cost"] * h["shares"]
        except Exception:
            pass

    col_value.metric("持仓市值", f"{total_value-cash:,.0f}元")
    col_total.metric("总资产", f"{total_value:,.0f}元")

    init_cash = 100000
    total_pnl = total_value - init_cash
    total_pnl_pct = (total_value / init_cash - 1) * 100
    col_pnl.metric("总盈亏", f"{total_pnl:+,.0f}元 ({total_pnl_pct:+.2f}%)")

    st.divider()

    # 买入/卖出
    col_buy, col_holdings = st.columns([1, 2])
    with col_buy:
        st.subheader("下单")
        ticker_in = st.text_input("股票代码", placeholder="600519", key="trade_t")
        shares_in = st.number_input("数量(股)", value=100, step=100, min_value=100, key="trade_s")

        if ticker_in and ticker_in.isdigit():
            qq = _q((ticker_in,))
            cur_price = qq[0]["price"] if qq and qq[0].get("price", 0) > 0 else 0
            if cur_price > 0:
                st.caption(f"现价: {cur_price:.2f} | 金额: {cur_price*shares_in:,.0f}元")
                col_btn1, col_btn2 = st.columns(2)
                with col_btn1:
                    if st.button("🟢 买入", use_container_width=True):
                        r = virtual_trade(USER, ticker_in, "buy", int(shares_in), cur_price)
                        if r["success"]: st.toast(f'买入 {ticker_in} {shares_in}股 @ {cur_price:.2f}')
                        else: st.error(r["error"])
                with col_btn2:
                    if st.button("🔴 卖出", use_container_width=True):
                        r = virtual_trade(USER, ticker_in, "sell", int(shares_in), cur_price)
                        if r["success"]: st.toast(f'卖出 {ticker_in} {shares_in}股 @ {cur_price:.2f}')
                        else: st.error(r["error"])

        if st.button("🔄 重置账户（回到10万初始资金）"):
            from database import _ensure_tables as _et
            conn = _et(USER)
            conn.execute("DELETE FROM virtual_portfolio")
            conn.execute("DELETE FROM virtual_transactions")
            conn.execute("UPDATE virtual_cash SET cash=100000 WHERE id=1")
            conn.execute("INSERT INTO virtual_transactions (ticker,action,shares,price,amount,cash_after) VALUES ('CASH','init',0,0,100000,100000)")
            conn.commit(); conn.close()
            st.rerun()

    with col_holdings:
        st.subheader("持仓明细")
        if holdings:
            codes_h = [h["ticker"] for h in holdings]
            try:
                quotes_h = get_quotes_batched(codes_h)
                hrows = []
                for h in holdings:
                    q = quotes_h.get(h["ticker"], {})
                    cp = q.get("price", 0)
                    pnl = (cp - h["avg_cost"]) * h["shares"] if cp > 0 else 0
                    pnl_pct = (cp / h["avg_cost"] - 1) * 100 if cp > 0 and h["avg_cost"] > 0 else 0
                    hrows.append({
                        "代码": h["ticker"], "名称": q.get("name", ""),
                        "持仓": h["shares"], "成本": f"{h['avg_cost']:.2f}",
                        "现价": f"{cp:.2f}" if cp > 0 else "-",
                        "盈亏": f"{pnl:+,.0f}元 ({pnl_pct:+.2f}%)",
                    })
                st.dataframe(pd.DataFrame(hrows), hide_index=True, height=250)
            except Exception:
                st.caption("行情加载中...")
        else:
            st.info("暂无持仓，开始模拟交易吧。")

    # 交易记录
    st.divider()
    st.subheader("交易记录")
    txs = get_virtual_transactions(USER, limit=30)
    if txs:
        tx_rows = [{
            "时间": t.get("created_at", "")[:19], "代码": t["ticker"],
            "操作": "🟢买入" if t["action"] == "buy" else ("🔴卖出" if t["action"] == "sell" else "💵初始"),
            "数量": t["shares"], "价格": f"{t['price']:.2f}",
            "金额": f"{t['amount']:,.0f}", "余额": f"{t['cash_after']:,.0f}",
        } for t in txs]
        st.dataframe(pd.DataFrame(tx_rows), hide_index=True, height=250)
    else:
        st.info("暂无交易。")

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

        # CSV 导出按钮
        export_df = df_f[["时间","ticker","状态","rating","technical_analysis","fundamental_analysis","final_report"]].rename(
            columns={"ticker":"代码","rating":"评级","technical_analysis":"技术面","fundamental_analysis":"基本面","final_report":"CIO决策"})
        csv = export_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button("📥 导出 CSV", csv, f"量化分析报告_{USER}.csv", "text/csv",
                          use_container_width=True)

        st.dataframe(export_df, hide_index=True, height=400)
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
