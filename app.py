"""
量化投研系统 — 多用户专业前端。
每人独立持仓 · 独立分析记录 · 同步批量分析。
启动: streamlit run app.py
"""

import os, sys, time, re
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor

import streamlit as st
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ---- 密钥 ----
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
from strategy import build_strategy_chart  # noqa: E402

refresh_universe(force=False)

st.set_page_config(page_title="量化投研系统", page_icon="📈", layout="wide")

# ---- 专业 CSS ----
st.markdown("""<style>
    html,body,[class*="css"]{font-family:"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif}
    .idx-card{background:#f5f6f9;border-radius:8px;padding:10px 14px;text-align:center;margin:2px}
    .idx-card .n{font-size:0.7rem;color:#888}.idx-card .p{font-size:1.15rem;font-weight:700}
    .metric-chip{background:#f0f2f5;border-radius:6px;padding:6px 10px;font-size:0.8rem;min-width:100px;text-align:center}
    .metric-chip .v{font-weight:700;font-size:1rem}.metric-chip .l{color:#888;font-size:0.65rem}
    .up{color:#e53935}.down{color:#43a047}
    [data-testid="stSidebar"]{background:linear-gradient(180deg,#1a1a2e 0%,#16213e 100%)}
    [data-testid="stSidebar"] *{color:#e0e0e0!important}
    [data-testid="stSidebar"] button{background:#0f3460!important;color:#e0e0e0!important;border:1px solid #1a1a4e!important}
</style>""", unsafe_allow_html=True)

# ============================================
# 登录
# ============================================
if "user" not in st.session_state:
    st.session_state["user"] = None

if st.session_state["user"] is None:
    st.title("🔐 量化投研系统")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        username = st.text_input("用户名", placeholder="输入你的用户名")
        password = st.text_input("密码", type="password", placeholder="输入密码")
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("登录 / 注册", use_container_width=True):
                if username and password:
                    if verify_user(username, password):
                        st.session_state["user"] = username
                        st.rerun()
                    else:
                        st.error("密码错误")
                else:
                    st.error("用户名和密码不能为空")
        with col_b:
            st.caption("首次输入即自动注册")
            st.caption("数据仅属于你的账户")
    st.stop()

# ---- 已登录，获取用户名 ----
USER = st.session_state["user"]

# ---- 缓存 ----
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

def _run_one(ticker):
    g=build_graph()
    s:StockAgentState={"ticker":ticker,"data_fetch_success":False,"error_message":"","raw_history_data":{},"technical_analysis":"","fundamental_analysis":"","final_report":""}
    try:return g.invoke(s)  # type: ignore[reportAttributeAccessIssue]
    except Exception as e:return{**s,"error_message":str(e)}

# ============================================
# 侧边栏
# ============================================
indices=_idx()
with st.sidebar:
    st.markdown(f"### 📈 {USER} 的投研系统")
    if st.button("🔒 退出登录", use_container_width=True):
        st.session_state["user"]=None;st.rerun()

    if indices:
        for idx in indices:
            c="#e53935" if idx["change"]>0 else "#43a047" if idx["change"]<0 else "#aaa"
            a="▲" if idx["change"]>0 else "▼" if idx["change"]<0 else "—"
            st.markdown(f"""<div style='background:#ffffff10;border-radius:8px;padding:10px 12px;margin:3px 0'>
                <div style='font-size:0.7rem;color:#999'>{idx['name']}</div>
                <div style='font-size:1.1rem;font-weight:700;color:#fff'>{idx['price']:.2f}</div>
                <div style='font-size:0.8rem;font-weight:600;color:{c}'>{a} {idx['change']:+.2f} ({idx['change_pct']:+.2f}%)</div></div>""",unsafe_allow_html=True)
    st.divider()

    page=st.radio("",[
        "🏠  市场概览","📈  个股深度分析","📐  策略回测",
        "🔍  全市场搜索","⭐  持仓管理","🚀  批量量化分析","📋  历史报告",
    ],label_visibility="collapsed")
    st.divider()
    s=get_summary(USER)
    st.metric("你的分析报告",f"{s['total']} 份")
    st.caption("数据: 新浪实时 · 腾讯历史")
    st.caption("AI: DeepSeek-Chat")

def _bar():
    if indices:
        cols=st.columns(len(indices))
        for i,idx in enumerate(indices):
            with cols[i]:
                c="#e53935" if idx["change"]>0 else "#43a047" if idx["change"]<0 else "#888"
                a="▲" if idx["change"]>0 else "▼" if idx["change"]<0 else "—"
                st.markdown(f"""<div class='idx-card'><div class='n'>{idx['name']}</div><div class='p'>{idx['price']:.2f}</div><div style='font-size:0.8rem;font-weight:600;color:{c}'>{a} {idx['change']:+.2f} ({idx['change_pct']:+.2f}%)</div></div>""",unsafe_allow_html=True)

# ============================================
# 页面 1: 市场概览
# ============================================
if page.startswith("🏠"):
    st.title("市场概览")
    st.caption(f"👤 {USER} | 数据更新: {time.strftime('%H:%M:%S')} · A股 5538 支")
    _bar();st.divider()
    col_a,col_b=st.columns([2,1])
    with col_a:
        st.subheader("🔥 成交量活跃榜 Top 20")
        active=_active(30)
        if active:
            rows=[{"代码":q["code"],"名称":q["name"],"现价":f"{q['price']:.2f}","涨跌%":f"{q['change_pct']:+.2f}%","成交量(手)":f"{q.get('volume',0)/100:,.0f}"} for q in active[:20]]
            st.dataframe(pd.DataFrame(rows).style.map(_cc,subset=["涨跌%"]),hide_index=True,height=680)
    with col_b:
        st.subheader("📋 我的持仓快照")
        wc=get_watchlist(USER) or ["600519","000858","600036","300750","000001"]
        qs=_q(tuple(wc[:12]))
        for q in qs:
            if q.get("name"):
                ch=q.get("change",0);c="#e53935" if ch>0 else "#43a047" if ch<0 else "#888"
                st.markdown(f"""<div style='background:#f6f8fa;border-radius:6px;padding:8px 10px;margin:2px 0;display:flex;justify-content:space-between'>
                    <div><strong>{q['code']}</strong><br><small style='color:#888'>{q['name']}</small></div>
                    <div style='text-align:right'><span style='font-weight:700'>{q['price']:.2f}</span><br><small style='color:{c};font-weight:600'>{ch:+.2f} ({q['change_pct']:+.2f}%)</small></div></div>""",unsafe_allow_html=True)
        st.divider()
        sm=get_summary(USER)
        st.metric("我的报告",sm["total"])
        if sm["total"]>0:st.metric("成功率",f"{sm['success']/sm['total']*100:.0f}%")
        if sm.get("ratings"):
            for r,cnt in sorted(sm["ratings"].items(),key=lambda x:-x[1]):st.caption(f"  {r}: {cnt}")

# ============================================
# 页面 2: 个股深度分析
# ============================================
elif page.startswith("📈"):
    st.title("个股深度分析")
    col_sel,_=st.columns([2,3])
    with col_sel:ticker=st.text_input("股票代码",value="600519",max_chars=6,placeholder="600519")
    if ticker and ticker.isdigit() and len(ticker)==6:
        info_q=_q((ticker,));name=info_q[0]["name"] if info_q and info_q[0].get("name") else ""
        st.subheader(f"{ticker} {name} — 历史走势")
        tf=st.radio("时间范围",list(TIMEFRAME_DAYS.keys()),horizontal=True,index=3,key="tf")
        with st.spinner("加载K线..."):
            fig,df=build_kline_chart(ticker,name,tf)
        if fig:st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":True,"displaylogo":False})
        else:st.warning("无法加载K线数据。")

        if df is not None and not df.empty:
            st.divider();st.subheader("📊 量化风险收益指标")
            metrics=calculate_metrics(df)
            if "error" not in metrics:
                cm=[("最新价",f"{metrics['current_price']:.2f}"),("年化收益",f"{metrics['annual_return']:.2f}%"),("年化波动",f"{metrics['annual_volatility']:.2f}%"),("夏普比率",f"{metrics['sharpe_ratio']:.2f}"),("最大回撤",f"{metrics['max_drawdown']:.2f}%"),("盈亏比",f"{metrics['profit_factor']:.2f}"),("日胜率",f"{metrics['win_rate']:.1f}%"),("95% VaR",f"{metrics['var_95']:.2f}%"),("52周高",f"{metrics['price_52w_high']:.2f}"),("52周低",f"{metrics['price_52w_low']:.2f}"),("距高点",f"{metrics['pct_from_high']:.2f}%"),("30日波动",f"{metrics['volatility_30d']:.2f}%"),("趋势",metrics['trend']),("总收益",f"{metrics['total_return']:.2f}%"),("平均盈利",f"{metrics['avg_win']:.2f}%"),("平均亏损",f"{metrics['avg_loss']:.2f}%")]
                for i in range(0,len(cm),8):
                    cols=st.columns(8)
                    for j,(label,value) in enumerate(cm[i:i+8]):
                        with cols[j]:st.markdown(f"""<div class='metric-chip'><div class='v'>{value}</div><div class='l'>{label}</div></div>""",unsafe_allow_html=True)

        st.divider();st.subheader("🤖 AI 投研")
        if st.button("▶️ 生成 AI 投研报告",type="primary",key="gen"):
            _init_rt()
            with st.spinner("分析中..."):
                result=_run_one(ticker);save_result(USER,result)
            if result.get("data_fetch_success"):
                t1,t2,t3=st.tabs(["技术面","基本面","CIO决策"])
                with t1:st.markdown((result.get("technical_analysis")or"").replace("\n","\n\n"))
                with t2:st.markdown((result.get("fundamental_analysis")or"").replace("\n","\n\n"))
                with t3:
                    rt=result.get("rating")or extract_rating(result.get("final_report",""))
                    if rt:st.markdown(f"### 评级: {rt}")
                    st.divider();st.markdown((result.get("final_report")or"").replace("\n","\n\n"))
            else:st.error(f"数据获取失败: {result.get('error_message','')[:200]}")

# ============================================
# 页面 3: 策略回测
# ============================================
elif page.startswith("📐"):
    st.title("📐 25日线 × 25月线 双均线策略")
    st.caption("金叉买入 · 死叉卖出 · 全自动回测")

    col_sel,_=st.columns([2,3])
    with col_sel:ticker=st.text_input("股票代码",value="600519",max_chars=6,placeholder="600519",key="stk")
    if ticker and ticker.isdigit() and len(ticker)==6:
        info_q=_q((ticker,));name=info_q[0]["name"] if info_q and info_q[0].get("name") else ""
        if st.button("🔍 运行策略回测",type="primary",key="run_st"):
            with st.spinner("加载800+交易日数据..."):
                sfig,sresult=build_strategy_chart(ticker,name)
            if sfig and sresult:
                st.plotly_chart(sfig,use_container_width=True,config={"displaylogo":False})
                c1,c2,c3,c4,c5=st.columns(5)
                c1.metric("策略收益",f"{sresult['total_return']:.2f}%")
                c2.metric("买入持有",f"{sresult['buy_hold_return']:.2f}%")
                c3.metric("最大回撤",f"{sresult['max_drawdown']:.2f}%")
                c4.metric("胜率",f"{sresult['win_rate']:.1f}%")
                c5.metric("交易次数",sresult['total_trades'])
                st.info(f"**当前状态**: {sresult['current_status']}")
                if sresult['latest_date']:st.caption(f"最近信号: {sresult['latest_signal']}（{sresult['latest_date']}）")
                if sresult['total_return']>sresult['buy_hold_return']:st.success(f"策略跑赢买入持有 +{sresult['total_return']-sresult['buy_hold_return']:.1f}%")
                else:st.warning(f"策略跑输 {sresult['buy_hold_return']-sresult['total_return']:.1f}%")

# ============================================
# 页面 4: 全市场搜索
# ============================================
elif page.startswith("🔍"):
    st.title("全市场搜索");univ=get_universe_stats()
    st.caption(f"A股 {univ['total']:,} 支")
    col_q,col_f=st.columns([3,1])
    with col_q:query=st.text_input("搜索",placeholder="茅台/宁德/600519")
    with col_f:bf=st.selectbox("板块",["全部"]+sorted(univ.get("boards",{}).keys()))
    if query:
        results=search_stocks(query,60)
        if bf!="全部":results=[r for r in results if r.get("board")==bf]
        if results:
            st.success(f"{len(results)} 支")
            for i in range(0,len(results),5):
                cols=st.columns(5)
                for j,r in enumerate(results[i:i+5]):
                    with cols[j]:
                        st.markdown(f"""<div style='background:#f6f8fa;border-radius:8px;padding:10px;text-align:center;margin:4px 0'><div style='font-weight:700'>{r['code']}</div><div style='font-size:0.85rem;color:#555'>{r['name']}</div><div style='font-size:0.7rem;color:#999'>{r['market']}·{r['board']}</div></div>""",unsafe_allow_html=True)
                        if st.button("➕ 加持仓",key=f"s_{r['code']}"):
                            add_to_watchlist(USER,"默认池",[r["code"]])
                            st.toast(f"已加入: {r['code']} {r['name']}")
    else:
        st.divider();st.subheader("按板块浏览")
        tabs=st.tabs(list(univ.get("boards",{}).keys()))
        for tab,board in zip(tabs,univ.get("boards",{}).keys()):
            with tab:
                stocks=get_by_board(board,24)
                for i in range(0,len(stocks),6):
                    cols=st.columns(6)
                    for j,s in enumerate(stocks[i:i+6]):
                        with cols[j]:st.code(s["code"],language=None);st.caption(s["name"])
                        if st.button("➕",key=f"b_{s['code']}"):add_to_watchlist(USER,"默认池",[s["code"]]);st.toast(f"已加入 {s['code']}")

# ============================================
# 页面 5: 持仓管理
# ============================================
elif page.startswith("⭐"):
    st.title("持仓管理")
    pns=get_watchlist_names(USER) or ["默认池"];al=st.selectbox("股票池",pns)
    codes=get_watchlist(USER,al)
    col_a,col_m=st.columns([3,1])
    with col_a:
        if codes:
            qs=_q(tuple(codes));valid=[q for q in qs if q.get("price",0)>0]
            st.caption(f"持仓 **{len(codes)}** 支 | 有效 **{len(valid)}** 支")
            if valid:
                rows=[{"代码":q["code"],"名称":q["name"],"现价":f"{q['price']:.2f}","涨跌%":f"{q['change_pct']:+.2f}%","量(手)":f"{q.get('volume',0)/100:,.0f}"} for q in valid]
                st.dataframe(pd.DataFrame(rows).style.map(_cc,subset=["涨跌%"]),hide_index=True,height=450)
        else:st.info("持仓为空。去「全市场搜索」添加。")
    with col_m:
        st.subheader("操作")
        batch=st.text_area("批量导入",placeholder="600519,000858,300750")
        if st.button("📥 导入",use_container_width=True) and batch:
            new=[c.strip() for c in batch.replace(" ",",").split(",") if c.strip()]
            v,iv=validate_tickers(new)
            if v:add_to_watchlist(USER,al,v);st.toast(f"已添加 {len(v)} 支")
            if iv:st.error(f"无效: {', '.join(iv)}");st.rerun()
        if codes:
            rm=st.multiselect("移除",codes)
            if st.button("🗑️ 移除",use_container_width=True,disabled=not rm):
                remove_from_watchlist(USER,al,rm);st.rerun()

# ============================================
# 页面 6: 批量量化分析（同步模式，不再卡死）
# ============================================
elif page.startswith("🚀"):
    st.title("批量量化分析")
    _init_rt()

    col_in,col_cfg=st.columns([2,1])
    with col_in:
        src=st.radio("标的来源",["我的持仓池","实时活跃榜 Top20","手动输入"],horizontal=True)
        if "持仓" in src:
            pn=get_watchlist_names(USER) or ["默认池"];sp=st.selectbox("池",pn)
            tickers=get_watchlist(USER,sp)
            if tickers:v,iv=validate_tickers(tickers);tickers=v
            st.success(f"有效: **{len(tickers)}** 支" if tickers else "池为空")
        elif "活跃" in src:tickers=[q["code"] for q in _active(20)];st.info(f"{len(tickers)} 支")
        else:
            raw=st.text_area("代码（每行一个）","600519\n000858\n300750\n000001")
            tickers=[t.strip() for t in raw.split("\n") if t.strip()]
            v,iv=validate_tickers(tickers);tickers=v
    with col_cfg:
        workers=st.slider("并发线程",1,5,3)
        st.caption(f"待分析: **{len(tickers) if tickers else 0}** 支")

    st.divider()

    if st.button("▶️ 启动分析",type="primary",use_container_width=True,disabled=not bool(tickers)):
        pbar=st.progress(0);stxt=st.empty()
        results=[];total=len(tickers)
        with ThreadPoolExecutor(max_workers=min(workers,3)) as pool:
            futs={pool.submit(_run_one,t):t for t in tickers}
            for i,future in enumerate(futs):
                t=futs[future]
                try:final=future.result()
                except Exception as e:final={"ticker":t,"data_fetch_success":False,"error_message":str(e)}
                save_result(USER,final);results.append(final)
                pbar.progress((i+1)/total);stxt.text(f"已完成 {i+1}/{total} — {t}")
        ok=sum(1 for r in results if r.get("data_fetch_success"))
        pbar.empty();stxt.empty()
        c1,c2,c3=st.columns(3)
        c1.metric("成功",ok);c2.metric("失败",total-ok);c3.metric("合计",total)
        if ok:st.success("完成！前往「历史报告」查看。")

# ============================================
# 页面 7: 历史报告
# ============================================
else:
    st.title("历史分析报告")
    all_r=get_results(USER,limit=500)
    if not all_r:st.info("暂无数据。")
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
        detail=st.selectbox("选择股票",sorted(df_all["ticker"].dropna().unique()))
        if detail:
            recs=[r for r in all_r if r["ticker"]==detail]
            if recs:
                r=recs[0];t1,t2,t3=st.tabs(["技术面","基本面","CIO决策"])
                with t1:st.markdown((r.get("technical_analysis")or"*无*").replace("\n","\n\n"))
                with t2:st.markdown((r.get("fundamental_analysis")or"*无*").replace("\n","\n\n"))
                with t3:
                    rt=r.get("rating")or extract_rating(r.get("final_report",""))
                    if rt:st.markdown(f"### {rt}")
                    st.divider();st.markdown((r.get("final_report")or"*无*").replace("\n","\n\n"))
