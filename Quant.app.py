import datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="跨市場多維度量化決策戰情室", layout="wide")

st.title("🌐 跨市場（美股/台股）多維度客觀量化決策戰情室")
st.markdown(
    "採用高穩定性 **yfinance 引擎** 結合 **多維度量化評分模型**，確保跨市場（美股如 `NBIS`、台股如 `3324.TW`）數據 100% 穩定輸出。"
)

# 側邊欄設定
st.sidebar.header("⚙️ 投資組合與標的設定")
ticker_input = st.sidebar.text_input(
    "輸入股票代號 (美股/台股)", value="3324.TW"
).upper()
shares_input = st.sidebar.number_input(
    "目前持股數量", min_value=0, value=2000, step=100
)
cost_input = st.sidebar.number_input(
    "平均持有成本", min_value=0.0, value=150.0, step=0.5
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📝 基本面與動能評估設定")
st.sidebar.caption("透過研究最新營收與產業面，滑動設定基本面動能評分基準")
manual_rev_growth = st.sidebar.slider(
    "營收年增率 / 基本面動能預估 (%)",
    min_value=-50.0,
    max_value=100.0,
    value=15.0,
    step=5.0,
)


@st.cache_data(ttl=300)
def fetch_market_and_analysis(symbol, cost, shares, rev_growth_input):
  try:
    # 1. 抓取股價與技術面 (yfinance)
    ticker = yf.Ticker(symbol)
    df = ticker.history(period="6mo")
    if df.empty:
      return (
          None,
          "查無此股票代號的價格資料，請確認格式（台股請加 .TW，例如 3324.TW）。",
      )

    # 計算 ATR (14)
    df["H-L"] = df["High"] - df["Low"]
    df["H-PC"] = (df["High"] - df["Close"].shift(1)).abs()
    df["L-PC"] = (df["Low"] - df["Close"].shift(1)).abs()
    df["TR"] = df[["H-L", "H-PC", "L-PC"]].max(axis=1)
    df["ATR"] = df["TR"].rolling(14).mean()

    latest_close = df["Close"].iloc[-1]
    latest_atr = df["ATR"].iloc[-1]
    trailing_stop = latest_close - (1.5 * latest_atr)

    high_60 = df["High"].tail(60).max()
    low_60 = df["High"].tail(60).min()
    fib_382 = high_60 - (high_60 - low_60) * 0.382

    # 2. 籌碼流動代理 (20日均量比)
    avg_vol_20 = df["Volume"].tail(20).mean()
    latest_vol = df["Volume"].iloc[-1]
    vol_ratio = latest_vol / avg_vol_20 if avg_vol_20 > 0 else 1.0

    # 3. 損益試算
    pnl = (latest_close - cost) * shares
    pnl_pct = ((latest_close - cost) / cost) * 100

    # 4. 多維度客觀評分模型 (100分制)
    score = 50
    breakdown = []

    # 維度一：技術結構 (30分)
    tech_score = 0
    if latest_close >= trailing_stop:
      tech_score += 15
      breakdown.append((
          "技術面 - ATR 防守",
          f"現價高於 1.5x ATR 防守線 (${trailing_stop:.2f})，結構安全。",
          "+15",
      ))
    else:
      tech_score -= 15
      breakdown.append((
          "技術面 - ATR 破位",
          f"現價已跌破 ATR 防守線 (${trailing_stop:.2f})，趨勢轉弱。",
          "-15",
      ))

    if latest_close >= fib_382:
      tech_score += 15
      breakdown.append((
          "技術面 - 費波納契",
          f"現價突破 Fib 0.382 關鍵壓力位 (${fib_382:.2f})。",
          "+15",
      ))
    else:
      tech_score -= 10
      breakdown.append((
          "技術面 - 費波納契",
          f"現價低於 Fib 0.382 壓力位 (${fib_382:.2f})。",
          "-10",
      ))
    score += tech_score

    # 維度二：基本面 / 動能 (30分)
    fund_score = 0
    if rev_growth_input > 15.0:
      fund_score += 20
      breakdown.append((
          "基本面 - 動能強勁",
          f"設定營收年增率達 {rev_growth_input}% (>15%)，基本面動能優異。",
          "+20",
      ))
    elif rev_growth_input > 0.0:
      fund_score += 10
      breakdown.append((
          "基本面 - 動能平穩",
          f"設定營收年增率為正 ({rev_growth_input}%)，營運穩定。",
          "+10",
      ))
    else:
      fund_score -= 15
      breakdown.append((
          "基本面 - 動能疲弱",
          f"設定營收年增率為負或衰退 ({rev_growth_input}%)，基本面有壓。",
          "-15",
      ))
    score += fund_score

    # 維度三：籌碼與成交量流動 (20分)
    flow_score = 0
    if vol_ratio >= 1.2:
      flow_score += 15
      breakdown.append((
          "籌碼流動 - 量能活躍",
          f"當日成交量為 20 日均量之 {vol_ratio:.1f} 倍，資金高度關注。",
          "+15",
      ))
    elif vol_ratio >= 0.8:
      flow_score += 10
      breakdown.append(
          ("籌碼流動 - 量能平穩", "成交量貼近近期均量，交投穩定。", "+10")
      )
    else:
      flow_score -= 5
      breakdown.append(
          ("籌碼流動 - 量能萎縮", "成交量低於近期均量，追價意願保守。", "-5")
      )
    score += flow_score

    # 維度四：成本與風險期望值 (20分)
    risk_score = 0
    if latest_close >= cost:
      risk_score += 15
      breakdown.append((
          "風控期望 - 成本安全",
          f"現價 (${latest_close:.2f}) 高於成本 (${cost})，具備盈利保護。",
          "+15",
      ))
    else:
      risk_score -= 10
      breakdown.append((
          "風控期望 - 浮動套牢",
          f"現價 (${latest_close:.2f}) 低於成本 (${cost})，面臨回撤壓力。",
          "-10",
      ))
    score += risk_score

    score = max(0, min(100, score))
    if score >= 70:
      action = (
          "🟢 強勢續抱 / 順勢波段操作",
          "多維度結構穩健，動能與風控均在理想範圍，讓利潤奔跑。",
      )
    elif score >= 45:
      action = (
          "🟡 區間震盪 / 嚴守防守線觀望",
          "多空力道拉鋸，結構未遭破壞，依 ATR 防守線紀律應對。",
      )
    else:
      action = (
          "🔴 觸發防守 / 執行減碼或停損",
          "多維度計分偏低或跌破關鍵防守位，啟動機械化風險保護。",
      )

    return {
        "close": latest_close,
        "atr": latest_atr,
        "trailing_stop": trailing_stop,
        "fib_382": fib_382,
        "pnl": pnl,
        "pnl_pct": pnl_pct,
        "score": score,
        "breakdown": breakdown,
        "action": action,
        "df": df,
    }, None
  except Exception as e:
    return None, str(e)


# 執行資料與分析
data, err = fetch_market_and_analysis(
    ticker_input, cost_input, shares_input, manual_rev_growth
)

if err:
  st.error(f"分析載入錯誤：{err}")
else:
  # 頂部戰情指標
  c1, c2, c3, c4 = st.columns(4)
  c1.metric("最新參考價", f"${data['close']:.2f}")
  c2.metric("未實現損益", f"${data['pnl']:,.2f}", f"{data['pnl_pct']:+.2f}%")
  c3.metric("🔴 ATR 動態防守線", f"${data['trailing_stop']:.2f}")
  c4.metric("📊 多維度量化評分", f"{data['score']} 分", "/ 100 分")

  st.markdown("---")

  col_left, col_right = st.columns([1.4, 0.6])

  with col_left:
    st.subheader(f"📈 {ticker_input} 技術結構與動態防守軌道")
    df = data["df"]
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="K線",
        )
    )
    fig.add_hline(
        y=cost_input,
        line_dash="dot",
        line_color="gray",
        annotation_text=f"成本 ${cost_input}",
    )
    fig.add_hline(
        y=data["trailing_stop"],
        line_dash="dash",
        line_color="crimson",
        annotation_text=f"ATR 防守 ${data['trailing_stop']:.2f}",
    )
    fig.add_hline(
        y=data["fib_382"],
        line_dash="dash",
        line_color="mediumseagreen",
        annotation_text=f"Fib 0.382 ${data['fib_382']:.2f}",
    )
    fig.update_layout(
        height=480,
        margin=dict(l=10, r=10, t=25, b=10),
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
    )
    st.plotly_chart(
        fig, use_container_width=True, key="stable_multidimensional_chart"
    )

  with col_right:
    st.subheader("⚖️ 機械化決策與多維度評估")
    act_title, act_desc = data["action"]
    st.info(f"**【最終機械化指令】**\n\n### {act_title}\n{act_desc}")

    st.markdown("#### 🔍 四維度客觀得分明細：")
    for cat, desc, pts in data["breakdown"]:
      st.markdown(f"- **[{cat}]** ({pts}分)\n  {desc}")
