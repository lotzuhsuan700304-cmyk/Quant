import datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf

st.set_page_config(
    page_title="機構級跨市場量化決策與蒙地卡羅戰情室", layout="wide"
)

st.title("🌐 機構級跨市場（美股/台股）雙引擎量化戰情室")
st.markdown(
    "整合 **Fugle 富果 API (台股)** 與 **yfinance (美股)** 雙引擎，具備 **ATR 風控**、**費波納契空間/週期**、**半凱利倉位** 與 **蒙地卡羅預測**。"
)

# 側邊欄設定
st.sidebar.header("⚙️ 投資組合與標的設定")
ticker_input = st.sidebar.text_input(
    "輸入股票代號 (美股如 NBIS / 台股如 3324)", value="3324"
).upper()

# ⚠️ 資安設計：富果 API Key 透過側邊欄輸入，避免寫死在公開程式碼中
fugle_api_key = st.sidebar.text_input(
    "🔑 富果 Fugle API Key (看台股必填)", type="password"
)

total_capital = st.sidebar.number_input(
    "總可用資金 (USD/NTD)", min_value=1000.0, value=2000000.0, step=10000.0
)
shares_input = st.sidebar.number_input(
    "目前持股數量", min_value=0, value=2000, step=100
)
cost_input = st.sidebar.number_input(
    "平均持有成本", min_value=0.0, value=150.0, step=0.5
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📝 基本面動能預估")
manual_rev_growth = st.sidebar.slider(
    "營收年增率 / 產業動能預估 (%)",
    min_value=-50.0,
    max_value=100.0,
    value=20.0,
    step=5.0,
)


@st.cache_data(ttl=300)
def fetch_market_data_dual_engine(symbol, fugle_key):
    """雙引擎數據抓取模組：自動判別美股/台股並套用對應 API"""
    is_taiwan_stock = any(char.isdigit() for char in symbol)

    if is_taiwan_stock:
        # --- 台股：使用 Fugle 富果 API ---
        if not fugle_key:
            return None, "請在左側邊欄輸入你的富果 API Key 以抓取台股數據。"
        
        stock_id = symbol.replace(".TW", "").replace(".TWO", "").strip()
        start_date = (datetime.date.today() - datetime.timedelta(days=365)).strftime("%Y-%m-%d")
        end_date = datetime.date.today().strftime("%Y-%m-%d")
        
        url = f"https://api.fugle.tw/marketdata/v1.0/stock/historical/candles/{stock_id}?timeframe=D&from={start_date}&to={end_date}"
        headers = {"X-API-KEY": fugle_key}
        
        try:
            res = requests.get(url, headers=headers, timeout=10)
            if res.status_code != 200:
                # 備援機制：如果富果 API 失敗，退回 yfinance
                df = yf.Ticker(f"{stock_id}.TW").history(period="1y")
                return df, "yfinance (富果連線失敗，啟動備援)"
            
            data = res.json()
            if "data" not in data or len(data["data"]) == 0:
                return None, "富果 API 回傳空資料，請確認台股代號是否正確。"
            
            # 將富果資料格式化為與 yfinance 完全相同的 DataFrame 結構
            df = pd.DataFrame(data["data"])
            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)
            df.rename(
                columns={
                    "open": "Open",
                    "high": "High",
                    "low": "Low",
                    "close": "Close",
                    "volume": "Volume",
                },
                inplace=True,
            )
            df = df.sort_index()  # 確保時間由舊到新排列
            return df, "Fugle 富果官方 API"
        except Exception as e:
            return None, f"富果 API 發生錯誤: {str(e)}"
    
    else:
        # --- 美股：使用 yfinance API ---
        try:
            df = yf.Ticker(symbol).history(period="1y")
            if df.empty:
                return None, "查無此美股代號資料，請確認拼字 (例如: NBIS)。"
            return df, "yfinance API"
        except Exception as e:
            return None, f"yfinance 錯誤: {str(e)}"


def run_institutional_analysis(df, capital, cost, shares, rev_growth_input):
    """純量化分析引擎 (完全解耦，不綁定特定 API)"""
    # 1. 計算勝率與盈虧比 (凱利公式)
    df["Daily_Return"] = df["Close"].pct_change()
    returns_clean = df["Daily_Return"].dropna()
    wins = returns_clean[returns_clean > 0]
    losses = returns_clean[returns_clean < 0]

    win_rate = len(wins) / len(returns_clean) if len(returns_clean) > 0 else 0.5
    avg_win = wins.mean() if len(wins) > 0 else 0.01
    avg_loss = abs(losses.mean()) if len(losses) > 0 else 0.01
    win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 1.0

    kelly_fraction = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio if win_loss_ratio > 0 else 0
    half_kelly_pct = max(0.0, kelly_fraction * 0.5) * 100

    # 2. 計算 ATR (14)
    df["H-L"] = df["High"] - df["Low"]
    df["H-PC"] = (df["High"] - df["Close"].shift(1)).abs()
    df["L-PC"] = (df["Low"] - df["Close"].shift(1)).abs()
    df["TR"] = df[["H-L", "H-PC", "L-PC"]].max(axis=1)
    df["ATR"] = df["TR"].rolling(14).mean()

    latest_close = df["Close"].iloc[-1]
    latest_atr = df["ATR"].iloc[-1]
    trailing_stop = latest_close - (1.5 * latest_atr)

    # 3. 費波納契空間與週期
    recent_df = df.tail(60)
    high_60 = recent_df["High"].max()
    low_60 = recent_df["Low"].min()
    move_range = high_60 - low_60

    fib_382 = high_60 - move_range * 0.382
    fib_ext_1618 = high_60 + move_range * 0.618

    low_idx_label = recent_df["Low"].idxmin()
    low_pos = df.index.get_loc(low_idx_label)
    fib_cycle_offsets = [8, 13, 21, 34, 55]
    cycle_dates = [df.index[low_pos + o] for o in fib_cycle_offsets if (low_pos + o) < len(df)]

    # 4. 蒙地卡羅模擬 (30 天)
    mu = returns_clean.mean()
    sigma = returns_clean.std()
    num_simulations = 200
    num_days = 30

    simulated_prices = np.zeros((num_days, num_simulations))
    for i in range(num_simulations):
        sim_returns = np.random.normal(mu, sigma, num_days)
        simulated_prices[:, i] = latest_close * np.cumprod(1 + sim_returns)

    mc_median = np.median(simulated_prices, axis=1)
    mc_p5 = np.percentile(simulated_prices, 5, axis=1)
    mc_p95 = np.percentile(simulated_prices, 95, axis=1)

    # 5. 損益與部位試算
    current_market_value = latest_close * shares
    pnl = (latest_close - cost) * shares
    pnl_pct = ((latest_close - cost) / cost) * 100
    actual_capital_pct = (current_market_value / capital) * 100 if capital > 0 else 0

    # 6. 多維度計分與決策
    score = 50
    breakdown = []

    if actual_capital_pct <= half_kelly_pct * 1.5:
        score += 25
        breakdown.append(("風控 - 半凱利匹配", f"當前曝險 ({actual_capital_pct:.1f}%) 在安全區間。", "+25"))
    else:
        score -= 15
        breakdown.append(("風控 - 曝險過高", f"當前曝險 ({actual_capital_pct:.1f}%) 過重，建議降至 {half_kelly_pct:.1f}%。", "-15"))

    if latest_close >= trailing_stop:
        score += 25
        breakdown.append(("技術 - ATR 防守", f"現價穩居 1.5x ATR 防守線 (${trailing_stop:.2f}) 之上。", "+25"))
    else:
        score -= 25
        breakdown.append(("技術 - ATR 破位", f"跌破 ATR 防守線 (${trailing_stop:.2f})。", "-25"))

    if rev_growth_input > 15.0:
        score += 25
        breakdown.append(("基本 - 動能強勁", f"預期年增率 {rev_growth_input}%，動能強。", "+25"))
    elif rev_growth_input > 0.0:
        score += 10
        breakdown.append(("基本 - 動能平穩", f"預期年增率 {rev_growth_input}%。", "+10"))
    else:
        score -= 15
        breakdown.append(("基本 - 動能疲弱", f"預期年增率為負或衰退。", "-15"))

    if mc_median[-1] > latest_close:
        score += 25
        breakdown.append(("機率 - 蒙地卡羅看多", f"30天中位數 (${mc_median[-1]:.2f}) 高於現價。", "+25"))
    else:
        score -= 15
        breakdown.append(("機率 - 蒙地卡羅看空", f"30天中位數 (${mc_median[-1]:.2f}) 低於現價。", "-15"))

    score = max(0, min(100, score))
    if score >= 70:
        action = ("🟢 強勢續抱 / 蒙地卡羅與凱利模型支援", "各項風控指標與期望值均在理想區間，讓利潤奔跑。")
    elif score >= 45:
        action = ("🟡 區間震盪 / 依 ATR 與半凱利調整部位", "多空拉鋸，嚴守 ATR 防守線。")
    else:
        action = ("🔴 觸發防守 / 建議執行減碼", "評分過低或超越曝險上限，啟動保護機制。")

    return {
        "close": latest_close,
        "atr": latest_atr,
        "trailing_stop": trailing_stop,
        "fib_382": fib_382,
        "fib_ext_1618": fib_ext_1618,
        "cycle_dates": cycle_dates,
        "win_rate": win_rate * 100,
        "half_kelly_pct": half_kelly_pct,
        "actual_capital_pct": actual_capital_pct,
        "mc_median_final": mc_median[-1],
        "mc_p5_final": mc_p5[-1],
        "mc_p95_final": mc_p95[-1],
        "pnl": pnl,
        "pnl_pct": pnl_pct,
        "score": score,
        "breakdown": breakdown,
        "action": action,
        "df": df,
    }

# --- 主程式執行區 ---
df, data_source = fetch_market_data_dual_engine(ticker_input, fugle_api_key)

if df is None:
    st.error(f"⚠️ {data_source}")
else:
    data = run_institutional_analysis(df, total_capital, cost_input, shares_input, manual_rev_growth)
    
    st.info(f"✅ 成功載入數據。目前資料來源：**{data_source}**")
    
    # 頂部戰情指標
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("最新參考價", f"${data['close']:.2f}")
    c2.metric("未實現損益", f"${data['pnl']:,.2f}", f"{data['pnl_pct']:+.2f}%")
    c3.metric("🎯 半凱利上限", f"{data['half_kelly_pct']:.1f}%", f"目前曝險: {data['actual_capital_pct']:.1f}%")
    c4.metric("🎲 MC 30天預期價", f"${data['mc_median_final']:.2f}")
    c5.metric("📊 機構量化評分", f"{data['score']} 分", "/ 100")

    st.markdown("---")

    col_left, col_right = st.columns([1.4, 0.6])

    with col_left:
        st.subheader(f"📈 {ticker_input} 費波納契空間與 ATR 軌道 [歷史勝率: {data['win_rate']:.1f}%]")
        chart_df = data["df"].tail(120)
        fig = go.Figure()

        fig.add_trace(go.Candlestick(
            x=chart_df.index, open=chart_df["Open"], high=chart_df["High"],
            low=chart_df["Low"], close=chart_df["Close"], name="K線"
        ))
        fig.add_hline(y=cost_input, line_dash="dot", line_color="gray", annotation_text=f"成本 ${cost_input}")
        fig.add_hline(y=data["trailing_stop"], line_dash="dash", line_color="crimson", annotation_text=f"ATR 防守 ${data['trailing_stop']:.2f}")
        fig.add_hline(y=data["fib_ext_1618"], line_dash="dash", line_color="gold", annotation_text=f"Fib 1.618目標 ${data['fib_ext_1618']:.2f}")

        for i, c_date in enumerate(data["cycle_dates"]):
            if c_date >= chart_df.index[0]:
                fig.add_vline(x=c_date, line_dash="dot", line_color="cyan", annotation_text=f"週期窗 {i+1}")

        fig.update_layout(
            height=500, margin=dict(l=10, r=10, t=25, b=10),
            xaxis_rangeslider_visible=False, template="plotly_dark"
        )
        st.plotly_chart(fig, use_container_width=True, key="fugle_dual_engine_chart")

    with col_right:
        st.subheader("⚖️ 決策與蒙地卡羅明細")
        act_title, act_desc = data["action"]
        st.success(f"**【最終機械化指令】**\n\n### {act_title}\n{act_desc}")

        st.markdown("#### 🎲 蒙地卡羅 30 天預測區間：")
        st.markdown(f"- **🟢 95% 樂觀極值 (P95)**：`${data['mc_p95_final']:.2f}`")
        st.markdown(f"- **🟡 30 天機率中位數**：`${data['mc_median_final']:.2f}`")
        st.markdown(f"- **🔴 5% 悲觀極值 (P5)**：`${data['mc_p5_final']:.2f}`")

        st.markdown("---")
        st.markdown("#### 🔍 評分明細：")
        for cat, desc, pts in data["breakdown"]:
            st.markdown(f"- **[{cat}]** ({pts}分)\n  {desc}")
