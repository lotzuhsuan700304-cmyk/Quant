import datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

st.set_page_config(
    page_title="機構級跨市場量化決策與蒙地卡羅戰情室", layout="wide"
)

st.title("🌐 機構級跨市場（美股/台股）量化決策與風控戰情室")
st.markdown(
    "整合 **ATR 風控**、**費波納契空間/週期**、**半凱利倉位** 與 **蒙地卡羅預測**的高穩定量化戰情室。"
)

# 側邊欄設定
st.sidebar.header("⚙️ 投資組合與標的設定")
ticker_input = st.sidebar.text_input(
    "輸入股票代號 (美股如 NBIS / 台股請加 .TW 如 3324.TW)",
    value="4979.TW",
).upper()

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
def fetch_market_data(symbol):
  """高穩定市場數據抓取引擎 (加入 User-Agent 偽裝避免 Yahoo 封鎖)"""
  try:
    # 使用 yfinance 抓取，並透過會話偽裝瀏覽器
    import requests

    session = requests.Session()
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML,"
            " like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
    })

    ticker = yf.Ticker(symbol, session=session)
    df = ticker.history(period="1y")

    if df is None or df.empty or len(df) == 0:
      return None, f"查無代號 {symbol} 的歷史數據，請確認台股是否加上 .TW"
    return df, "yfinance 穩定數據引擎 (已啟用瀏覽器偽裝)"
  except Exception as e:
    return None, f"資料抓取異常: {str(e)}"


def run_institutional_analysis(df, capital, cost, shares, rev_growth_input):
  """機構級純量化分析與蒙地卡羅運算引擎"""
  if df is None or df.empty or "Close" not in df.columns:
    return None

  df["Daily_Return"] = df["Close"].pct_change()
  returns_clean = df["Daily_Return"].dropna()
  wins = returns_clean[returns_clean > 0]
  losses = returns_clean[returns_clean < 0]

  win_rate = (
      len(wins) / len(returns_clean) if len(returns_clean) > 0 else 0.5
  )
  avg_win = wins.mean() if len(wins) > 0 else 0.01
  avg_loss = abs(losses.mean()) if len(losses) > 0 else 0.01
  win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 1.0

  kelly_fraction = (
      (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio
      if win_loss_ratio > 0
      else 0
  )
  half_kelly_pct = max(0.0, kelly_fraction * 0.5) * 100

  df["H-L"] = df["High"] - df["Low"]
  df["H-PC"] = (df["High"] - df["Close"].shift(1)).abs()
  df["L-PC"] = (df["Low"] - df["Close"].shift(1)).abs()
  df["TR"] = df[["H-L", "H-PC", "L-PC"]].max(axis=1)
  df["ATR"] = df["TR"].rolling(14).mean()

  latest_close = df["Close"].iloc[-1]
  latest_atr = (
      df["ATR"].iloc[-1]
      if not np.isnan(df["ATR"].iloc[-1])
      else latest_close * 0.02
  )
  trailing_stop = latest_close - (1.5 * latest_atr)

  recent_df = df.tail(60)
  high_60 = recent_df["High"].max()
  low_60 = recent_df["Low"].min()
  move_range = high_60 - low_60

  fib_382 = high_60 - move_range * 0.382
  fib_ext_1618 = high_60 + move_range * 0.618

  low_idx_label = recent_df["Low"].idxmin()
  low_pos = df.index.get_loc(low_idx_label)
  fib_cycle_offsets = [8, 13, 21, 34, 55]
  cycle_dates = [
      df.index[low_pos + o] for o in fib_cycle_offsets if (low_pos + o) < len(df)
  ]

  mu = returns_clean.mean() if not returns_clean.empty else 0.001
  sigma = returns_clean.std() if not returns_clean.empty else 0.02
  num_simulations = 200
  num_days = 30

  simulated_prices = np.zeros((num_days, num_simulations))
  for i in range(num_simulations):
    sim_returns = np.random.normal(mu, sigma, num_days)
    simulated_prices[:, i] = latest_close * np.cumprod(1 + sim_returns)

  mc_median = np.median(simulated_prices, axis=1)
  mc_p5 = np.percentile(simulated_prices, 5, axis=1)
  mc_p95 = np.percentile(simulated_prices, 95, axis=1)

  current_market_value = latest_close * shares
  pnl = (latest_close - cost) * shares
  pnl_pct = ((latest_close - cost) / cost) * 100 if cost > 0 else 0
  actual_capital_pct = (current_market_value / capital) * 100 if capital > 0 else 0

  score = 50
  breakdown = []

  if actual_capital_pct <= half_kelly_pct * 1.5:
    score += 25
    breakdown.append((
        "風控 - 半凱利匹配",
        f"當前曝險 ({actual_capital_pct:.1f}%) 在安全區間。",
        "+25",
    ))
  else:
    score -= 15
    breakdown.append((
        "風控 - 曝險過高",
        (
            f"當前曝險 ({actual_capital_pct:.1f}%) 過重，建議降至"
            f" {half_kelly_pct:.1f}%。"
        ),
        "-15",
    ))

  if latest_close >= trailing_stop:
    score += 25
    breakdown.append((
        "技術 - ATR 防守",
        f"現價穩居 1.5x ATR 防守線 (${trailing_stop:.2f}) 之上。",
        "+25",
    ))
  else:
    score -= 25
    breakdown.append(
        ("技術 - ATR 破位", f"跌破 ATR 防守線 (${trailing_stop:.2f})。", "-25")
    )

  if rev_growth_input > 15.0:
    score += 25
    breakdown.append(
        ("基本 - 動能強勁", f"預期年增率 {rev_growth_input}%，動能強。", "+25")
    )
  elif rev_growth_input > 0.0:
    score += 10
    breakdown.append(
        ("基本 - 動能平穩", f"預期年增率 {rev_growth_input}%。", "+10")
    )
  else:
    score -= 15
    breakdown.append(("基本 - 動能疲弱", "預期年增率為負或衰退。", "-15"))

  if mc_median[-1] > latest_close:
    score += 25
    breakdown.append((
        "機率 - 蒙地卡羅看多",
        f"30天中位數 (${mc_median[-1]:.2f}) 高於現價。",
        "+25",
    ))
  else:
    score -= 15
    breakdown.append((
        "機率 - 蒙地卡羅看空",
        f"30天中位數 (${mc_median[-1]:.2f}) 低於現價。",
        "-15",
    ))

  score = max(0, min(100, score))
  if score >= 70:
    action = (
        "🟢 強勢續抱 / 蒙地卡羅與凱利模型支援",
        "各項風控指標與期望值均在理想區間，讓利潤奔跑。",
    )
  elif score >= 45:
    action = (
        "🟡 區間震盪 / 依 ATR 與半凱利調整部位",
        "多空拉鋸，嚴守 ATR 防守線。",
    )
  else:
    action = (
        "🔴 觸發防守 / 建議執行減碼",
        "評分過低或超越曝險上限，啟動保護機制。",
    )

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
df, source_msg = fetch_market_data(ticker_input)

if df is None:
  st.error(
      f"⚠️ {source_msg}\n\n請確認代號是否正確（例如台股請輸入 `4979.TW`，美股請輸入"
      " `NBIS`）。"
  )
else:
  data = run_institutional_analysis(
      df, total_capital, cost_input, shares_input, manual_rev_growth
  )

  if data is None:
    st.error("⚠️ 數據解析失敗，該標的歷史資料筆數不足或格式異常。")
  else:
    st.success(f"✅ 成功載入數據源：**{source_msg}**")

    # 頂部戰情指標
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("最新參考價", f"${data['close']:.2f}")
    c2.metric("未實現損益", f"${data['pnl']:,.2f}", f"{data['pnl_pct']:+.2f}%")
    c3.metric(
        "🎯 半凱利上限",
        f"{data['half_kelly_pct']:.1f}%",
        f"目前曝險: {data['actual_capital_pct']:.1f}%",
    )
    c4.metric("🎲 MC 30天預期價", f"${data['mc_median_final']:.2f}")
    c5.metric("📊 機構量化評分", f"{data['score']} 分", "/ 100")

    st.markdown("---")

    col_left, col_right = st.columns([1.4, 0.6])

    with col_left:
      st.subheader(
          f"📈 {ticker_input} 費波納契空間與 ATR 軌道 [歷史勝率:"
          f" {data['win_rate']:.1f}%]"
      )
      chart_df = data["df"].tail(120)
      fig = go.Figure()

      fig.add_trace(
          go.Candlestick(
              x=chart_df.index,
              open=chart_df["Open"],
              high=chart_df["High"],
              low=chart_df["Low"],
              close=chart_df["Close"],
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
          y=data["fib_ext_1618"],
          line_dash="dash",
          line_color="gold",
          annotation_text=f"Fib 1.618目標 ${data['fib_ext_1618']:.2f}",
      )

      for i, c_date in enumerate(data["cycle_dates"]):
        if c_date >= chart_df.index[0]:
          fig.add_vline(
              x=c_date,
              line_dash="dot",
              line_color="cyan",
              annotation_text=f"週期窗 {i+1}",
          )

      fig.update_layout(
          height=500,
          margin=dict(l=10, r=10, t=25, b=10),
          xaxis_rangeslider_visible=False,
          template="plotly_dark",
      )
      st.plotly_chart(
          fig, use_container_width=True, key="browser_agent_war_room_chart"
      )

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
