import datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="跨市場多維度客觀量化決策戰情室", layout="wide")

st.title("🌐 跨市場（美股/台股）多維度客觀量化決策戰情室")
st.markdown(
    "輸入任意股票代號（美股如 `NBIS`, `AVGO`；台股如 `3324.TW`、`2330.TW`），系統自動整合**技術面、基本面、籌碼流動性與風控模型**進行客觀計分與決策。"
)

# 側邊欄參數設定
st.sidebar.header("⚙️ 投資組合與標的設定")
ticker_input = st.sidebar.text_input(
    "輸入股票代號 (美股/台股)", value="NBIS"
).upper()
shares_input = st.sidebar.number_input(
    "目前持股數量", min_value=0, value=580, step=10
)
cost_input = st.sidebar.number_input(
    "平均持有成本", min_value=0.0, value=226.5, step=0.5
)


@st.cache_data(ttl=300)
def fetch_comprehensive_analysis(symbol, cost, shares):
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period="6mo")
        if df.empty:
            return None, "查無此股票數據，請確認代號是否正確（台股請加上 .TW）。"

        # 1. 技術面指標計算 (ATR & Fibonacci)
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
        fib_618 = high_60 - (high_60 - low_60) * 0.618

        # 2. 基本面與估值指標擷取 (透過 yfinance info)
        info = {}
        try:
            info = ticker.info
        except:
            pass

        pe_ratio = info.get("trailingPE", None)
        revenue_growth = info.get("revenueGrowth", None)

        # 3. 籌碼與流動性代理指標 (近期成交量變化率)
        avg_vol_20 = df["Volume"].tail(20).mean()
        latest_vol = df["Volume"].iloc[-1]
        vol_ratio = latest_vol / avg_vol_20 if avg_vol_20 > 0 else 1.0

        # 4. 損益計算
        pnl = (latest_close - cost) * shares
        pnl_pct = ((latest_close - cost) / cost) * 100

        # 5. 多維度客觀計分模型 (總分 100 分)
        score = 50  # 基準分
        breakdown = []

        # 維度一：技術結構 (佔 30 分)
        tech_score = 0
        if latest_close >= trailing_stop:
            tech_score += 15
            breakdown.append(
                (
                    "技術面 - ATR 防守",
                    f"現價高於 1.5x ATR 防守線 (${trailing_stop:.2f})，結構未破位。",
                    "+15",
                )
            )
        else:
            tech_score -= 15
            breakdown.append(
                (
                    "技術面 - ATR 破位",
                    f"現價已跌破 ATR 防守線 (${trailing_stop:.2f})，趨勢轉弱。",
                    "-15",
                )
            )

        if latest_close >= fib_382:
            tech_score += 15
            breakdown.append(
                (
                    "技術面 - 費波納契",
                    f"現價突破 Fib 0.382 關鍵壓力位 (${fib_382:.2f})。",
                    "+15",
                )
            )
        else:
            tech_score -= 10
            breakdown.append(
                (
                    "技術面 - 費波納契",
                    f"現價低於 Fib 0.382 壓力位 (${fib_382:.2f})。",
                    "-10",
                )
            )
        score += tech_score

        # 維度二：基本面 / 動能 (佔 30 分)
        fund_score = 0
        if revenue_growth is not None:
            if revenue_growth > 0.15:
                fund_score += 20
                breakdown.append(
                    (
                        "基本面 - 營收動能",
                        f"最新營收年增率達 {revenue_growth*100:.1f}% (>15%)，動能強勁。",
                        "+20",
                    )
                )
            elif revenue_growth > 0:
                fund_score += 10
                breakdown.append(
                    (
                        "基本面 - 營收動能",
                        f"營收年增率為正 ({revenue_growth*100:.1f}%)。",
                        "+10",
                    )
                )
            else:
                fund_score -= 15
                breakdown.append(
                    (
                        "基本面 - 營收動能",
                        f"營收年增率為負 ({revenue_growth*100:.1f}%)。",
                        "-15",
                    )
                )
        else:
            fund_score += 10  # 無數據給中立分
            breakdown.append(
                (
                    "基本面 - 財務數據",
                    "無法取得即時財報成長率，給予中立評分。",
                    "+10",
                )
            )

        if pe_ratio is not None and pe_ratio > 0:
            if pe_ratio < 40:
                fund_score += 10
                breakdown.append(
                    (
                        "估值面 - 本益比",
                        f"本益比 ({pe_ratio:.1f}x) 處於合理或成長匹配區間。",
                        "+10",
                    )
                )
            else:
                fund_score -= 5
                breakdown.append(
                    (
                        "估值面 - 本益比",
                        f"本益比偏高 ({pe_ratio:.1f}x)，估值壓力較大。",
                        "-5",
                    )
                )
        score += fund_score

        # 維度三：籌碼與成交量流動性 (佔 20 分)
        flow_score = 0
        if vol_ratio >= 1.2:
            flow_score += 15
            breakdown.append(
                (
                    "籌碼流動 - 量能活躍",
                    f"當日成交量為 20 日均量之 {vol_ratio:.1f} 倍，資金活躍度高。",
                    "+15",
                )
            )
        elif vol_ratio >= 0.8:
            flow_score += 10
            breakdown.append(
                (
                    "籌碼流動 - 量能穩定",
                    "成交量貼近近期均量，市場交投平穩。",
                    "+10",
                )
            )
        else:
            flow_score -= 5
            breakdown.append(
                (
                    "籌碼流動 - 量能萎縮",
                    "成交量低於近期均量，追價意願不足。",
                    "-5",
                )
            )
        score += flow_score

        # 維度四：成本與風險期望值 (佔 20 分)
        risk_score = 0
        if latest_close >= cost:
            risk_score += 15
            breakdown.append(
                (
                    "風控期望 - 成本安全",
                    f"現價 (${latest_close:.2f}) 高於成本 (${cost})，具備盈利保護。",
                    "+15",
                )
            )
        else:
            risk_score -= 10
            breakdown.append(
                (
                    "風控期望 - 浮動套牢",
                    f"現價 (${latest_close:.2f}) 低於成本 (${cost})，面臨回撤考驗。",
                    "-10",
                )
            )
        score += risk_score

        # 總分標準化與決策指令
        score = max(0, min(100, score))
        if score >= 70:
            action = (
                "🟢 強勢續抱 / 順勢波段操作",
                "多維度結構穩健，動能與風控均在理想範圍內，讓利潤奔跑。",
            )
        elif score >= 45:
            action = (
                "🟡 區間震盪 / 嚴守防守線觀望",
                "多空力道拉鋸，無結構性破位，依 ATR 防守線紀律應對。",
            )
        else:
            action = (
                "🔴 觸發防守 / 執行減碼或停損",
                "多維度計分偏低或跌破關鍵防守位，啟動機械化風險保護機制。",
            )

        return {
            "close": latest_close,
            "atr": latest_atr,
            "trailing_stop": trailing_stop,
            "fib_382": fib_382,
            "fib_618": fib_618,
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
data, err = fetch_comprehensive_analysis(ticker_input, cost_input, shares_input)

if err:
    st.error(f"分析載入錯誤：{err}")
else:
    # 頂部核心戰情指標
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("最新參考價", f"${data['close']:.2f}")
    c2.metric("未實現損益", f"${data['pnl']:,.2f}", f"{data['pnl_pct']:+.2f}%")
    c3.metric("🔴 ATR 動態防守線", f"${data['trailing_stop']:.2f}")
    c4.metric("📊 多維度量化評分", f"{data['score']} 分", "/ 100 分")

    st.markdown("---")

    # 左右雙欄：左側 K線與技術軌道，右側多維度論點與機械決策
    col_left, col_right = st.columns([1.4, 0.6])

    with col_left:
        st.subheader(f"📈 {ticker_input} 技術結構與 ATR / 費波納契動態軌道")
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
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("⚖️ 機械化決策與多維度評估")
        act_title, act_desc = data["action"]
        st.info(f"**【最終機械化指令】**\n\n### {act_title}\n{act_desc}")

        st.markdown("#### 🔍 四維度客觀得分明細：")
        for cat, desc, pts in data["breakdown"]:
            st.markdown(f"- **[{cat}]** ({pts}分)\n  {desc}")

        st.markdown("---")
        st.caption(
            "💡 系統完全依據技術面、基本面代理指標、成交量流動性與風控期望值自動加權運算，杜絕主觀情緒凹單。"
        )
