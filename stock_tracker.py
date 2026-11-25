import time
import numpy as np
import pandas as pd
import requests
import yfinance as yf

# Mapeo de referencia por sector/industria a ETF (con respaldo a SPY si no hay ETF específico)
INDUSTRY_OR_SECTOR_ETF_MAP = {
    "Semiconductors": "SMH",
    "Semiconductor Equipment & Materials": "SMH",
    "Biotechnology": "XBI",
    "Oil & Gas E&P": "XOP",
    "Oil & Gas Equipment & Services": "OIH",
    "Banks - Regional": "KRE",
    "Aerospace & Defense": "ITA",
    "Solar": "TAN",
    "Technology": "XLK",
    "Financial Services": "XLF",
    "Healthcare": "XLV",
    "Consumer Cyclical": "XLY",
    "Consumer Defensive": "XLP",
    "Industrials": "XLI",
    "Energy": "XLE",
    "Basic Materials": "XLB",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    "Communication Services": "XLC",
}


def get_tickers():
    """Descarga tickers de EE. UU. y limpia símbolos no estándar."""
    try:
        url = "https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/main/all/all_tickers.txt"
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=20)
        resp.raise_for_status()

        raw_tickers = [
            t.strip().upper().replace(".", "-") for t in resp.text.splitlines()
        ]
        clean_tickers = sorted(
            {
                t
                for t in raw_tickers
                if t and t.replace("-", "").isalpha() and len(t) <= 6
            }
        )
        print(f"📋 Tickers cargados: {len(clean_tickers)}")
        return clean_tickers
    except Exception as e:
        print(f"❌ Error descargando lista de tickers: {e}")
        return []


def download_in_batches(tickers, batch_size=500, pause_sec=1):
    """Descarga historial de 1 año por lotes rápidos incluyendo SPY y ETFs de referencia."""
    ref_etfs = sorted(set(INDUSTRY_OR_SECTOR_ETF_MAP.values()))
    symbols = sorted(set(tickers + ["SPY"] + ref_etfs))
    frames = []
    total_batches = (len(symbols) + batch_size - 1) // batch_size

    for idx in range(0, len(symbols), batch_size):
        batch_num = idx // batch_size + 1
        batch = symbols[idx : idx + batch_size]
        print(f"📥 Lote {batch_num}/{total_batches} ({len(batch)} símbolos)...")
        try:
            df_batch = yf.download(
                batch,
                period="1y",
                group_by="ticker",
                threads=True,
                progress=False,
            )
            if df_batch is not None and not df_batch.empty:
                frames.append(df_batch)
        except Exception as e:
            print(f"⚠️ Error en lote {batch_num}: {e}")

        if batch_num < total_batches and pause_sec > 0:
            time.sleep(pause_sec)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, axis=1)
    combined = combined.loc[:, ~combined.columns.duplicated()]
    return combined


def calc_period_return(close_series, lookback=80):
    """Retorno porcentual en lookback ruedas."""
    s = close_series.dropna()
    if len(s) <= lookback:
        return None
    prev_val = float(s.iloc[-lookback - 1])
    curr_val = float(s.iloc[-1])
    if prev_val <= 0:
        return None
    return (curr_val - prev_val) / prev_val


def ols_slope_normalized(close_series, n_bars):
    """Pendiente de regresión lineal (OLS) normalizada sobre los últimos n_bars cierres."""
    s = close_series.dropna().tail(n_bars)
    if len(s) < max(5, n_bars // 2):
        return np.nan
    y = s.to_numpy(dtype=float)
    x = np.arange(len(y), dtype=float)
    slope, _ = np.polyfit(x, y, 1)
    base = y[0] if y[0] != 0 else np.nan
    return float(slope / base) if pd.notna(base) else np.nan


def up_down_volume_ratio(df_ohlcv, n_bars=25):
    """Proporción de volumen en días al alza vs días a la baja en las últimas n_bars ruedas."""
    sub = df_ohlcv[["Close", "Volume"]].dropna().tail(n_bars + 1)
    if len(sub) < 2:
        return np.nan
    diff = sub["Close"].diff().iloc[1:]
    vol = sub["Volume"].iloc[1:]
    up_vol = float(vol[diff > 0].sum())
    down_vol = float(vol[diff < 0].sum())
    if down_vol <= 0:
        return np.nan
    return round(up_vol / down_vol, 4)


def get_asset_meta(ticker):
    """Obtiene nombre, sector, industria y marketCap con respaldo seguro."""
    try:
        info = yf.Ticker(ticker).info
        name = info.get("shortName") or info.get("longName") or ticker
        sector = info.get("sector") or "Unknown"
        industry = info.get("industry") or "Unknown"
        market_cap = info.get("marketCap")
        market_cap = float(market_cap) if market_cap is not None else np.nan
        return name, sector, industry, market_cap
    except Exception:
        return ticker, "Unknown", "Unknown", np.nan


def scan_market():
    t0 = time.time()
    print("🚀 Iniciando escáner rápido (Precio >= $10 | Vol 20d >= 300k antes de .info)...")
    tickers = get_tickers()
    if not tickers:
        return

    data = download_in_batches(tickers, batch_size=500, pause_sec=1)
    if data.empty or not isinstance(data.columns, pd.MultiIndex):
        print("❌ No se obtuvieron datos históricos válidos.")
        return

    available = set(data.columns.get_level_values(0).unique())
    if "SPY" not in available:
        print("❌ Falta SPY en los datos descargados.")
        return

    spy_ret_80 = calc_period_return(data["SPY"]["Close"], lookback=80)
    if spy_ret_80 is None:
        print("❌ Historial insuficiente de SPY.")
        return

    etf_ret_80 = {}
    for etf in set(INDUSTRY_OR_SECTOR_ETF_MAP.values()):
        if etf in available:
            etf_ret_80[etf] = calc_period_return(data[etf]["Close"], lookback=80)

    candidates_pre = []

    # 1. Embudo rápido previo: precio, volumen 20d y Trend Template (sin llamadas .info)
    for ticker in tickers:
        if (
            ticker == "SPY"
            or ticker in INDUSTRY_OR_SECTOR_ETF_MAP.values()
            or ticker not in available
        ):
            continue

        try:
            df_s = data[ticker].dropna(subset=["Close", "Volume"])
            close = df_s["Close"]
            volume = df_s["Volume"]

            if len(close) < 200:
                continue

            price = float(close.iloc[-1])
            avg_vol_20d = float(volume.tail(20).mean())

            # Filtros rápidos de liquidez y precio antes de cualquier cálculo pesado
            if price < 10.0 or avg_vol_20d < 300_000:
                continue

            sma200 = float(close.rolling(200).mean().iloc[-1])
            sma150 = float(close.rolling(150).mean().iloc[-1])
            sma50 = float(close.rolling(50).mean().iloc[-1])

            if any(pd.isna(x) for x in (sma200, sma150, sma50)):
                continue

            if len(close) > 230:
                sma200_30d_ago = float(close.iloc[:-30].rolling(200).mean().iloc[-1])
                sma200_slope = (
                    (sma200 / sma200_30d_ago - 1.0) if sma200_30d_ago > 0 else 0.0
                )
                trend_sma200 = sma200 > sma200_30d_ago
            else:
                sma200_slope = 0.0
                trend_sma200 = False

            w52 = close.tail(252)
            high_52w = float(w52.max())
            low_52w = float(w52.min())

            vs_low52 = (price - low_52w) / low_52w if low_52w > 0 else 0.0
            vs_high52 = (high_52w - price) / high_52w if high_52w > 0 else 1.0

            cond_trend = (
                (price > sma150)
                and (price > sma200)
                and (sma150 > sma200)
                and trend_sma200
                and (sma50 > sma150)
                and (price > sma50)
                and (vs_low52 >= 0.30)
                and (vs_high52 <= 0.25)
            )

            if not cond_trend:
                continue

            candidates_pre.append(
                (
                    ticker,
                    df_s,
                    close,
                    volume,
                    price,
                    avg_vol_20d,
                    sma200,
                    sma150,
                    sma50,
                    sma200_slope,
                    trend_sma200,
                    high_52w,
                    low_52w,
                    vs_low52,
                    vs_high52,
                )
            )
        except Exception:
            continue

    print(f"🎯 Candidatas tras filtros rápidos: {len(candidates_pre)} (consultando .info solo para ellas)")

    rows = []
    for item in candidates_pre:
        (
            ticker,
            df_s,
            close,
            volume,
            price,
            avg_vol_20d,
            sma200,
            sma150,
            sma50,
            sma200_slope,
            trend_sma200,
            high_52w,
            low_52w,
            vs_low52,
            vs_high52,
        ) = item

        try:
            last_vol = float(volume.iloc[-1])
            avg_vol_50d = float(volume.tail(50).mean())
            avg_vol_5d = float(volume.tail(5).mean())

            rvol_20d = (last_vol / avg_vol_20d) if avg_vol_20d > 0 else np.nan
            dry_up_5d_vs_50d = (
                (avg_vol_5d / avg_vol_50d) if avg_vol_50d > 0 else np.nan
            )
            min_vol_10d_vs_50d = (
                (float(volume.tail(10).min()) / avg_vol_50d)
                if avg_vol_50d > 0
                else np.nan
            )

            stock_ret_80 = calc_period_return(close, lookback=80)
            rs_vs_spy = (
                (stock_ret_80 - spy_ret_80) if stock_ret_80 is not None else 0.0
            )

            name, sector, industry, market_cap = get_asset_meta(ticker)
            ref_etf = INDUSTRY_OR_SECTOR_ETF_MAP.get(
                industry, INDUSTRY_OR_SECTOR_ETF_MAP.get(sector, "SPY")
            )
            ref_ret = etf_ret_80.get(ref_etf, spy_ret_80)
            rs_vs_sector = (
                (stock_ret_80 - ref_ret)
                if (stock_ret_80 is not None and ref_ret is not None)
                else rs_vs_spy
            )

            stock_vs_ind_etf_rel = (
                round(stock_ret_80 - ref_ret, 4)
                if (stock_ret_80 is not None and ref_ret is not None)
                else np.nan
            )
            ols_5w = ols_slope_normalized(close, n_bars=25)
            ols_30w = ols_slope_normalized(close, n_bars=150)
            up_down_5w = up_down_volume_ratio(df_s, n_bars=25)
            prox_sma50 = round((price - sma50) / sma50, 4) if sma50 > 0 else np.nan

            trend_checks = [
                price > sma150,
                price > sma200,
                sma150 > sma200,
                trend_sma200,
                sma50 > sma150,
                price > sma50,
                vs_low52 >= 0.30,
                vs_high52 <= 0.25,
            ]
            score_30_trend = round(
                (sum(1 for c in trend_checks if c) / len(trend_checks)) * 30.0, 1
            )
            rs_spy_part = max(0.0, min(12.0, (rs_vs_spy + 0.10) * 60.0))
            high_prox_part = max(
                0.0, min(8.0, (1.0 - min(vs_high52, 0.25) / 0.25) * 8.0)
            )
            score_20_rs = round(rs_spy_part + high_prox_part, 1)

            liq_part = 8.0 if avg_vol_20d >= 300_000 else 0.0
            dry_up_val = dry_up_5d_vs_50d if pd.notna(dry_up_5d_vs_50d) else 1.0
            rvol_val = rvol_20d if pd.notna(rvol_20d) else 1.0
            dry_up_part = max(0.0, min(7.0, (1.15 - dry_up_val) * 14.0))
            rvol_part = max(0.0, min(5.0, rvol_val * 2.5))
            score_20_vol = round(liq_part + dry_up_part + rvol_part, 1)

            score_15_sector = round(
                max(0.0, min(15.0, (rs_vs_sector + 0.10) * 75.0)), 1
            )
            python_score_85 = round(
                score_30_trend + score_20_rs + score_20_vol + score_15_sector, 1
            )

            rows.append(
                {
                    "Ticker": ticker,
                    "Nombre": name,
                    "Sector": sector,
                    "Industria": industry,
                    "ETF Referencia": ref_etf,
                    "Precio Actual": round(price, 2),
                    "Vol Promedio 20d": int(round(avg_vol_20d)),
                    "RVOL (Hoy/20d)": round(rvol_20d, 2) if pd.notna(rvol_20d) else "",
                    "Dry-Up (5d/50d)": round(dry_up_5d_vs_50d, 2)
                    if pd.notna(dry_up_5d_vs_50d)
                    else "",
                    "Min Vol Ratio (10d/50d)": round(min_vol_10d_vs_50d, 2)
                    if pd.notna(min_vol_10d_vs_50d)
                    else "",
                    "SMA 200": round(sma200, 2),
                    "SMA 150": round(sma150, 2),
                    "SMA 50": round(sma50, 2),
                    "SMA 200 Slope": round(sma200_slope, 4),
                    "High 52W": round(high_52w, 2),
                    "Low 52W": round(low_52w, 2),
                    "Vs Low 52": round(vs_low52, 4),
                    "Vs High 52": round(vs_high52, 4),
                    "RS vs SPY": round(rs_vs_spy, 4),
                    "RS vs Sector": round(rs_vs_sector, 4),
                    "Score 30 Trend": score_30_trend,
                    "Score 20 RS": score_20_rs,
                    "Score 20 Vol": score_20_vol,
                    "Score 15 Sector": score_15_sector,
                    "Python Score (85)": python_score_85,
                    "_market_cap": market_cap,
                    "Stock vs ETF Industria Rel": stock_vs_ind_etf_rel
                    if pd.notna(stock_vs_ind_etf_rel)
                    else "",
                    "Rank MktCap Industria": "",
                    "OLS Acel 5w": round(ols_5w, 6) if pd.notna(ols_5w) else "",
                    "OLS Acel 30w": round(ols_30w, 6) if pd.notna(ols_30w) else "",
                    "Vol Prom 5d": int(round(avg_vol_5d)),
                    "Vol Prom 50d": int(round(avg_vol_50d)),
                    "Ratio Up/Down Vol 5w": up_down_5w
                    if pd.notna(up_down_5w)
                    else "",
                    "Prox SMA 50": prox_sma50 if pd.notna(prox_sma50) else "",
                }
            )
        except Exception:
            continue

    df_out = pd.DataFrame(rows)
    output_file = "oportunidades_minervini_filtradas.csv"

    if not df_out.empty:
        valid_mc = df_out["_market_cap"].notna() & (df_out["Industria"] != "Unknown")
        ranks = pd.Series(index=df_out.index, dtype="object")
        ranks.loc[valid_mc] = (
            df_out.loc[valid_mc]
            .groupby("Industria")["_market_cap"]
            .rank(ascending=False, method="min")
            .astype(int)
        )
        df_out["Rank MktCap Industria"] = ranks.fillna("")
        df_out = df_out.drop(columns=["_market_cap"])
        df_out = df_out.sort_values(by="Python Score (85)", ascending=False)

        df_out.to_csv(output_file, index=False, encoding="utf-8-sig")
        print(
            f"✅ Guardadas {len(df_out)} candidatas en {output_file} ({time.time() - t0:.1f}s)"
        )
    else:
        print(f"⚠️ Ninguna acción cumplió los filtros en esta corrida ({time.time() - t0:.1f}s).")


if __name__ == "__main__":
    scan_market()
