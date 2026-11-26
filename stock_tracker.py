import time
import numpy as np
import pandas as pd
import requests
import yfinance as yf

# ==============================================================================
# 1. MAPEO DE REFERENCIA SECTORIAL Y DE INDUSTRIA A ETFS BENCHMARK
# ==============================================================================
# Asigna cada subgrupo industrial o sector a su ETF de referencia correspondiente.
# Si la industria no se encuentra en la lista, el script utilizará SPY como respaldo.
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


# ==============================================================================
# 2. FUNCIONES DE DESCARGA Y GESTIÓN DE DATOS HISTÓRICOS
# ==============================================================================
def get_tickers():
    """
    Descarga la lista actualizada de tickers de acciones de EE. UU. desde GitHub 
    y limpia los símbolos no estándar (warrants, acciones preferentes, etc.).
    """
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


def download_single_batch(batch, retries=2, retry_wait=12):
    """
    Descarga un conjunto o lote de símbolos mediante yfinance utilizando un período
    de 2 años para contar con un colchón suficiente de días bursátiles (>= 252 ruedas).
    Incluye un mecanismo de reintentos ante fallos temporales o límites de cuota (Rate Limit).
    """
    for attempt in range(retries + 1):
        try:
            df = yf.download(
                batch,
                period="2y",  # Período extendido a 2 años para asegurar la ventana de 52 semanas
                group_by="ticker",
                threads=False,
                progress=False,
            )
            if df is not None and not df.empty:
                return df
        except Exception as e:
            if attempt == retries:
                print(f"⚠️ Error definitivo en lote ({len(batch)} símbolos): {e}")

        if attempt < retries:
            time.sleep(retry_wait * (attempt + 1))

    return pd.DataFrame()


def download_priority_benchmarks():
    """
    Descarga de forma prioritaria y aislada el índice SPY y los ETFs de referencia.
    Garantiza que las métricas de Fuerza Relativa tengan benchmarks válidos desde el inicio.
    """
    ref_etfs = sorted(set(INDUSTRY_OR_SECTOR_ETF_MAP.values()))
    benchmarks = sorted(set(["SPY"] + ref_etfs))
    print(f"📌 Descargando benchmarks prioritarios ({len(benchmarks)} símbolos: SPY + ETFs)...")

    df_bench = download_single_batch(benchmarks, retries=3, retry_wait=10)
    if df_bench.empty or not isinstance(df_bench.columns, pd.MultiIndex):
        print("🔄 Reintentando descarga individual de SPY...")
        df_spy = download_single_batch(["SPY"], retries=3, retry_wait=12)
        if not df_spy.empty:
            return df_spy
        return pd.DataFrame()

    available = set(df_bench.columns.get_level_values(0).unique())
    if "SPY" not in available:
        print("🔄 SPY no vino en el lote de benchmarks; reintentando solo SPY...")
        df_spy = download_single_batch(["SPY"], retries=3, retry_wait=12)
        if not df_spy.empty:
            combined = pd.concat([df_bench, df_spy], axis=1)
            return combined.loc[:, ~combined.columns.duplicated()]

    return df_bench


def download_in_batches(tickers, batch_size=200, pause_sec=4):
    """
    Divide y procesa la descarga del mercado general en lotes moderados para evitar 
    bloqueos por peticiones masivas, concatenando los datos descargados con los benchmarks.
    """
    df_bench = download_priority_benchmarks()
    if df_bench.empty:
        return pd.DataFrame()

    ref_etfs = set(INDUSTRY_OR_SECTOR_ETF_MAP.values())
    excluded = {"SPY"} | ref_etfs
    remaining_tickers = [t for t in sorted(set(tickers)) if t not in excluded]

    frames = [df_bench]
    total_batches = (len(remaining_tickers) + batch_size - 1) // batch_size

    for idx in range(0, len(remaining_tickers), batch_size):
        batch_num = idx // batch_size + 1
        batch = remaining_tickers[idx : idx + batch_size]
        print(f"📥 Lote {batch_num}/{total_batches} ({len(batch)} símbolos)...")
        df_batch = download_single_batch(batch, retries=2, retry_wait=12)
        if not df_batch.empty:
            frames.append(df_batch)

        if batch_num < total_batches and pause_sec > 0:
            time.sleep(pause_sec)

    combined = pd.concat(frames, axis=1)
    combined = combined.loc[:, ~combined.columns.duplicated()]
    return combined


# ==============================================================================
# 3. FUNCIONES DE CÁLCULO ESTADÍSTICO Y MÉTRICAS TÉCNICAS
# ==============================================================================
def calc_period_return(close_series, lookback=252):
    """
    Calcula el rendimiento porcentual de una serie de cierres en la ventana especificada.
    Por defecto utiliza 252 ruedas de negociación (1 año bursátil / 52 semanas).
    """
    s = close_series.dropna()
    if len(s) <= lookback:
        return None
    prev_val = float(s.iloc[-lookback - 1])
    curr_val = float(s.iloc[-1])
    if prev_val <= 0:
        return None
    return (curr_val - prev_val) / prev_val


def ols_slope_normalized(close_series, n_bars):
    """
    Calcula la pendiente de regresión lineal (OLS) normalizada sobre los últimos n_bars cierres.
    Divide la pendiente obtenida entre el primer precio del rango para obtener una tasa de cambio comparable.
    """
    s = close_series.dropna().tail(n_bars)
    if len(s) < max(5, n_bars // 2):
        return np.nan
    y = s.to_numpy(dtype=float)
    x = np.arange(len(y), dtype=float)
    slope, _ = np.polyfit(x, y, 1)
    base = y[0] if y[0] != 0 else np.nan
    return float(slope / base) if pd.notna(base) else np.nan


def up_down_volume_ratio(df_ohlcv, n_bars=25):
    """
    Calcula el ratio entre el volumen transado en días al alza vs. el volumen transado en días a la baja 
    en un horizonte de n_bars ruedas (~5 semanas).
    """
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
    """
    Consulta la información fundamental de la empresa (Yahoo Finance info) únicamente
    para los activos pre-filtrados para extraer Nombre, Sector, Industria y Capitalización Bursátil.
    """
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


# ==============================================================================
# 4. MODELO DE SCORING Y EVALUACIÓN DE NOTAS (MÁXIMO 75 PUNTOS TÓTAL)
# ==============================================================================
def calc_score_aceleracion(ar):
    """
    Sub-bloque A: Aceleración OLS de la SMA 200 (AR = Beta 5w / Beta 30w) (Máximo 20 Puntos).
    Puntúa el comportamiento de la aceleración según el tramo del indicador AR.
    """
    if pd.isna(ar):
        return 0.0
    if 1.00 <= ar <= 1.30:
        return 20.0  # Zona dulce / Aceleración fresca
    if 1.30 < ar <= 2.00:
        return 14.0  # Aceleración sólida y consolidada
    if ar > 2.00:
        return 8.0   # Movimiento sobreextendido / muy acelerado
    if 0.00 < ar < 1.00:
        return 8.0   # Aceleración lenta o incipiente
    return 0.0       # AR <= 0: Sin aceleración o retroceso


def calc_score_timing(ud_ratio, rvol_3d_20d, dist_sma50):
    """
    Sub-bloque B: Timing & Volumen (Máximo 25 Puntos Totales).
    - Up/Down Volume Ratio (Máx. 11 pts)
    - Aceleración Corta de Volumen RVOL 3d/20d (Máx. 4 pts)
    - Proximidad/Distancia a la SMA 50 (Máx. 10 pts)
    """
    # 1. Up/Down Volume Ratio (Máx 11 Puntos)
    if pd.isna(ud_ratio) or ud_ratio <= 1.00:
        score_ud = 0.0
    elif ud_ratio >= 2.00:
        score_ud = 11.0
    elif 1.50 <= ud_ratio < 2.00:
        score_ud = 8.0
    elif 1.20 <= ud_ratio < 1.50:
        score_ud = 5.0
    else:  # 1.00 < ud_ratio < 1.20
        score_ud = 2.0

    # 2. RVOL 3d/20d (Máx 4 Puntos)
    if pd.isna(rvol_3d_20d) or rvol_3d_20d < 0.80:
        score_rvol = 0.0
    elif 1.10 <= rvol_3d_20d <= 1.80:
        score_rvol = 4.0
    elif 0.80 <= rvol_3d_20d < 1.10:
        score_rvol = 2.0
    else:  # RVOL > 1.80
        score_rvol = 2.0 if (pd.notna(ud_ratio) and ud_ratio > 1.20) else 1.0

    # 3. Distancia a la SMA 50 (%) (Máx 10 Puntos)
    dist_pct = dist_sma50 * 100.0 if pd.notna(dist_sma50) else -999.0
    if 0.1 <= dist_pct <= 3.0:
        score_dist = 10.0  # Entrada de muy bajo riesgo (Low Risk Entry)
    elif 3.0 < dist_pct <= 6.0:
        score_dist = 7.0   # Buen timing de entrada
    elif 6.0 < dist_pct <= 10.0:
        score_dist = 3.0   # Extendido
    else:
        score_dist = 0.0   # Fuera de rango (debajo de la SMA50 o >10%)

    return round(score_ud + score_rvol + score_dist, 1)


def calc_score_rs(rs_vs_spy, rs_ind_val, rank_ind):
    """
    Sub-bloque C: Fuerza Relativa (Máximo 30 Puntos Totales).
    - Fuerza Relativa vs SPY (52 semanas) (Máx. 20 pts)
    - Ranking de Industria por Capitalización Bursátil (Máx. 10 pts)
    """
    # 1. RS vs SPY (52 semanas)
    if pd.isna(rs_vs_spy) or rs_vs_spy <= 0.00:
        score_spy = 0.0
    elif rs_vs_spy >= 0.75:
        score_spy = 20.0
    elif 0.50 <= rs_vs_spy < 0.75:
        score_spy = 15.0
    elif 0.25 <= rs_vs_spy < 0.50:
        score_spy = 10.0
    else:  # 0.00 < RS_vs_SPY < 0.25
        score_spy = 5.0

    # 2. Ranking de Industria por MktCap
    score_ind = 0.0
    if pd.notna(rank_ind) and rank_ind != "":
        try:
            r = int(rank_ind)
            if 1 <= r <= 3:
                score_ind = 10.0  # Liderazgo de industria (Top 1-3)
            elif 4 <= r <= 5:
                score_ind = 5.0   # Top 4-5
            else:
                score_ind = 0.0
        except ValueError:
            pass

    if score_ind == 0.0 and pd.notna(rs_ind_val):
        if rs_ind_val >= 115:
            score_ind = 10.0
        elif 105 <= rs_ind_val < 115:
            score_ind = 5.0
        else:
            score_ind = 0.0

    return round(score_spy + score_ind, 1)


# ==============================================================================
# 5. MOTOR PRINCIPAL DEL ESCÁNER Y PROCESAMIENTO DEL CSV
# ==============================================================================
def scan_market():
    t0 = time.time()
    print("🚀 Iniciando escáner (Benchmarks prioritarios + Filtros cuantitativos)...")
    tickers = get_tickers()
    if not tickers:
        return

    # Descarga por lotes del mercado completo
    data = download_in_batches(tickers, batch_size=200, pause_sec=4)
    if data.empty or not isinstance(data.columns, pd.MultiIndex):
        print("❌ No se obtuvieron datos históricos válidos.")
        return

    available = set(data.columns.get_level_values(0).unique())
    if "SPY" not in available:
        print("❌ Falta SPY en los datos descargados.")
        return

    # Retornos base de SPY a 52 semanas y a 80 ruedas
    spy_ret_52w = calc_period_return(data["SPY"]["Close"], lookback=252)
    spy_ret_80 = calc_period_return(data["SPY"]["Close"], lookback=80)
    if spy_ret_52w is None:
        print("❌ Historial insuficiente de SPY.")
        return

    # Retornos de referencia para los ETFs de sector/industria
    etf_ret_80 = {}
    for etf in set(INDUSTRY_OR_SECTOR_ETF_MAP.values()):
        if etf in available:
            etf_ret_80[etf] = calc_period_return(data[etf]["Close"], lookback=80)

    candidates_pre = []

    # --------------------------------------------------------------------------
    # PASO 5A: EVALUACIÓN DE FILTROS BASE Y FILTROS OBLIGATORIOS BINARIOS
    # --------------------------------------------------------------------------
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

            # Filtros iniciales de liquidez y precio
            if price < 10.0 or avg_vol_20d < 300_000:
                continue

            # FILTRO OBLIGATORIO 1: Fuerza Relativa Positiva (RS 52w vs SPY > 0)
            stock_ret_52w = calc_period_return(close, lookback=252)
            if stock_ret_52w is None:
                continue
            rs_vs_spy_52w = ((1.0 + stock_ret_52w) / (1.0 + spy_ret_52w)) - 1.0
            if rs_vs_spy_52w <= 0.0:
                continue

            # FILTRO OBLIGATORIO 2: Volumen Reciente Promedio (RVOL 3d/20d > 0.80)
            avg_vol_3d = float(volume.tail(3).mean())
            rvol_3d_20d = (avg_vol_3d / avg_vol_20d) if avg_vol_20d > 0 else 0.0
            if rvol_3d_20d <= 0.80:
                continue

            # FILTRO OBLIGATORIO 3: Variable AR (Ratio Beta 5w / Beta 30w) > 0.0
            ols_5w = ols_slope_normalized(close, n_bars=25)
            ols_30w = ols_slope_normalized(close, n_bars=150)
            if pd.isna(ols_5w) or pd.isna(ols_30w) or ols_30w <= 0:
                ar_val = np.nan
            else:
                ar_val = ols_5w / ols_30w

            if pd.isna(ar_val) or ar_val <= 0.0:
                continue

            # ESTRUCTURA TÉCNICA: Trend Template de Mark Minervini (Etapa 2)
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
                    rvol_3d_20d,
                    sma200,
                    sma150,
                    sma50,
                    sma200_slope,
                    high_52w,
                    low_52w,
                    vs_low52,
                    vs_high52,
                    rs_vs_spy_52w,
                    ols_5w,
                    ols_30w,
                    ar_val,
                )
            )
        except Exception:
            continue

    print(
        f"🎯 Candidatas tras filtros rápidos: {len(candidates_pre)} (consultando .info solo para ellas)"
    )

    # --------------------------------------------------------------------------
    # PASO 5B: EXTRACCIÓN DE METADATOS Y ENSAMBLADO DE VARIABLES
    # --------------------------------------------------------------------------
    rows = []
    for item in candidates_pre:
        (
            ticker,
            df_s,
            close,
            volume,
            price,
            avg_vol_20d,
            rvol_3d_20d,
            sma200,
            sma150,
            sma50,
            sma200_slope,
            high_52w,
            low_52w,
            vs_low52,
            vs_high52,
            rs_vs_spy_52w,
            ols_5w,
            ols_30w,
            ar_val,
        ) = item

        try:
            avg_vol_50d = float(volume.tail(50).mean())
            avg_vol_5d = float(volume.tail(5).mean())

            dry_up_5d_vs_50d = (
                (avg_vol_5d / avg_vol_50d) if avg_vol_50d > 0 else np.nan
            )
            min_vol_10d_vs_50d = (
                (float(volume.tail(10).min()) / avg_vol_50d)
                if avg_vol_50d > 0
                else np.nan
            )

            stock_ret_80 = calc_period_return(close, lookback=80)

            # Obtención de metadatos del activo (Yahoo Finance .info)
            name, sector, industry, market_cap = get_asset_meta(ticker)
            ref_etf = INDUSTRY_OR_SECTOR_ETF_MAP.get(
                industry, INDUSTRY_OR_SECTOR_ETF_MAP.get(sector, "SPY")
            )
            ref_ret = etf_ret_80.get(ref_etf, spy_ret_80)
            rs_vs_sector = (
                (stock_ret_80 - ref_ret)
                if (stock_ret_80 is not None and ref_ret is not None)
                else rs_vs_spy_52w
            )

            up_down_5w = up_down_volume_ratio(df_s, n_bars=25)
            prox_sma50 = round((price - sma50) / sma50, 4) if sma50 > 0 else np.nan

            # Estructura del diccionario respetando exactamente las posiciones de las columnas
            rows.append(
                {
                    "Ticker": ticker,
                    "Nombre": name,
                    "Sector": sector,
                    "Industria": industry,
                    "ETF Referencia": ref_etf,
                    "Precio Actual": round(price, 2),
                    "Vol Promedio 20d": int(round(avg_vol_20d)),
                    "RVOL (3d/20d)": round(rvol_3d_20d, 2),  # Variable reemplazada manteniendo su lugar ordinal
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
                    "RS vs SPY": round(rs_vs_spy_52w, 4),
                    "RS vs Sector": round(rs_vs_sector, 4),
                    "Rank MktCap Industria": "",  # Se calculará tras el groupby por Industria
                    "OLS Acel 5w": round(ols_5w, 6) if pd.notna(ols_5w) else "",
                    "OLS Acel 30w": round(ols_30w, 6) if pd.notna(ols_30w) else "",
                    "Vol Prom 5d": int(round(avg_vol_5d)),
                    "Vol Prom 50d": int(round(avg_vol_50d)),
                    "Ratio Up/Down Vol 5w": up_down_5w
                    if pd.notna(up_down_5w)
                    else "",
                    "Prox SMA 50": prox_sma50 if pd.notna(prox_sma50) else "",
                    "AR": round(ar_val, 4) if pd.notna(ar_val) else "",  # Nueva variable ubicada al final
                    # Campos internos auxiliares para el cálculo de notas
                    "_market_cap": market_cap,
                    "_ud_ratio": up_down_5w,
                    "_rvol_3d_20d": rvol_3d_20d,
                    "_dist_sma50": prox_sma50,
                    "_rs_vs_spy": rs_vs_spy_52w,
                    "_ar": ar_val,
                }
            )
        except Exception:
            continue

    df_out = pd.DataFrame(rows)
    output_file = "oportunidades_minervini_filtradas.csv"

    # --------------------------------------------------------------------------
    # PASO 5C: CÁLCULO DE RANKING DE INDUSTRIA Y SCORES RE-CALIBRADOS
    # --------------------------------------------------------------------------
    if not df_out.empty:
        # Ranking de empresas por MarketCap agrupadas por Industria
        valid_mc = df_out["_market_cap"].notna() & (df_out["Industria"] != "Unknown")
        ranks = pd.Series(index=df_out.index, dtype="object")
        ranks.loc[valid_mc] = (
            df_out.loc[valid_mc]
            .groupby("Industria")["_market_cap"]
            .rank(ascending=False, method="min")
            .astype(int)
        )
        df_out["Rank MktCap Industria"] = ranks.fillna("")

        # Evaluación individual de las 3 sub-notas y el Score Total
        scores_acel = []
        scores_timing = []
        scores_rs = []
        scores_total = []

        for _, row in df_out.iterrows():
            sc_acel = calc_score_aceleracion(row["_ar"])
            sc_tim = calc_score_timing(
                row["_ud_ratio"], row["_rvol_3d_20d"], row["_dist_sma50"]
            )
            sc_rs = calc_score_rs(
                row["_rs_vs_spy"], np.nan, row["Rank MktCap Industria"]
            )
            sc_tot = round(sc_acel + sc_tim + sc_rs, 1)

            scores_acel.append(sc_acel)
            scores_timing.append(sc_tim)
            scores_rs.append(sc_rs)
            scores_total.append(sc_tot)

        # Incorporación de las columnas de puntuación
        df_out["Score Aceleracion"] = scores_acel
        df_out["Score Timing"] = scores_timing
        df_out["Score Relative Strength"] = scores_rs
        df_out["Score Total"] = scores_total

        # Limpieza de columnas auxiliares de cálculo interno
        temp_cols = [
            "_market_cap",
            "_ud_ratio",
            "_rvol_3d_20d",
            "_dist_sma50",
            "_rs_vs_spy",
            "_ar",
        ]
        df_out = df_out.drop(columns=temp_cols)

        # Ordenamiento final descendente por el Score Total
        df_out = df_out.sort_values(by="Score Total", ascending=False)

        # Exportación del CSV filtrado y puntuado
        df_out.to_csv(output_file, index=False, encoding="utf-8-sig")
        print(
            f"✅ Guardadas {len(df_out)} candidatas en {output_file} ({time.time() - t0:.1f}s)"
        )
    else:
        print(
            f"⚠️ Ninguna acción cumplió los filtros en esta corrida ({time.time() - t0:.1f}s)."
        )


if __name__ == "__main__":
    scan_market()
