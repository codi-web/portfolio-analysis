# [source: 1]
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import scipy.optimize as optimize
import time
from typing import List, Optional
import requests
from requests.exceptions import RequestException

# --- Configuración General ---
warnings.filterwarnings("ignore")

# Configuración de la página: Título más descriptivo y layout ancho
st.set_page_config(
    page_title="Optimizador y Analizador de Portafolios",
    page_icon="assets/icons.png", # Añadir un icono
    layout="wide"
)

# Establecer un tema de Seaborn para mejorar la estética de los gráficos
sns.set_theme(style="whitegrid")
# Definir una paleta de colores para consistencia (ej: tab10)
COLOR_PALETTE = plt.cm.get_cmap('tab10').colors

# --- Funciones de Ayuda ---

@st.cache_data # Mantener caché de datos
def download_data_with_retry(symbols: List[str], start_date: datetime, end_date: datetime, max_retries: int = 3) -> Optional[pd.DataFrame]:
    """Descarga datos de Yahoo Finance con reintentos y manejo de errores mejorado."""
 
    chunk_size = 5
    all_data = []
    symbols_failed = []
    processed_symbols = set() # Para evitar procesar duplicados si fallan y están en otro chunk

    # Mostrar barra de progreso en el área principal si es posible, o en la barra lateral
    try:
        progress_bar = st.progress(0)
        progress_text = st.empty() # Para mostrar texto debajo de la barra
    except st.errors.StreamlitAPIException: # Si se llama desde un sitio sin st.progress (raro)
        progress_bar = None
        progress_text = None

    total_symbols_to_process = len(symbols)
    symbols_processed_count = 0

    for i, chunk_idx in enumerate(range(0, len(symbols), chunk_size)):
        chunk = [s for s in symbols[chunk_idx:chunk_idx + chunk_size] if s not in processed_symbols]
        if not chunk: continue # Saltar si todos los símbolos del chunk ya fueron procesados (fallaron)

        attempt = 0
        success = False
        data = None # Inicializar data a None
        while attempt < max_retries and not success:
            try:
                # Descargar datos para el chunk
                data = yf.download(chunk,
                                 start=start_date,
                                 end=end_date,
                                 progress=False,
                                 auto_adjust=False, # Mantener consistencia
                                 timeout=15) # Aumentar timeout ligeramente

                if data is not None and not data.empty:
                    # Verificar si se devolvió un MultiIndex (varios símbolos) o Index simple (un símbolo)
                    if isinstance(data.columns, pd.MultiIndex):
                        close_data = data.get('Close')
                    # Si solo se descargó un símbolo, 'Close' no existe como nivel superior
                    elif 'Close' in data.columns:
                         close_data = data[['Close']].rename(columns={'Close': chunk[0]}) # Crear DataFrame y renombrar
                    elif not data.empty and len(chunk) == 1: # Si no tiene 'Close' pero no está vacío y era un solo símbolo
                         close_data = data[[data.columns[0]]].rename(columns={data.columns[0]: chunk[0]}) # Tomar la primera columna como precio
                         st.warning(f"Se usó la columna '{data.columns[0]}' como precio para {chunk[0]} (no se encontró 'Close').")
                    else:
                         close_data = None # No se encontró 'Close'

                    if close_data is not None and not close_data.empty:
                         # Identificar símbolos que sí se descargaron en este chunk
                         downloaded_symbols = [s for s in chunk if s in close_data.columns]
                         failed_in_chunk = [s for s in chunk if s not in downloaded_symbols]

                         if downloaded_symbols:
                              # Alinear los datos descargados al índice completo de fechas (días hábiles)
                              full_date_range = pd.date_range(start=start_date, end=end_date, freq='B')
                              # Reindexar y forward-fill valores faltantes
                              close_data_reindexed = close_data[downloaded_symbols].reindex(full_date_range).ffill()
                              all_data.append(close_data_reindexed)
                              processed_symbols.update(downloaded_symbols)
                              symbols_processed_count += len(downloaded_symbols)


                         if failed_in_chunk:
                              symbols_failed.extend(failed_in_chunk)
                              processed_symbols.update(failed_in_chunk) # Marcar como procesado (fallido)
                              symbols_processed_count += len(failed_in_chunk)


                         success = True # Se procesó el chunk (con o sin fallos parciales)
                    else:
                         st.warning(f"No se encontró columna 'Close' válida para {chunk} en intento {attempt+1}")
                         symbols_failed.extend(chunk)
                         processed_symbols.update(chunk)
                         symbols_processed_count += len(chunk)
                         break # Salir del while si no hay 'Close' válido

                else: # data is None or data.empty
                    st.warning(f"No se recibieron datos para {chunk} en intento {attempt+1}")
                    symbols_failed.extend(chunk)
                    processed_symbols.update(chunk)
                    symbols_processed_count += len(chunk)
                    # No salir del while aquí, intentar de nuevo si quedan reintentos
                    attempt += 1
                    time.sleep(attempt * 0.5) # Espera corta antes de reintentar


            except Exception as e:
                st.warning(f"Error descargando {chunk} (intento {attempt+1}/{max_retries}): {type(e).__name__}")
                attempt += 1
                time.sleep(attempt * 1) # Incrementar espera en fallos
                if attempt == max_retries:
                    st.error(f"Fallo persistente al descargar {chunk} después de {max_retries} intentos.")
                    symbols_failed.extend(chunk)
                    processed_symbols.update(chunk)
                    symbols_processed_count += len(chunk)


        # Actualizar barra de progreso después de cada chunk
        if progress_bar:
             progress_percentage = min(1.0, symbols_processed_count / total_symbols_to_process)
             progress_bar.progress(progress_percentage)
        if progress_text:
             progress_text.text(f"Descargando... {symbols_processed_count}/{total_symbols_to_process} símbolos procesados.")


    # Limpiar barra y texto de progreso
    if progress_bar: progress_bar.empty()
    if progress_text: progress_text.empty()

    if not all_data:
        st.error("No se pudieron descargar datos válidos para ningún activo.")
        return None

    # Concatenar y asegurar que no haya columnas duplicadas (manteniendo la primera ocurrencia)
    # Puede haber duplicados si un símbolo estaba en múltiples chunks (aunque el set lo previene)
    # O si yfinance devuelve el mismo símbolo bajo diferentes casos (ej: 'aapl', 'AAPL')
    try:
        combined_data = pd.concat(all_data, axis=1)
        combined_data = combined_data.loc[:, ~combined_data.columns.duplicated(keep='first')]
    except Exception as e:
         st.error(f"Error al concatenar datos: {e}")
         # Intentar concatenar solo los que no den problemas (simplificado)
         valid_dfs = []
         for df in all_data:
             if isinstance(df, pd.DataFrame) and not df.empty:
                 valid_dfs.append(df)
         if not valid_dfs: return None
         try:
             combined_data = pd.concat(valid_dfs, axis=1)
             combined_data = combined_data.loc[:, ~combined_data.columns.duplicated(keep='first')]
         except Exception as final_e:
              st.error(f"Error final al concatenar datos: {final_e}")
              return None


    # Obtener la lista final de símbolos descargados exitosamente
    final_downloaded_symbols = combined_data.columns.tolist()

    # Símbolos que estaban en la lista original pero no se descargaron
    permanently_failed = list(set(symbols) - set(final_downloaded_symbols))
    if permanently_failed:
         st.warning(f"No se pudieron descargar o procesar datos finales para: {', '.join(permanently_failed)}. Se excluirán del análisis.")


    # Eliminar filas donde *todos* los valores son NaN (usualmente al principio/fin)
    combined_data.dropna(axis=0, how='all', inplace=True)

    # Rellenar NaNs restantes (ej: días festivos no alineados) con el último valor válido
    combined_data.ffill(inplace=True)
    # Rellenar NaNs iniciales restantes con el primer valor válido (bfill)
    combined_data.bfill(inplace=True)
    # Volver a eliminar filas que puedan seguir siendo completamente NaN si bfill no encontró nada
    combined_data.dropna(axis=0, how='all', inplace=True)

    if combined_data.empty:
         st.error("Los datos resultantes están vacíos después del procesamiento.")
         return None

    # Asegurarse de devolver solo los símbolos solicitados que sí se descargaron
    final_columns = [s for s in symbols if s in combined_data.columns]
    return combined_data[final_columns]


def verify_symbol(symbol: str) -> bool:
    """Verifica si un símbolo es válido en Yahoo Finance (más robusto)"""
    if not symbol or not isinstance(symbol, str):
        return False
    try:
        ticker = yf.Ticker(symbol)
        # Intenta obtener 'history' que es más fiable para verificar existencia
        hist = ticker.history(period="5d", interval="1d", raise_errors=True) # raise_errors para capturar fallos
        if hist.empty:
             # A veces history está vacío pero info funciona (ej: índices sin histórico reciente?)
             info = ticker.info
             return bool(info and info.get('quoteType')) # Necesita info Y un quoteType
        return True # Si history no está vacío, es válido
    except (RequestException, KeyError, IndexError, ValueError, TypeError): # Errores comunes de yfinance o red
        return False
    except Exception as e: # Captura cualquier otro error inesperado
        # st.warning(f"Error verificando {symbol}: {e}") # Descomentar para depuración
        return False

def sortino_ratio(returns, risk_free_rate=0.0232):
    """Calcula el ratio de Sortino."""
    # Asegurarse que 'returns' es una Serie de pandas
    if not isinstance(returns, pd.Series):
        returns = pd.Series(returns)

    negative_returns = returns[returns < 0].copy() # Usar .copy() para evitar SettingWithCopyWarning
    if negative_returns.empty:
        return 0.0 # Si no hay retornos negativos, el riesgo a la baja es 0, ratio infinito o indefinido, devolver 0

    # Calcular desviación estándar de retornos negativos diarios
    downside_std_dev_daily = negative_returns.std()

    if downside_std_dev_daily == 0 or pd.isna(downside_std_dev_daily):
        return 0.0 # Si la desviación es 0, también devolver 0

    # Anualizar la desviación estándar a la baja
    downside_deviation_annualized = downside_std_dev_daily * np.sqrt(252)

    # Calcular el retorno esperado anualizado
    expected_return_annualized = returns.mean() * 252

    # Calcular Sortino Ratio
    sortino = (expected_return_annualized - risk_free_rate) / downside_deviation_annualized
    return sortino if pd.notna(sortino) else 0.0 # Devolver 0 si el resultado es NaN


def sharpe_ratio(returns, risk_free_rate=0.0232):
    """Calcula el ratio de Sharpe."""
    # Asegurarse que 'returns' es una Serie de pandas
    if not isinstance(returns, pd.Series):
        returns = pd.Series(returns)

    # Calcular volatilidad (desviación estándar anualizada)
    volatility = returns.std() * np.sqrt(252)

    if volatility == 0 or pd.isna(volatility):
        return 0.0 # Evitar división por cero o NaN

    # Calcular retorno esperado anualizado
    expected_return = returns.mean() * 252

    # Calcular Sharpe Ratio
    sharpe = (expected_return - risk_free_rate) / volatility
    return sharpe if pd.notna(sharpe) else 0.0 # Devolver 0 si el resultado es NaN

def calculate_cvar(returns, confidence_level=0.95):
    """Calcula el Conditional Value at Risk (CVaR) diario."""
     # Asegurarse que 'returns' es una Serie de pandas
    if not isinstance(returns, pd.Series):
        returns = pd.Series(returns)

    if returns.empty or returns.isna().all(): # Si está vacío o todo NaN
        return np.nan

    # Calcular VaR (Percentil)
    var = np.percentile(returns.dropna(), (1 - confidence_level) * 100)

    # Filtrar retornos menores o iguales al VaR
    tail_returns = returns[returns <= var].dropna()

    if tail_returns.empty:
        # Si no hay retornos en la cola (poco probable con datos continuos, pero posible)
        # Devolver el VaR como la peor pérdida observada en ese nivel
        return var if pd.notna(var) else np.nan

    # Calcular CVaR (media de los retornos en la cola)
    cvar = tail_returns.mean()
    return cvar if pd.notna(cvar) else np.nan

def optimize_portfolio(log_returns, cov_matrix, num_assets):
    """Optimiza el portafolio para mínima volatilidad (ejemplo)."""
    # (Código existente) - Asegurar que la función de volatilidad maneje posibles NaNs
    def calculate_portfolio_volatility(weights, cov_matrix):
        """Calcula la volatilidad anualizada del portafolio."""
        # Volatilidad = sqrt(w^T * Cov * w)
        portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
        # El cov_matrix ya está anualizado, por lo que esto da varianza anualizada
        # La volatilidad es la raíz cuadrada
        portfolio_volatility = np.sqrt(portfolio_variance)
        return portfolio_volatility if pd.notna(portfolio_volatility) else np.inf # Devolver infinito si algo falla

    # Restricción: la suma de pesos debe ser 1
    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
    # Límites: cada peso entre 0 y 1
    bounds = tuple((0, 1) for _ in range(num_assets))
    # Punto inicial: pesos iguales
    initial_weights = num_assets * [1. / num_assets, ]

    try:
         result = optimize.minimize(
            calculate_portfolio_volatility,
            initial_weights,
            args=(cov_matrix,),
            method='SLSQP', # Método común para este tipo de optimización
            bounds=bounds,
            constraints=constraints
        )
         if result.success:
              return result.x # Devuelve los pesos óptimos
         else:
              st.warning(f"La optimización de mínima volatilidad no convergió: {result.message}")
              # Devolver pesos iniciales como fallback o manejar de otra forma
              return np.array(initial_weights)
    except ValueError as e:
         st.error(f"Error durante la optimización (posiblemente por NaNs en matriz covarianza): {e}")
         # Devolver pesos iniciales como fallback
         return np.array(initial_weights)

# --- FUNCIÓN CORREGIDA ---
def monte_carlo_simulation(log_returns, cov_matrix, num_portfolios, num_assets, risk_free_rate=0.0232):
    """Realiza la simulación de Monte Carlo."""
    # Inicializar array solo para retornos, volatilidad y Sharpe
    results = np.zeros((3, num_portfolios)) # Cambio: Solo 3 filas
    # Inicializar una lista para almacenar los arrays de pesos
    all_weights = [] # Cambio: Lista separada para pesos

    mean_returns_annualized = log_returns.mean() * 252
    # cov_matrix ya debería estar anualizada (se multiplica por 252 fuera)

    for i in range(num_portfolios):
        # Generar pesos aleatorios y normalizarlos
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)

        # Calcular Retorno Anualizado del Portafolio
        portfolio_return_annualized = np.sum(weights * mean_returns_annualized)

        # Calcular Volatilidad Anualizada del Portafolio
        portfolio_variance_annualized = np.dot(weights.T, np.dot(cov_matrix, weights))
        # Añadir manejo defensivo por si la varianza es negativa (muy raro, pero posible por errores numéricos)
        if portfolio_variance_annualized < 0:
             portfolio_variance_annualized = 0
        portfolio_stddev_annualized = np.sqrt(portfolio_variance_annualized)

        # Calcular Ratio de Sharpe
        sharpe = 0.0
        if portfolio_stddev_annualized > 1e-8: # Evitar división por volatilidad casi cero
             sharpe = (portfolio_return_annualized - risk_free_rate) / portfolio_stddev_annualized
        elif portfolio_return_annualized > risk_free_rate:
             sharpe = np.inf # Retorno positivo sin riesgo -> Sharpe infinito

        # Guardar resultados anualizados
        results[0, i] = portfolio_return_annualized
        results[1, i] = portfolio_stddev_annualized
        results[2, i] = sharpe

        # Guardar los pesos en la lista separada
        all_weights.append(weights) # Cambio: Añadir a la lista

    # Devolver las métricas y la lista de pesos
    return results[0], results[1], results[2], all_weights # Cambio: Devolver la lista de pesos

# --- Interfaz Principal de Streamlit ---
def main():
        # Creación de columnas con proporción 1:9
    col1, col2 = st.columns([0.1, 0.9])  # [1][3][4]

    with col1:
        # Usar ruta relativa para mejor portabilidad
        st.image("assets/icons.png", width=80)  # Asegúrate que el archivo se llame "icono.png"

    with col2:
        st.title("Optimizador y Analizador de Portafolios v1.1")
        st.markdown("Una herramienta interactiva para analizar y optimizar portafolios de inversión.")
    # --- Inicialización del Estado de Sesión ---
    # Usar claves descriptivas y valores por defecto razonables
    if 'portfolio_assets' not in st.session_state:
        # [source: 15], [source: 16], [source: 17], [source: 18]
        st.session_state.portfolio_assets = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META',  # Tech
            'JPM', 'BAC', 'GS', 'MS', 'V',            # Finance
            'JNJ', 'PG', 'KO', 'WMT', 'MCD'           # Consumer
        ]
    if 'simulation_count' not in st.session_state:
        # [source: 18]
        st.session_state.simulation_count = 20000 # Mantener valor ajustado
    if 'investment_amount' not in st.session_state:
        st.session_state.investment_amount = 10000
    if 'benchmark_symbol' not in st.session_state:
        # [source: 19]
        st.session_state.benchmark_symbol = "^GSPC" # Usar S&P 500 como default más común
    if 'analysis_start_date' not in st.session_state:
        st.session_state.analysis_start_date = datetime.now() - timedelta(days=5*365)
    if 'analysis_end_date' not in st.session_state:
        st.session_state.analysis_end_date = datetime.now()
    if 'risk_free_rate' not in st.session_state:
         st.session_state.risk_free_rate = 0.0232 # Tasa libre de riesgo (configurable)

    # --- Barra Lateral de Configuración (Sidebar) ---
    with st.sidebar:
        st.header("⚙️ Configuración del Análisis")
        st.divider() # Añadir separador visual

        # --- Sección de Gestión de Activos ---
        with st.expander("📈 Gestión de Activos", expanded=True):
            st.info(f"Actualmente analizando {len(st.session_state.portfolio_assets)} activos.")
            # Añadir Activo
            col1_add, col2_add = st.columns([3, 1])
            with col1_add:
                new_asset = st.text_input("Añadir símbolo de activo:", key="new_asset_input", placeholder="Ej: AAPL, MSFT").upper().strip() # .strip()
            with col2_add:
                st.write("") # Placeholder for vertical alignment
                st.write("") # Placeholder for vertical alignment
                # Botón con tooltip
                if st.button("➕ Añadir", key="add_asset_button", help="Verifica y agrega el símbolo a la lista"):
                    if new_asset:
                        if new_asset not in st.session_state.portfolio_assets:
                       
                            with st.spinner(f"Verificando {new_asset}..."):
                                if verify_symbol(new_asset):
                                    st.session_state.portfolio_assets.append(new_asset)
                                    st.success(f"✅ {new_asset} agregado.")
                                    time.sleep(1) # Pausa para ver el mensaje
                                    st.rerun()
                                else:
                                  
                                    st.error(f"❌ Símbolo '{new_asset}' no válido o no encontrado.")
                        else:
                            st.warning(f"⚠️ {new_asset} ya está en la lista.")
                    else:
                        st.warning("⚠️ Ingresa un símbolo para añadir.")

            # Eliminar Activo
            if st.session_state.portfolio_assets: # Mostrar solo si hay activos
                col1_rem, col2_rem = st.columns([3, 1])
                with col1_rem:
                    asset_to_remove = st.selectbox(
                        "Eliminar activo de la lista:",
                        options=[""] + sorted(st.session_state.portfolio_assets), # Añadir opción vacía y ordenar
                        index=0, # Empezar con la opción vacía seleccionada
                        key="remove_asset_select"
                    )
                with col2_rem:
                    st.write("") # Placeholder for vertical alignment
                    st.write("") # Placeholder for vertical alignment
                    # Botón con icono y tooltip
                    if st.button("➖ Eliminar", key="remove_asset_button", help="Quita el activo seleccionado de la lista"):
                        if asset_to_remove:
                            st.session_state.portfolio_assets.remove(asset_to_remove)
                            st.success(f"🗑️ {asset_to_remove} eliminado.")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.warning("⚠️ Selecciona un activo para eliminar.")

            # Mostrar lista actual (opcional, pero útil)
            if st.session_state.portfolio_assets:
                 st.write("**Activos Actuales:**")
                 # Mostrar en varias líneas si son muchos
                 asset_string = ", ".join(st.session_state.portfolio_assets)
                 if len(asset_string) > 50: # Umbral arbitrario
                     st.caption(asset_string)
                 else:
                     st.write(f"`{asset_string}`")
            else:
                 st.write("No hay activos seleccionados.")

        st.divider()
        # --- Sección de Parámetros ---
        with st.expander("🛠️ Parámetros del Análisis"):
            # [source: 22], [source: 23]
            st.session_state.simulation_count = st.number_input(
                "Número de Simulaciones Monte Carlo:",
                min_value=1000,
                max_value=100000, # Limitar un poco para rendimiento web
                value=st.session_state.simulation_count,
                step=1000,
                help="Cantidad de portafolios aleatorios a generar. Más simulaciones = más preciso pero más lento."
            )
          
            st.session_state.investment_amount = st.number_input(
                "Monto Total de Inversión (€):",
                min_value=100.0, # Permitir decimales
                max_value=10000000.0,
                value=float(st.session_state.investment_amount), # Asegurar float
                step=100.0,
                format="%.2f" # Formato con 2 decimales
            )
         
            st.session_state.benchmark_symbol = st.text_input(
                "Símbolo del Benchmark:",
                value=st.session_state.benchmark_symbol,
                help="Índice para comparar el rendimiento (ej: ^GSPC, ^STOXX50E)"
            ).upper().strip() # .strip()

            # Tasa libre de riesgo
            st.session_state.risk_free_rate = st.number_input(
                 "Tasa Libre de Riesgo Anual (%):",
                 min_value=0.0,
                 max_value=15.0, # Aumentar rango posible
                 value=st.session_state.risk_free_rate * 100, # Mostrar como porcentaje
                 step=0.05, # Permitir pasos más finos
                 format="%.2f",
                 help="Usada para calcular Sharpe y Sortino. Ej: Rendimiento bono del tesoro."
            ) / 100.0 # Convertir de nuevo a decimal para cálculos

            # Fechas
            # Usar columnas para mejor disposición
            col_date1, col_date2 = st.columns(2)
            with col_date1:
                st.session_state.analysis_start_date = st.date_input(
                    "Fecha de Inicio:",
                    value=st.session_state.analysis_start_date,
                    max_value=datetime.now().date() - timedelta(days=1) # No permitir fecha futura o hoy
                )
            with col_date2:
                st.session_state.analysis_end_date = st.date_input(
                    "Fecha de Fin:",
                    value=st.session_state.analysis_end_date,
                    max_value=datetime.now().date() # No permitir fecha futura
                )
            # Validar que la fecha de fin sea posterior a la de inicio
            if st.session_state.analysis_start_date >= st.session_state.analysis_end_date:
                st.warning("La fecha de fin debe ser posterior a la fecha de inicio.")


        st.divider()

        # --- Selección de Visualización ---
        st.subheader("📊 Visualizaciones")
        # [source: 27], [source: 28], [source: 29], [source: 30]
        # Renombrar opciones para mayor claridad
        graph_options = {
            "Precios Históricos (Normalizado)": "Fig 1. Gráfico de Precios",
            "Rentabilidad Acumulada por Activo": "Fig 2. Rentabilidad Simple",
            "Distribución de Retornos Diarios": "Fig 3. Histograma de Retornos",
            "Volatilidad Anualizada por Activo": "Fig 4. Gráfico de Volatilidad",
            "Volatilidad Móvil (30d)": "Fig 5. Volatilidad rentabilidad", # Cambiado
            "Matriz de Correlación": "Fig 6. Matriz de correlación",
            "Frontera Eficiente (Monte Carlo)": "Fig 7. Simulación de Monte Carlo",
            "Distribución Óptima (%)": "Fig 8. Distribución del Portafolio Óptimo",
            "Distribución Óptima (€)": "Fig 8A. Distribución del valor en €",
            "Distribución Máx. Retorno (%)": "Fig 9. Portafolio Máximo Retorno",
            #"Distribución Máx. Volatilidad (%)": "Fig 10. Portafolio Máxima Volatilidad", # Ocultar por defecto? Menos útil
            "Comparación Rendimiento vs Benchmark": "Fig 11. Comparación con Benchmark",
            "Sensibilidad al Benchmark (Beta)": "Fig 12. Análisis de Sensibilidad", # Cambiado
            "Simulación de Escenarios": "Fig 13. Análisis de Escenarios"
        }
        # Usar los nombres descriptivos en el selectbox, pero guardar la clave original
        selected_graph_label = st.selectbox(
            "Seleccionar tipo de gráfico:",
            options=list(graph_options.keys()), # Mostrar nombres descriptivos
            index=6 # Empezar con Monte Carlo por defecto
        )
        # Obtener la clave original correspondiente a la etiqueta seleccionada
        selected_graph = graph_options[selected_graph_label]


    # --- Área Principal ---
    st.divider() # Separador visual

    # Botón de análisis principal, centrado o más prominente
    # col_run_button, _, _ = st.columns([1, 2, 1]) # Columna central para el botón
    # with col_run_button:
    # Usar un botón más grande o destacado si es posible
    run_analysis = st.button("💼 Ejecutar Análisis del Portafolio", type="primary", use_container_width=True)

    # Placeholder para mostrar resultados
    results_placeholder = st.container()

    if run_analysis:
        with results_placeholder: # Poner todo el proceso dentro del placeholder
            # Validar fechas antes de proceder
            if st.session_state.analysis_start_date >= st.session_state.analysis_end_date:
                 st.error("Error: La fecha de fin debe ser posterior a la fecha de inicio. Ajusta las fechas en la barra lateral.")
                 st.stop() # Detener ejecución si las fechas son inválidas

            # Validar que haya activos
            if not st.session_state.portfolio_assets:
                 st.error("Error: No hay activos seleccionados para analizar. Agrega activos en la barra lateral.")
                 st.stop()

            try:
                start_time = time.time() # Medir tiempo de ejecución

                # --- Descargar Datos ---
                st.write("---") # Separador antes de los resultados
                st.subheader("📥 Descargando y Procesando Datos...")
                with st.spinner("Obteniendo precios históricos..."):
                    # [source: 31], [source: 32]
                    prices = download_data_with_retry(
                        st.session_state.portfolio_assets,
                        st.session_state.analysis_start_date,
                        st.session_state.analysis_end_date
                    )

                if prices is None or prices.empty:
                    st.error("Fallo Crítico: No se pudieron obtener datos válidos para los activos principales. Revisa los símbolos y el rango de fechas.")
                    st.stop()

                # Actualizar la lista de activos en session_state si algunos fallaron
                original_assets = st.session_state.portfolio_assets.copy()
                actual_assets_analyzed = prices.columns.tolist()
                if set(original_assets) != set(actual_assets_analyzed):
                     st.session_state.portfolio_assets = actual_assets_analyzed # Actualizar estado para reflejar la realidad
                     st.warning(f"Análisis procederá con {len(actual_assets_analyzed)} activos con datos válidos.")
                num_assets = len(actual_assets_analyzed)
                if num_assets == 0:
                     st.error("Error: No quedan activos con datos válidos tras la descarga.")
                     st.stop()


                # Descargar datos del benchmark
                benchmark_prices = None # Inicializar
                if st.session_state.benchmark_symbol: # Solo descargar si hay un símbolo
                    st.write(f"Obteniendo datos para Benchmark ({st.session_state.benchmark_symbol})...")
                    with st.spinner(f"Descargando {st.session_state.benchmark_symbol}..."):
                        # [source: 33], [source: 34], [source: 35]
                        benchmark_prices_df = download_data_with_retry(
                            [st.session_state.benchmark_symbol],
                            st.session_state.analysis_start_date,
                            st.session_state.analysis_end_date
                        )

                    if benchmark_prices_df is not None and not benchmark_prices_df.empty:
                        if st.session_state.benchmark_symbol in benchmark_prices_df.columns:
                            benchmark_prices = benchmark_prices_df[st.session_state.benchmark_symbol].copy()
                            benchmark_prices.dropna(inplace=True) # Limpiar NaNs específicos del benchmark
                            if benchmark_prices.empty:
                                 st.warning(f"Datos del benchmark {st.session_state.benchmark_symbol} vacíos tras limpieza. No se usará.")
                                 benchmark_prices = None # Anular si está vacío
                        else:
                            st.warning(f"No se encontraron datos para el símbolo de benchmark '{st.session_state.benchmark_symbol}'. La comparación no estará disponible.")
                    else:
                        st.warning(f"No se pudieron descargar datos para el benchmark {st.session_state.benchmark_symbol}. Se omitirá la comparación.")
                else:
                    st.info("No se especificó un símbolo de benchmark. Se omitirá la comparación.")


                # --- Cálculos Principales ---
                st.subheader("⚙️ Realizando Cálculos...")
                with st.spinner("Calculando retornos, matrices y optimizando..."):
                    # Calcular retornos logarítmicos
                    # Asegurarse que los precios no tengan NaNs antes de pct_change
                    prices.ffill().bfill(inplace=True)
                    log_returns = np.log(1 + prices.pct_change())

                    # Eliminar la primera fila (NaN) y cualquier fila/columna que sea enteramente NaN
                    log_returns = log_returns.iloc[1:]
                    log_returns.dropna(axis=0, how='all', inplace=True)
                    log_returns.dropna(axis=1, how='all', inplace=True)

                    # Reemplazar posibles Inf/-Inf si ocurrieron (ej: precio fue 0)
                    log_returns.replace([np.inf, -np.inf], 0.0, inplace=True)
                    # Rellenar cualquier NaN restante con 0 (asumiendo retorno 0 para ese día)
                    log_returns.fillna(0.0, inplace=True)


                    # Comprobar si quedan activos después de calcular retornos
                    actual_assets_analyzed = log_returns.columns.tolist()
                    num_assets = len(actual_assets_analyzed)
                    if num_assets == 0:
                        st.error("No quedan datos válidos después de calcular los retornos. Revisa los activos o el rango de fechas.")
                        st.stop()
                    # Actualizar estado de sesión si más activos fueron eliminados
                    if set(st.session_state.portfolio_assets) != set(actual_assets_analyzed):
                         st.session_state.portfolio_assets = actual_assets_analyzed
                         st.warning(f"Activos con datos insuficientes para cálculo de retornos eliminados. Analizando {num_assets} activos.")

                    # Calcular matriz de covarianza anualizada
                    # Manejar caso de un solo activo
                    if num_assets > 1:
                        cov_matrix = log_returns.cov() * 252
                    else:
                        # Crear matriz 1x1 con la varianza anualizada
                        variance = log_returns.var().iloc[0] * 252
                        cov_matrix = pd.DataFrame([[variance]], index=actual_assets_analyzed, columns=actual_assets_analyzed)

                    # Verificar si hay NaNs en la matriz de covarianza
                    if cov_matrix.isnull().values.any():
                         st.error("Error: Se encontraron valores NaN en la matriz de covarianza. Esto puede deberse a datos insuficientes o constantes para algunos activos.")
                         st.info(f"Matriz de covarianza:\n{cov_matrix}")
                         st.stop()

                    # Realizar simulación de Monte Carlo
                    #
                    mc_returns, mc_volatility, mc_sharpe, mc_weights = monte_carlo_simulation(
                        log_returns,
                        cov_matrix,
                        st.session_state.simulation_count,
                        num_assets,
                        st.session_state.risk_free_rate # Pasar la tasa libre de riesgo
                    )

                    # Limpiar posibles Inf en Sharpe (si volatilidad fue 0)
                    mc_sharpe = np.nan_to_num(mc_sharpe, nan=0.0, posinf=np.finfo(np.float64).max, neginf=np.finfo(np.float64).min)


                    # --- Encontrar Portafolios Clave ---
                    # Portafolio Óptimo (Máximo Sharpe)
                    if np.all(mc_sharpe == 0) and np.all(mc_returns <= st.session_state.risk_free_rate):
                         # Caso especial: todos los Sharpe son 0 o negativos, y todos los retornos bajos
                         st.warning("Todos los portafolios simulados tienen Sharpe Ratio bajo o negativo. El 'óptimo' se seleccionará por mínima volatilidad entre los de retorno no negativo si existen, o el de menor pérdida.")
                         non_negative_return_indices = np.where(mc_returns >= 0)[0]
                         if len(non_negative_return_indices) > 0:
                              min_vol_among_non_neg = np.argmin(mc_volatility[non_negative_return_indices])
                              max_sharpe_idx = non_negative_return_indices[min_vol_among_non_neg]
                         else: # Todos los retornos son negativos
                              max_sharpe_idx = np.argmax(mc_returns) # Elige el que menos pierde
                    else:
                         # Caso normal: encontrar máximo Sharpe
                         max_sharpe_idx = np.argmax(mc_sharpe)

                    optimal_weights = mc_weights[max_sharpe_idx]
                    optimal_return = mc_returns[max_sharpe_idx]
                    optimal_volatility = mc_volatility[max_sharpe_idx]
                    optimal_sharpe = mc_sharpe[max_sharpe_idx]


                    # Portafolio de Máximo Retorno
                    max_return_idx = np.argmax(mc_returns)
                    # [source: 38]
                    max_return_weights = mc_weights[max_return_idx]
                    max_return_return = mc_returns[max_return_idx]
                    max_return_volatility = mc_volatility[max_return_idx]
                    max_return_sharpe = mc_sharpe[max_return_idx]


                    # Portafolio de Mínima Volatilidad
                    min_vol_idx = np.argmin(mc_volatility)
                    min_vol_weights = mc_weights[min_vol_idx]
                    min_vol_return = mc_returns[min_vol_idx]
                    min_vol_volatility = mc_volatility[min_vol_idx]
                    min_vol_sharpe = mc_sharpe[min_vol_idx]


                    # Calcular Sortino y CVaR para el portafolio óptimo (máximo Sharpe)
                    # Asegurar que los pesos óptimos y retornos log tengan los mismos activos/índice
                    common_assets = log_returns.columns.intersection(actual_assets_analyzed)
                    optimal_portfolio_log_returns = (log_returns[common_assets] * optimal_weights).sum(axis=1)

                    optimal_sortino = sortino_ratio(optimal_portfolio_log_returns, st.session_state.risk_free_rate)
                    optimal_cvar_daily = calculate_cvar(optimal_portfolio_log_returns) # CVaR diario

                st.success("✅ Cálculos completados.")
                st.write(f"Tiempo de análisis: {time.time() - start_time:.2f} segundos.")
                st.write("---")


                # --- Mostrar Resultados ---
                st.subheader("📈 Resultados del Análisis")

                # Usar columnas para organizar estadísticas y gráficos
                col_results_1, col_results_2 = st.columns(2)

                with col_results_1:
                    st.subheader("⭐ Portafolio Óptimo (Máx Sharpe)")
                    st.metric(label="Retorno Anualizado Esperado", value=f"{optimal_return:.2%}")
                    st.metric(label="Volatilidad Anualizada", value=f"{optimal_volatility:.2%}")
                    st.metric(label="Ratio de Sharpe", value=f"{optimal_sharpe:.2f}" if np.isfinite(optimal_sharpe) else "N/A")
                    st.metric(label="Ratio de Sortino", value=f"{optimal_sortino:.2f}")
                    st.metric(label="CVaR Diario (95%)", value=f"{optimal_cvar_daily:.2%}" if pd.notna(optimal_cvar_daily) else "N/A", delta_color="inverse") # CVaR es negativo

                with col_results_2:
                    st.subheader("🚀 Portafolio Máximo Retorno")
                    st.metric(label="Retorno Anualizado Esperado", value=f"{max_return_return:.2%}")
                    st.metric(label="Volatilidad Anualizada", value=f"{max_return_volatility:.2%}")
                    st.metric(label="Ratio de Sharpe", value=f"{max_return_sharpe:.2f}" if np.isfinite(max_return_sharpe) else "N/A")

                    st.subheader("🛡️ Portafolio Mínima Volatilidad")
                    st.metric(label="Retorno Anualizado Esperado", value=f"{min_vol_return:.2%}")
                    st.metric(label="Volatilidad Anualizada", value=f"{min_vol_volatility:.2%}")
                    st.metric(label="Ratio de Sharpe", value=f"{min_vol_sharpe:.2f}" if np.isfinite(min_vol_sharpe) else "N/A")

                st.write("---")
                st.subheader(f"📊 Gráfico Seleccionado: {selected_graph_label}")


                # --- Generación del Gráfico Seleccionado ---
                # Crear figura y eje para el gráfico
                base_figsize = (10, 5) # Tamaño estándar más compacto para web
                fig, ax = plt.subplots(figsize=base_figsize)

                # Usar la paleta de colores definida
                num_colors = len(actual_assets_analyzed) # Usar activos realmente analizados
                plot_colors = [COLOR_PALETTE[i % len(COLOR_PALETTE)] for i in range(num_colors)]


                # --- Lógica de Graficación (adaptada para claridad y mejor UI) ---

                # Asegurar que 'prices' y 'log_returns' contienen solo los activos analizados
                prices_plot = prices[actual_assets_analyzed]
                log_returns_plot = log_returns[actual_assets_analyzed]

                if selected_graph == "Fig 1. Gráfico de Precios":
                    # [source: 39]
                    if not prices_plot.empty:
                        # Normalizar precios para comparar rendimiento relativo
                        normalized_prices = (prices_plot / prices_plot.iloc[0]) * 100
                        for i, asset in enumerate(normalized_prices.columns):
                            ax.plot(normalized_prices.index, normalized_prices[asset], label=asset, color=plot_colors[i], alpha=0.8)
                        ax.set_title("Evolución Relativa de Precios (Base 100)")
                        ax.set_xlabel("Fecha")
                        ax.set_ylabel("Precio Normalizado (Inicio = 100)")
                        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
                        ax.grid(True, linestyle='--', alpha=0.6)
                    else:
                         ax.text(0.5, 0.5, 'Datos de precios no disponibles', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)


                elif selected_graph == "Fig 2. Rentabilidad Simple":
                    # 
                    if not log_returns_plot.empty:
                        daily_cumulative_returns = (log_returns_plot + 1).cumprod()
                        for i, asset in enumerate(daily_cumulative_returns.columns):
                             ax.plot(daily_cumulative_returns.index, daily_cumulative_returns[asset], label=asset, color=plot_colors[i], alpha=0.8)
                        ax.set_title("Rentabilidad Acumulada por Activo")
                        ax.set_xlabel("Fecha")
                        ax.set_ylabel("Crecimiento de 1€ invertido")
                        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
                        ax.grid(True, linestyle='--', alpha=0.6)
                    else:
                        ax.text(0.5, 0.5, 'Datos de retornos no disponibles', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)


                elif selected_graph == "Fig 3. Histograma de Retornos":
                     # 
                     if not log_returns_plot.empty:
                        for i, asset in enumerate(log_returns_plot.columns):
                            sns.histplot(log_returns_plot[asset], ax=ax, label=asset, bins=50, alpha=0.6, kde=True, color=plot_colors[i], stat="density") # Usar densidad
                        ax.set_title('Distribución de Retornos Logarítmicos Diarios')
                        ax.set_xlabel('Retorno Logarítmico Diario')
                        ax.set_ylabel('Densidad')
                        ax.legend(fontsize='small')
                     else:
                         ax.text(0.5, 0.5, 'Datos de retornos no disponibles', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)


                elif selected_graph == "Fig 4. Gráfico de Volatilidad":
                    # 
                    if not log_returns_plot.empty:
                        volatility = log_returns_plot.std() * np.sqrt(252) # Volatilidad anualizada
                        volatility = volatility.sort_values(ascending=True) # Ordenar ascendente
                        volatility.plot(kind='barh', ax=ax, color=plot_colors) # Barras horizontales
                        ax.set_title("Volatilidad Anualizada por Activo")
                        ax.set_xlabel("Volatilidad Anualizada (Desv. Estándar)")
                        ax.set_ylabel("Activo")
                        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}')) # Formato porcentaje
                        ax.grid(axis='x', linestyle='--', alpha=0.6)
                    else:
                        ax.text(0.5, 0.5, 'Datos de retornos no disponibles', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)


                elif selected_graph == "Fig 5. Volatilidad rentabilidad": # Cambiado a Volatilidad Móvil
                     # 
                    if not log_returns_plot.empty:
                         rolling_window = 30
                         rolling_vol = log_returns_plot.rolling(window=rolling_window).std() * np.sqrt(252) # Volatilidad móvil anualizada
                         rolling_vol.dropna(inplace=True) # Eliminar NaNs iniciales del rolling
                         if not rolling_vol.empty:
                              for i, asset in enumerate(rolling_vol.columns):
                                  ax.plot(rolling_vol.index, rolling_vol[asset], label=asset, color=plot_colors[i], alpha=0.7)
                              ax.set_title(f"Volatilidad Móvil Anualizada ({rolling_window} días)")
                              ax.set_xlabel("Fecha")
                              ax.set_ylabel("Volatilidad Anualizada")
                              ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0%}')) # Formato porcentaje
                              ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
                              ax.grid(True, linestyle='--', alpha=0.6)
                         else:
                              ax.text(0.5, 0.5, f'No hay suficientes datos para ventana móvil de {rolling_window} días', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)

                    else:
                        ax.text(0.5, 0.5, 'Datos de retornos no disponibles', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)


                elif selected_graph == "Fig 6. Matriz de correlación":
                    # 
                    if not log_returns_plot.empty and num_assets > 1:
                        corr_matrix = log_returns_plot.corr()
                        annot_size = 8 if num_assets > 15 else 10
                        mask = np.triu(np.ones_like(corr_matrix, dtype=bool)) # Máscara para triángulo superior
                        sns.heatmap(corr_matrix, annot=True, cmap='vlag', ax=ax,
                                    annot_kws={"size": annot_size}, fmt=".2f", linewidths=.5,
                                    linecolor='lightgray', center=0, mask=mask) # Añadir máscara y centrar en 0
                        ax.set_title('Matriz de Correlación de Retornos Diarios')
                        plt.xticks(rotation=45, ha='right') # Mejor rotación
                        plt.yticks(rotation=0)
                        # Ajustar tamaño si es necesario (antes de mostrar)
                        if num_assets > 10:
                            fig.set_size_inches(max(base_figsize[0], num_assets * 0.6), max(base_figsize[1], num_assets * 0.5))
                    elif num_assets <= 1:
                         ax.text(0.5, 0.5, 'Se necesita más de 1 activo para la matriz de correlación', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
                    else: # log_returns_plot vacío
                         ax.text(0.5, 0.5, 'Datos de retornos no disponibles', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)


                elif selected_graph == "Fig 7. Simulación de Monte Carlo":
                    # 
                    if len(mc_returns) > 0: # Verificar que haya resultados de MC
                        # Muestreo para visualización si hay demasiados puntos
                        max_points_plot = 10000
                        step = 1
                        if len(mc_returns) > max_points_plot:
                            step = len(mc_returns) // max_points_plot

                        returns_to_plot = mc_returns[::step]
                        volatility_to_plot = mc_volatility[::step]
                        sharpe_to_plot = mc_sharpe[::step]

                        # Scatter plot de los portafolios simulados
                        scatter = ax.scatter(volatility_to_plot, returns_to_plot, c=sharpe_to_plot, cmap='viridis', marker='o', s=10, alpha=0.4, label='Portafolios Simulados')

                        # Resaltar portafolios clave con marcadores distintos y borde
                        ax.scatter(optimal_volatility, optimal_return, marker='*', s=300, c='red', edgecolors='black', label=f'Óptimo (Sharpe={optimal_sharpe:.2f})')
                        ax.scatter(min_vol_volatility, min_vol_return, marker='P', s=200, c='blue', edgecolors='black', label=f'Mín. Vol. ({min_vol_volatility:.2%})')
                        ax.scatter(max_return_volatility, max_return_return, marker='X', s=200, c='green', edgecolors='black', label=f'Máx. Ret. ({max_return_return:.2%})')

                        ax.set_title(f'Frontera Eficiente ({len(mc_returns):,} Simulaciones)')
                        ax.set_xlabel('Volatilidad Anualizada (Riesgo)')
                        ax.set_ylabel('Retorno Anualizado Esperado')
                        # Formato de porcentaje en los ejes
                        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))
                        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))

                        ax.legend(loc='best', fontsize='small', frameon=True, facecolor='white', framealpha=0.8) # Mejor ubicación y visibilidad
                        cbar = plt.colorbar(scatter, ax=ax, label='Ratio de Sharpe')
                        cbar.ax.tick_params(labelsize=8) # Ajustar tamaño ticks colorbar
                        ax.grid(True, linestyle='--', alpha=0.6)
                        # Ajustar límites para mejor visualización si es necesario
                        y_min = min(mc_returns.min(), min_vol_return, optimal_return, max_return_return)
                        y_max = max(mc_returns.max(), min_vol_return, optimal_return, max_return_return)
                        x_min = mc_volatility.min()
                        x_max = mc_volatility.max()
                        ax.set_ylim(bottom=y_min - abs(y_min*0.1), top=y_max + abs(y_max*0.1))
                        ax.set_xlim(left=x_min * 0.9, right=x_max * 1.1)

                    else:
                        ax.text(0.5, 0.5, 'No hay resultados de Monte Carlo para mostrar', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)

                elif selected_graph in ["Fig 8. Distribución del Portafolio Óptimo",
                                         "Fig 9. Portafolio Máximo Retorno",
                                         "Fig 10. Portafolio Máxima Volatilidad"]: # Gráficos de Torta
                    # [source: 52], [source: 57], [source: 58]
                    if selected_graph == "Fig 8. Distribución del Portafolio Óptimo":
                         weights_to_plot = optimal_weights
                         title = 'Distribución Portafolio Óptimo'
                    elif selected_graph == "Fig 9. Portafolio Máximo Retorno":
                         weights_to_plot = max_return_weights
                         title = 'Distribución Portafolio Máx. Retorno'
                    else: # Fig 10 (Oculto por defecto ahora)
                         # Recalcular max vol si es necesario (no guardado antes)
                         max_vol_idx = np.argmax(mc_volatility)
                         weights_to_plot = mc_weights[max_vol_idx]
                         title = 'Distribución Portafolio Máx. Volatilidad'

                    if weights_to_plot is not None and len(weights_to_plot) == len(actual_assets_analyzed):
                         threshold = 0.01 # 1%
                         # Crear Serie para facilitar el filtrado y agrupamiento
                         weights_series = pd.Series(weights_to_plot, index=actual_assets_analyzed)
                         weights_above_threshold = weights_series[weights_series >= threshold]
                         other_weight = weights_series[weights_series < threshold].sum()

                         # Agrupar 'Otros' si existe
                         plot_data = weights_above_threshold.copy()
                         if other_weight > 1e-6: # Añadir 'Otros' solo si es significativo
                              plot_data['Otros (<1%)'] = other_weight

                         plot_data.sort_values(ascending=False, inplace=True)

                         # Asegurar colores consistentes
                         pie_colors = []
                         for label in plot_data.index:
                              if label in actual_assets_analyzed:
                                   try:
                                       asset_index = actual_assets_analyzed.index(label)
                                       pie_colors.append(plot_colors[asset_index % len(plot_colors)])
                                   except ValueError: # Si el activo no estuviera por alguna razón
                                        pie_colors.append('grey')
                              else: # Para 'Otros'
                                   pie_colors.append('lightgrey')


                         # Crear Pie chart (Donut)
                         wedges, texts, autotexts = ax.pie(plot_data, labels=plot_data.index, colors=pie_colors,
                                                           autopct='%1.1f%%', startangle=90, pctdistance=0.85,
                                                           wedgeprops=dict(width=0.35, edgecolor='w')) # Donut chart más fino
                         plt.setp(autotexts, size=8, weight="bold", color="black") # Texto negro para mejor contraste
                         plt.setp(texts, size=9)
                         ax.set_title(f'{title} (Pesos > {threshold:.0%})')
                    else:
                         ax.text(0.5, 0.5, 'Pesos no disponibles para este portafolio', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)


                elif selected_graph == "Fig 8A. Distribución del valor en €":
                    # 
                    if optimal_weights is not None and len(optimal_weights) == len(actual_assets_analyzed):
                        valores_euros = pd.Series(
                            [w * st.session_state.investment_amount for w in optimal_weights],
                            index=actual_assets_analyzed
                        )
                        # Filtrar valores muy pequeños y ordenar descendente
                        valores_euros = valores_euros[valores_euros.abs() > 0.01].sort_values(ascending=False)

                        if not valores_euros.empty:
                             # Asignar colores
                             bar_colors = []
                             for label in valores_euros.index:
                                   try:
                                       asset_index = actual_assets_analyzed.index(label)
                                       bar_colors.append(plot_colors[asset_index % len(plot_colors)])
                                   except ValueError:
                                       bar_colors.append('grey')

                             bars = ax.bar(valores_euros.index, valores_euros.values, color=bar_colors)
                             ax.set_title(f'Distribución Portafolio Óptimo (€) - Total: €{st.session_state.investment_amount:,.0f}')
                             ax.set_xlabel("Activos")
                             ax.set_ylabel("Valor Asignado (€)")
                             plt.xticks(rotation=45, ha='right', fontsize=9)
                             ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'€{x:,.0f}')) # Formato moneda eje Y
                             ax.grid(axis='y', linestyle='--', alpha=0.6)

                             # Añadir etiquetas de valor encima de las barras (si no son demasiadas)
                             if len(valores_euros) < 15:
                                 for bar in bars:
                                     height = bar.get_height()
                                     ax.text(bar.get_x() + bar.get_width() / 2., height, f'€{height:,.0f}',
                                             ha='center', va='bottom', fontsize=8, rotation=90, color='black')
                             # Ajustar tamaño si hay muchos activos
                             if num_assets > 10:
                                 fig.set_size_inches(max(base_figsize[0], num_assets * 0.5), base_figsize[1])
                        else:
                             ax.text(0.5, 0.5, 'No hay asignaciones significativas en € para mostrar', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
                    else:
                         ax.text(0.5, 0.5, 'Pesos óptimos no disponibles', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)


                elif selected_graph == "Fig 11. Comparación con Benchmark":
                    # 
                    if benchmark_prices is not None and not benchmark_prices.empty and not optimal_portfolio_log_returns.empty:
                        # Calcular retornos simples diarios
                        portfolio_simple_returns = np.exp(optimal_portfolio_log_returns) - 1
                        benchmark_simple_returns = benchmark_prices.pct_change()

                        # Alinear índices y calcular retornos acumulados
                        compare_df = pd.DataFrame({
                            'Portfolio': portfolio_simple_returns,
                            'Benchmark': benchmark_simple_returns
                        }).dropna() # Eliminar días donde uno no tenga datos

                        if not compare_df.empty:
                            cumulative_returns = (1 + compare_df).cumprod() * 100 # Empezar en 100

                            ax.plot(cumulative_returns.index, cumulative_returns['Portfolio'], label='Portafolio Óptimo', color='red', linewidth=2)
                            ax.plot(cumulative_returns.index, cumulative_returns['Benchmark'], label=f'Benchmark ({st.session_state.benchmark_symbol})', color='grey', linestyle='--')

                            ax.set_title(f"Rendimiento Acumulado (Base 100): Portafolio vs {st.session_state.benchmark_symbol}")
                            ax.set_xlabel("Fecha")
                            ax.set_ylabel("Valor Acumulado (Inicio = 100)")
                            ax.legend()
                            ax.grid(True, linestyle='--', alpha=0.6)
                            # ax.set_yscale('log') # Opcional: escala logarítmica para largos periodos
                        else:
                             ax.text(0.5, 0.5, 'No hay suficientes datos comunes entre portafolio y benchmark', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
                    else:
                        st.warning("No se pueden mostrar la comparación con el benchmark (datos no disponibles o portafolio vacío).")
                        ax.text(0.5, 0.5, 'Datos de Benchmark o Portafolio no disponibles', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)


                elif selected_graph == "Fig 12. Análisis de Sensibilidad": # Cambiado a Beta
                     # [source: 62], [source: 63], [source: 64], [source: 65], [source: 66], [source: 67], [source: 68], [source: 69], [source: 70]
                    if benchmark_prices is not None and not benchmark_prices.empty and not log_returns_plot.empty and num_assets > 0:
                        benchmark_log_returns = np.log(1 + benchmark_prices.pct_change()).dropna()
                        betas = {}
                        # Alinear benchmark y retornos de activos
                        common_index = log_returns_plot.index.intersection(benchmark_log_returns.index)
                        if len(common_index) > 5: # Necesita un mínimo de puntos para calcular beta fiable
                            aligned_log_returns = log_returns_plot.loc[common_index]
                            aligned_benchmark_returns = benchmark_log_returns.loc[common_index]

                            # Calcular Beta para cada activo vs benchmark
                            market_variance = aligned_benchmark_returns.var()
                            if market_variance > 1e-12: # Asegurar que el mercado tiene varianza
                                for asset in aligned_log_returns.columns:
                                    # Covarianza entre el activo y el mercado
                                    covariance = aligned_log_returns[asset].cov(aligned_benchmark_returns)
                                    betas[asset] = covariance / market_variance
                            else:
                                st.warning("La varianza del benchmark es casi cero. No se puede calcular Beta.")
                        else:
                            st.warning(f"Muy pocos datos comunes ({len(common_index)}) para calcular Beta de forma fiable.")

                        if betas: # Si se calcularon betas
                            sorted_betas = pd.Series(betas).sort_values(ascending=False).dropna()
                            if not sorted_betas.empty:
                                beta_colors = ['navy' if b > 1.1 else 'cornflowerblue' if b >= 0.9 else 'salmon' if b >= 0 else 'darkred' for b in sorted_betas.values]
                                bars = ax.bar(sorted_betas.index, sorted_betas.values, color=beta_colors)
                                ax.axhline(1, color='black', linestyle='--', linewidth=1, label='Beta = 1 (Mercado)')
                                ax.set_title(f'Sensibilidad al Benchmark (Beta) - vs {st.session_state.benchmark_symbol}')
                                ax.set_xlabel('Activos')
                                ax.set_ylabel('Beta')
                                plt.xticks(rotation=45, ha='right', fontsize=9)
                                ax.legend()
                                ax.grid(axis='y', linestyle='--', alpha=0.6)
                                if num_assets > 10:
                                    fig.set_size_inches(max(base_figsize[0], num_assets * 0.5), base_figsize[1] + 1) # Más alto para etiquetas
                            else:
                                ax.text(0.5, 0.5, 'No se pudo calcular Beta para los activos.', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
                        else:
                            # Ya se mostró warning sobre datos insuficientes o varianza cero
                             ax.text(0.5, 0.5, 'No se pudo calcular Beta.', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)

                    else:
                        st.warning("No se puede calcular Beta (faltan datos del benchmark o retornos de activos).")
                        ax.text(0.5, 0.5, 'Datos no disponibles para Beta', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)


                elif selected_graph == "Fig 13. Análisis de Escenarios":
                     # 
                     if not optimal_portfolio_log_returns.empty:
                         # Usar retornos simples del portafolio óptimo
                         portfolio_simple_returns = np.exp(optimal_portfolio_log_returns) - 1
                         portfolio_mean_daily = portfolio_simple_returns.mean()
                         portfolio_std_daily = portfolio_simple_returns.std()

                         # Crear escenarios basados en la media y desviación estándar DIARIA
                         scenarios = {
                             f'Optimista ({portfolio_mean_daily + portfolio_std_daily:.3%}/día)': portfolio_mean_daily + portfolio_std_daily,
                             f'Base ({portfolio_mean_daily:.3%}/día)': portfolio_mean_daily,
                             f'Pesimista ({portfolio_mean_daily - portfolio_std_daily:.3%}/día)': portfolio_mean_daily - portfolio_std_daily
                         }
                         scenario_colors = {'Optimista': 'green', 'Base': 'orange', 'Pesimista': 'red'}
                         color_keys = list(scenario_colors.keys()) # Para asignar colores

                         # Simular crecimiento bajo cada escenario DIARIO constante
                         initial_investment = 100 # Simular crecimiento desde 100€
                         sim_days = len(portfolio_simple_returns)
                         sim_index = pd.date_range(start=portfolio_simple_returns.index.min(), periods=sim_days, freq='B') # Índice de días hábiles
                         sim_results = pd.DataFrame(index=sim_index)
                         day_numbers = np.arange(sim_days)

                         for i, (scenario_label, daily_rate) in enumerate(scenarios.items()):
                              sim_results[scenario_label] = initial_investment * (1 + daily_rate)**day_numbers
                              color_key = color_keys[i % len(color_keys)]
                              ax.plot(sim_results.index, sim_results[scenario_label], label=scenario_label, color=scenario_colors[color_key], linewidth=2)

                         ax.set_title('Simulación de Crecimiento bajo Escenarios de Retorno Diario')
                         ax.set_xlabel('Fecha Simulada')
                         ax.set_ylabel(f'Crecimiento Simulado de €{initial_investment}')
                         ax.legend(fontsize='small')
                         ax.grid(True, linestyle='--', alpha=0.6)
                         ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'€{x:,.0f}'))
                     else:
                         ax.text(0.5, 0.5, 'Retornos del portafolio óptimo no disponibles', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)


                # --- Mostrar Gráfico ---
                # Ajustar layout final antes de mostrar
                try:
                    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Ajustar para título principal
                except ValueError:
                    plt.tight_layout() # Fallback si rect da problemas
                # [source: 79], [source: 80] (Ajuste dinámico ya considerado dentro de cada if)
                st.pyplot(fig)


                # --- Opcional: Mostrar Tabla de Pesos Óptimos ---
                st.write("---")
                with st.expander("Ver Pesos Detallados del Portafolio Óptimo"):
                    if optimal_weights is not None and len(optimal_weights) == len(actual_assets_analyzed):
                        optimal_weights_df = pd.DataFrame({
                            'Activo': actual_assets_analyzed,
                            'Peso (%)': optimal_weights * 100,
                            'Valor (€)': optimal_weights * st.session_state.investment_amount
                        }).sort_values(by='Peso (%)', ascending=False) # Ordenar por peso

                        # Filtrar pesos muy pequeños para tabla más limpia
                        optimal_weights_df_filtered = optimal_weights_df[optimal_weights_df['Peso (%)'] >= 0.01]

                        # Formatear para visualización
                        optimal_weights_df_filtered['Peso (%)'] = optimal_weights_df_filtered['Peso (%)'].map('{:,.2f}%'.format)
                        optimal_weights_df_filtered['Valor (€)'] = optimal_weights_df_filtered['Valor (€)'].map('€{:,.2f}'.format)

                        st.dataframe(optimal_weights_df_filtered.set_index('Activo'), use_container_width=True)
                    else:
                        st.write("Pesos óptimos no disponibles.")

            except MemoryError:
                 st.error("Error de Memoria: La simulación o el procesamiento de datos con la configuración actual requiere demasiada memoria RAM.")
                 st.info("Intenta reducir el 'Número de Simulaciones Monte Carlo', la cantidad de activos, o el rango de fechas en la barra lateral.")
            except Exception as e:
                st.error(f"Ocurrió un error inesperado durante el análisis:")
                st.exception(e) # Muestra el traceback completo para depuración


    # --- Notas / Disclaimer ---
    st.divider()
    st.markdown("""
    ### Notas/ Disclaimer
    *   La aplicación utiliza datos de Yahoo Finance para obtener precios históricos.
    *   Se recomienda usar símbolos válidos de Yahoo Finance.
    *   El análisis se realiza sobre datos históricos y no constituye asesoramiento financiero.
    *   Los resultados son para fines educativos y de investigación.
    """)

# --- Ejecución Principal ---
if __name__ == "__main__":
    main()
