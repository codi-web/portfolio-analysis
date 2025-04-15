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
from typing import List, Dict, Optional
import requests
from requests.exceptions import RequestException

# Configuración de advertencias
warnings.filterwarnings("ignore")

# Configuración de la página
st.set_page_config(page_title="Análisis de Portafolio", layout="wide")

@st.cache_data
def download_data_with_retry(symbols: List[str], start_date: datetime, end_date: datetime, max_retries: int = 3) -> Optional[pd.DataFrame]:
    """Descarga datos de Yahoo Finance con reintentos"""
    for attempt in range(max_retries):
        try:
            # Descargar en grupos más pequeños para evitar errores
            chunk_size = 5
            all_data = []
            
            for i in range(0, len(symbols), chunk_size):
                chunk = symbols[i:i + chunk_size]
                try:
                    # Usar auto_adjust=False para mantener consistencia con versiones anteriores
                    data = yf.download(chunk, 
                                     start=start_date, 
                                     end=end_date, 
                                     progress=False,
                                     auto_adjust=False)
                    if not data.empty:
                        all_data.append(data['Close'])
                    time.sleep(1)  # Esperar 1 segundo entre descargas
                except Exception as e:
                    st.warning(f"Error al descargar {chunk}: {str(e)}")
                    time.sleep(2)  # Esperar 2 segundos antes de reintentar
                    continue
            
            if all_data:
                combined_data = pd.concat(all_data, axis=1)
                return combined_data
            
            time.sleep(2)  # Esperar 2 segundos entre reintentos
        except Exception as e:
            if attempt == max_retries - 1:
                st.error(f"Error al descargar datos: {str(e)}")
                return None
            time.sleep(2)
    return None

def verify_symbol(symbol: str) -> bool:
    """Verifica si un símbolo es válido en Yahoo Finance"""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        return bool(info)
    except:
        return False

def sortino_ratio(returns, risk_free_rate=0.0232):
    negative_returns = returns[returns < 0]
    downside_deviation = np.std(negative_returns)
    if downside_deviation == 0:
        return np.nan
    expected_return = returns.mean() * 252
    return (expected_return - risk_free_rate) / downside_deviation

def sharpe_ratio(returns, risk_free_rate=0.0232):
    expected_return = returns.mean() * 252
    volatility = returns.std() * np.sqrt(252)
    return (expected_return - risk_free_rate) / volatility

def calculate_cvar(returns, confidence_level=0.95):
    if len(returns) == 0:
        return np.nan
    var = np.percentile(returns, (1 - confidence_level) * 100)
    cvar = returns[returns <= var].mean()
    return cvar

def generate_random_color():
    return "#{:02x}{:02x}{:02x}".format(np.random.randint(0, 255), 
                                       np.random.randint(0, 255), 
                                       np.random.randint(0, 255))

def optimize_portfolio(log_returns, cov_matrix, num_assets):
    result = optimize.minimize(
        calculate_portfolio_volatility,
        num_assets * [1. / num_assets, ],
        args=(cov_matrix,),
        method='SLSQP',
        bounds=[(0, 1), ] * num_assets,
        constraints=(
            {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1},)
    )
    return result.x

def calculate_portfolio_volatility(weights, cov_matrix):
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

def monte_carlo_simulation(log_returns, cov_matrix, num_portfolios, num_assets):
    results = [[] for _ in range(4)]
    risk_free_rate = 0.0232

    for i in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        portfolio_return = np.sum(weights * log_returns.mean()) * 252
        portfolio_stddev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_stddev
        results[0].append(portfolio_return)
        results[1].append(portfolio_stddev)
        results[2].append(sharpe_ratio)
        results[3].append(weights.tolist())

    return results

def main():
   # st.title("Análisis de Portafolio v9.1.1")
# Crear columnas para logo y título
    # Ajusta las proporciones [logo, titulo] según necesites, por ejemplo [5, 2] o [6, 1]
    col1, col2 = st.columns([5, 2]) 
    
    with col1: # <-- Columna 1 ahora es para el título
        st.title("Análisis de Portafolio v9.1.1") 
        
    with col2: # <-- Columna 2 ahora es para el logo
        # --- INICIO CAMBIO LOGO ---
        # Coloca la ruta a tu archivo de logo aquí. 
        # Puede ser una ruta relativa (ej. 'images/logo.png') o absoluta.
        # Asegúrate de que el archivo exista.
        logo_path = "assets/icons.png" # <--- Corregido: Solo la ruta al archivo
        try:
            # Ajusta el 'width' según el tamaño deseado para tu logo
            st.image(logo_path, width=150) # <--- Tamaño aumentado
        except Exception as e:
            # Muestra una advertencia si no se puede cargar el logo, pero no detiene la app
            st.warning(f"Advertencia: No se pudo cargar el logo desde '{logo_path}'. Error: {e}")
        # --- FIN CAMBIO LOGO ---
    
    # Inicialización de variables de estado
    if 'assets' not in st.session_state:
        # Lista reducida de activos más estables
       st.session_state.assets = [
           'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META',  # Tech
           'JPM', 'BAC', 'GS', 'MS', 'V',            # Finance
           'JNJ', 'PG', 'KO', 'WMT', 'MCD'           # Consumer
       ]
        
    if 'num_portfolios' not in st.session_state:
        # Reducir el valor por defecto para mejorar la latencia inicial
        st.session_state.num_portfolios = 20000 # Antes era 100000 
    if 'total_money' not in st.session_state:
        st.session_state.total_money = 10000
    if 'benchmark' not in st.session_state:
        st.session_state.benchmark = "SPY"

    # Sidebar para configuración
    with st.sidebar:
        st.header("Configuración")
        
        # Agregar activo
        new_asset = st.text_input("Agregar nuevo activo (símbolo):")
        if st.button("Agregar Activo") and new_asset:
            if new_asset not in st.session_state.assets:
                if verify_symbol(new_asset):
                    st.session_state.assets.append(new_asset)
                    st.success(f"Activo {new_asset} agregado")
                    st.rerun()#st.experimental_rerun() to st.rerun()
                else:
                    st.error(f"El símbolo {new_asset} no es válido")

        # Eliminar activo
        asset_to_remove = st.selectbox("Seleccionar activo a eliminar:", st.session_state.assets)
        if st.button("Eliminar Activo"):
            st.session_state.assets.remove(asset_to_remove)
            st.success(f"Activo {asset_to_remove} eliminado")
            st.rerun()#st.experimental_rerun() to st.rerun() 

        # Configuración de parámetros
        st.session_state.num_portfolios = st.number_input("Número de Portafolios:", 
                                                        min_value=1000, 
                                                        max_value=1000000, 
                                                        value=st.session_state.num_portfolios)
        
        st.session_state.total_money = st.number_input("Total de dinero (€):", 
                                                     min_value=1000, 
                                                     max_value=1000000, 
                                                     value=st.session_state.total_money)
        
        st.session_state.benchmark = st.text_input("Índice de referencia:", 
                                                 value=st.session_state.benchmark)

        # Fechas
        end_date = datetime.now()
        start_date = end_date - timedelta(days=5*365)
        st.session_state.start_date = st.date_input("Fecha de inicio:", value=start_date)
        st.session_state.end_date = st.date_input("Fecha de finalización:", value=end_date)

        # Selección de gráfico
        graph_options = [
            "Fig 1. Gráfico de Precios",
            "Fig 2. Rentabilidad Simple",
            "Fig 3. Histograma de Retornos",
            "Fig 4. Gráfico de Volatilidad",
            "Fig 5. Volatilidad rentabilidad",
            "Fig 6. Matriz de correlación",
            "Fig 7. Simulación de Monte Carlo",
            "Fig 8. Distribución del Portafolio Óptimo",
            "Fig 8A. Distribución del valor en €",
            "Fig 9. Portafolio Máximo Retorno",
            "Fig 10. Portafolio Máxima Volatilidad",
            "Fig 11. Comparación con Benchmark",
            "Fig 12. Análisis de Sensibilidad",
            "Fig 13. Análisis de Escenarios"
        ]
        selected_graph = st.selectbox("Seleccionar tipo de gráfico:", graph_options)

    # Botón de análisis
    if st.button("Analizar"):
        try:
            with st.spinner("Descargando datos..."):
                # Descargar datos
                prices = download_data_with_retry(st.session_state.assets, 
                                               st.session_state.start_date, 
                                               st.session_state.end_date)
                
                if prices is None or prices.empty:
                    st.error("No se pudieron descargar los datos de los activos. Por favor, verifica los símbolos y vuelve a intentar.")
                    return

                benchmark_prices = download_data_with_retry([st.session_state.benchmark], 
                                                         st.session_state.start_date, 
                                                         st.session_state.end_date)
                
                if benchmark_prices is None or benchmark_prices.empty:
                    st.error(f"No se pudieron descargar los datos del benchmark {st.session_state.benchmark}. Por favor, verifica el símbolo y vuelve a intentar.")
                    return

            # Calcular retornos
            log_returns = np.log(1 + prices.pct_change().dropna())
            cov_matrix = log_returns.cov() * 252

            # Realizar simulación de Monte Carlo
            results = monte_carlo_simulation(log_returns, 
                                           cov_matrix, 
                                           st.session_state.num_portfolios, 
                                           len(st.session_state.assets))

            # Encontrar portafolios óptimos
            max_sharpe_idx = np.argmax(results[2])
            optimal_weights = results[3][max_sharpe_idx]
            max_return_idx = np.argmax(results[0])
            max_return_weights = results[3][max_return_idx]
            max_volatility_idx = np.argmax(results[1])
            max_volatility_weights = results[3][max_volatility_idx]

            # Crear figura según selección
            fig, ax = plt.subplots(figsize=(12, 6))

            if selected_graph == "Fig 1. Gráfico de Precios":
                prices.plot(ax=ax)
                ax.set_title("Histórico de Precios de Cierre")
                ax.set_xlabel("Fecha")
                ax.set_ylabel("Precio de cierre (USD)")
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

            elif selected_graph == "Fig 2. Rentabilidad Simple":
                daily_cumulative_returns = (log_returns + 1).cumprod()
                daily_cumulative_returns.plot(ax=ax)
                ax.set_title("Rentabilidad Simple Acumulativa Diaria")
                ax.set_xlabel("Fecha")
                ax.set_ylabel("Crecimiento de 1€ inversión")
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

            elif selected_graph == "Fig 3. Histograma de Retornos":
                for asset in st.session_state.assets:
                    ax.hist(log_returns[asset], label=asset, bins=200, alpha=0.5)
                ax.set_title('Histograma de Retornos')
                ax.set_xlabel('Retorno Logarítmico Diario')
                ax.set_ylabel('Frecuencia')
                ax.legend()

            elif selected_graph == "Fig 4. Gráfico de Volatilidad":
                log_returns.std().plot(kind='bar', ax=ax)
                ax.set_title("Volatilidad de los Retornos Logarítmicos Diarios")
                ax.set_xlabel("Activo")
                ax.set_ylabel("Volatilidad")

            elif selected_graph == "Fig 5. Volatilidad rentabilidad":
                for asset in st.session_state.assets:
                    log_returns[asset].plot(ax=ax, label=asset)
                ax.set_title("Volatilidad diaria logarítmica de la rentabilidad")
                ax.set_xlabel("Fecha")
                ax.set_ylabel("Retorno diario logarítmico")
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

            elif selected_graph == "Fig 6. Matriz de correlación":
                corr_matrix = log_returns.corr()
                # Ajustar tamaño de fuente de anotaciones basado en número de activos
                annot_kws = {"size": 8 if len(st.session_state.assets) > 15 else 10} 
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax, annot_kws=annot_kws, fmt=".2f") # Añadir formato y tamaño de fuente
                ax.set_title('Matriz de correlación de los activos')
                plt.xticks(rotation=90) # Rotar etiquetas eje X
                plt.yticks(rotation=0)  # Mantener etiquetas eje Y horizontales
                plt.tight_layout() # Ajustar layout

            elif selected_graph == "Fig 7. Simulación de Monte Carlo":
                # Opcional: Muestrear puntos si num_portfolios es muy grande para acelerar el renderizado
                num_points_to_plot = st.session_state.num_portfolios
                step = 1
                if num_points_to_plot > 50000: # Umbral ajustable
                    step = num_points_to_plot // 50000 
                
                returns_to_plot = results[0][::step]
                volatility_to_plot = results[1][::step]
                sharpe_to_plot = results[2][::step]

                scatter = ax.scatter(volatility_to_plot, returns_to_plot, c=sharpe_to_plot, cmap='YlGnBu', marker='o', s=10, alpha=0.7) # Añadido alpha
                
                # Los puntos óptimos se dibujan siempre
                ax.scatter(results[1][max_sharpe_idx], results[0][max_sharpe_idx], 
                         marker='o', s=200, label='Portafolio Óptimo', color='green', edgecolors='black') # Añadido borde
                ax.scatter(results[1][max_return_idx], results[0][max_return_idx], 
                         marker='o', s=200, label='Portafolio Máximo Retorno', color='blue', edgecolors='black') # Añadido borde
                ax.scatter(results[1][max_volatility_idx], results[0][max_volatility_idx], 
                         color='red', marker='o', s=200, label='Portafolio Máxima Volatilidad', edgecolors='black') # Añadido borde
                
                ax.set_title('Simulación de Monte Carlo')
                ax.set_xlabel('Volatilidad (Riesgo)')
                ax.set_ylabel('Retorno Esperado')
                ax.legend()
                plt.colorbar(scatter, ax=ax, label='Ratio de Sharpe')

            elif selected_graph == "Fig 8. Distribución del Portafolio Óptimo":
                sizes = [round(weight * 100, 2) for weight in optimal_weights]
                colors = [generate_random_color() for _ in range(len(st.session_state.assets))]
                ax.pie(sizes, labels=st.session_state.assets, colors=colors, autopct='%1.1f%%')
                ax.set_title('Distribución del Portafolio Óptimo')

            elif selected_graph == "Fig 8A. Distribución del valor en €":
                valores_euros = [round(weight * st.session_state.total_money, 2) for weight in optimal_weights]
                colors = [generate_random_color() for _ in range(len(st.session_state.assets))]
                bars = ax.bar(st.session_state.assets, valores_euros, color=colors)
                ax.set_title(f'Distribución del valor en € (Total: €{st.session_state.total_money})')
                ax.set_xlabel("Activos")
                ax.set_ylabel("Valor €")
                # Rotar etiquetas del eje X para mejor legibilidad
                plt.xticks(rotation=90) 
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'€{height:,.2f}',
                           ha='center', va='bottom', fontsize=8) # Reducir tamaño fuente texto barra
                plt.tight_layout() # Ajustar layout

            elif selected_graph == "Fig 9. Portafolio Máximo Retorno":
                sizes = [round(weight * 100, 2) for weight in max_return_weights]
                colors = [generate_random_color() for _ in range(len(st.session_state.assets))]
                ax.pie(sizes, labels=st.session_state.assets, colors=colors, autopct='%1.1f%%')
                ax.set_title('Distribución del Portafolio Máximo Retorno')

            elif selected_graph == "Fig 10. Portafolio Máxima Volatilidad":
                sizes = [round(weight * 100, 2) for weight in max_volatility_weights]
                colors = [generate_random_color() for _ in range(len(st.session_state.assets))]
                ax.pie(sizes, labels=st.session_state.assets, colors=colors, autopct='%1.1f%%')
                ax.set_title('Distribución del Portafolio Máxima Volatilidad')

            elif selected_graph == "Fig 11. Comparación con Benchmark":
                portfolio_returns = (log_returns @ optimal_weights).dropna()
                # Asegurarse que benchmark_returns sea una Serie si solo hay una columna
                if isinstance(benchmark_prices, pd.DataFrame) and len(benchmark_prices.columns) == 1:
                    benchmark_returns = benchmark_prices.iloc[:, 0].pct_change().dropna()
                else:
                     # Si ya es una Serie o tiene más de una columna (aunque no debería)
                     benchmark_returns = benchmark_prices.pct_change().dropna()

                cumulative_portfolio_returns = (1 + portfolio_returns).cumprod()
                cumulative_benchmark_returns = (1 + benchmark_returns).cumprod()
                cumulative_portfolio_returns.plot(ax=ax, label='Portafolio Óptimo', color='green')
                cumulative_benchmark_returns.plot(ax=ax, label=f'Benchmark ({st.session_state.benchmark})', color='blue')
                ax.set_title("Comparación de Retornos Acumulados con el Benchmark")
                ax.set_xlabel("Fecha")
                ax.set_ylabel("Retorno Acumulado")
                ax.legend()
                plt.tight_layout() # Añadir ajuste

            elif selected_graph == "Fig 12. Análisis de Sensibilidad":
                sensitivities_map = {}
                # Calcular correlación de cada activo con la media de retornos
                for asset in st.session_state.assets:
                    # Calcular correlación. Usar fillna(0) por si hay columnas con NaN constante
                    mean_returns = log_returns.mean(axis=1).fillna(0)
                    asset_returns = log_returns[asset].fillna(0)
                    # Asegurarse que ambos tengan el mismo índice para la correlación
                    common_index = mean_returns.index.intersection(asset_returns.index)
                    sensitivity_value = np.nan # Valor por defecto si no se puede calcular
                    if not common_index.empty:
                        # Calcular correlación solo si hay datos comunes y ambos std > 0
                        asset_std = asset_returns.loc[common_index].std()
                        mean_std = mean_returns.loc[common_index].std()
                        if asset_std > 0 and mean_std > 0:
                            sensitivity_value = asset_returns.loc[common_index].corr(mean_returns.loc[common_index])
                        else:
                            # Si una de las desviaciones es 0, la correlación es indefinida o 0. Asignar 0.
                            sensitivity_value = 0 
                    
                    sensitivities_map[asset] = sensitivity_value

                # Ordenar activos por sensibilidad descendente (manejar NaN poniéndolos al final)
                # Convertir NaN a un valor muy bajo para ordenar (-infinito)
                sorted_assets = sorted(sensitivities_map, key=lambda k: sensitivities_map.get(k, -np.inf), reverse=True)
                sorted_sensitivities = [sensitivities_map[asset] for asset in sorted_assets]

                # Graficar barras con datos ordenados
                ax.bar(sorted_assets, sorted_sensitivities, color='skyblue')
                ax.set_title('Análisis de Sensibilidad (Ordenado Descendente)') # Título actualizado
                ax.set_xlabel('Activos')
                # Etiqueta clarificada
                ax.set_ylabel('Sensibilidad (Correlación con media retornos)')
                # Usar los activos ordenados para las etiquetas del eje X
                ax.set_xticks(range(len(sorted_assets))) 
                ax.set_xticklabels(sorted_assets, rotation=90) # Rotar etiquetas eje X
                plt.tight_layout() # Ajustar layout

            elif selected_graph == "Fig 13. Análisis de Escenarios":
                # Usar retornos del portafolio óptimo para los escenarios
                portfolio_log_returns = (log_returns @ optimal_weights).dropna()
                
                # Calcular ajuste aditivo basado en la media (ej: +/- 20% de la media diaria)
                # Se suma/resta este ajuste a cada retorno diario
                base_mean_return = portfolio_log_returns.mean()
                # Usar un pequeño valor absoluto si la media es muy cercana a cero para evitar ajustes nulos
                adj_factor = 0.0005 # Factor de ajuste diario (ej: 0.05% diario)
                
                scenarios_adjustments = {
                    'Escenario Optimista': adj_factor, # Sumar un pequeño retorno diario
                    'Escenario Base': 0.0,             # Sin ajuste
                    'Escenario Pesimista': -adj_factor # Restar un pequeño retorno diario
                }

                # Colores fijos para asegurar consistencia visual
                scenario_colors = {
                    'Escenario Optimista': 'blue',
                    'Escenario Base': 'orange',
                    'Escenario Pesimista': 'red'
                }

                # Graficar cada escenario aplicando el ajuste ADITIVO
                for scenario, adjustment in scenarios_adjustments.items():
                    # Aplicar ajuste aditivo a los retornos logarítmicos diarios
                    adjusted_portfolio_log_returns = portfolio_log_returns + adjustment
                    # Calcular retornos acumulados (convertir de log a simple para acumular con producto)
                    # (1 + simple_return) = exp(log_return)
                    cumulative_returns = (np.exp(adjusted_portfolio_log_returns)).cumprod()
                    # Graficar iniciando desde 1
                    cumulative_returns.plot(ax=ax, label=scenario, color=scenario_colors[scenario])

                ax.set_title('Análisis de Escenarios del Portafolio Óptimo (Ajuste Aditivo)')
                ax.set_xlabel('Fecha')
                ax.set_ylabel('Crecimiento Simulado de 1€') # Etiqueta Y ajustada
                ax.legend()
                plt.tight_layout() # Ajustar layout

            # Mostrar la figura
            # Ajustar tamaño figura dinámicamente para heatmap y barras si hay muchos activos
            if selected_graph in ["Fig 6. Matriz de correlación", "Fig 8A. Distribución del valor en €", "Fig 12. Análisis de Sensibilidad"] and len(st.session_state.assets) > 15: # Añadido Fig 12 de nuevo
                 fig.set_size_inches(14, 10) # Aumentar tamaño si hay muchos activos
            elif selected_graph == "Fig 6. Matriz de correlación":
                 fig.set_size_inches(12, 10) # Tamaño estándar un poco más alto para heatmap
            # elif selected_graph == "Fig 12. Análisis de Sensibilidad" and len(st.session_state.assets) > 15: # Condición movida arriba
            #      fig.set_size_inches(14, 8) 
            else:
                 fig.set_size_inches(12, 6) # Tamaño estándar

            st.pyplot(fig)

            # Mostrar estadísticas
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Portafolio Óptimo")
                st.write(f"Retorno: {round(results[0][max_sharpe_idx] * 100, 2)}%")
                st.write(f"Volatilidad: {round(results[1][max_sharpe_idx], 2)}")
                st.write(f"Ratio de Sharpe: {round(results[2][max_sharpe_idx], 2)}")
                st.write(f"Ratio de Sortino: {round(sortino_ratio(log_returns @ optimal_weights), 2)}")

            with col2:
                st.subheader("Portafolio Máximo Retorno")
                st.write(f"Retorno: {round(results[0][max_return_idx] * 100, 2)}%")
                st.write(f"Volatilidad: {round(results[1][max_return_idx], 2)}")
                st.write(f"Ratio de Sharpe: {round(results[2][max_return_idx], 2)}")

        except Exception as e:
            st.error(f"Error al realizar el análisis: {str(e)}")

if __name__ == "__main__":
    main() 
