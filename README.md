# Análisis de Portafolio v9.1.2
![Banner](assets/Banner_Portfolio.png)

Esta aplicación web desarrollada con Streamlit permite realizar análisis de portafolio de inversión, incluyendo optimización de cartera, análisis de riesgo y visualización de datos financieros.

## Características

- Análisis de múltiples activos financieros
- Optimización de portafolio usando el método de Monte Carlo
- Cálculo de métricas de riesgo (Ratio de Sharpe, Ratio de Sortino, CVaR)
- Visualización de datos mediante diferentes tipos de gráficos
- Comparación con benchmark
- Análisis de sensibilidad y escenarios
- Interfaz de usuario intuitiva y sensible

## Requisitos

- Python 3.8 o superior
- Dependencias listadas en `requirements.txt`

## Instalación

1. Clonar el repositorio:
```bash
git clone [URL_DEL_REPOSITORIO]
cd Python_Portafolio
```

2. Crear y activar entorno virtual:
```bash
python -m venv venv
source venv/bin/activate  # Para Linux/Mac
# o
.\venv\Scripts\activate  # Para Windows
```

3. Instalar dependencias:
```bash
pip install -r requirements.txt
```

## Uso

1. Activar el entorno virtual (si no está activado):
```bash
source venv/bin/activate  # Para Linux/Mac
# o
.\venv\Scripts\activate  # Para Windows
```

2. Ejecutar la aplicación:
```bash
streamlit run streamlit_portfolio_v9.1.1.py
```

3. Abrir el navegador en la dirección indicada (típicamente http://localhost:8501)

## Funcionalidades

### Configuración
- Agregar/eliminar activos
- Configurar número de portafolios para simulación
- Establecer monto total de inversión
- Seleccionar benchmark
- Definir período de análisis

### Análisis
- Gráfico de precios históricos (Fig 1.)
  
![Fig.1](assets/historical_prices.png)
  
- Rentabilidad simple acumulativa (Fig 2.)
  
![Fig 2.](assets/simple_profitability.png)
 
- Histograma de retornos (Fig 3.)
  
![Fig 3.](assets/returns_histogram.png)
 
- Análisis de volatilidad (Fig 4.)

![Fig 4.](assets/volatility_chart.png)
  
- Volatilidad de la rentabilidad (Fig 5.)

![Fig 5.](assets/volatility_profitability.png)
  
-  Matriz de correlación (Fig 6.)
  
![Fig 6.](assets/correlation_matrix.png)
  
- Simulación de Monte Carlo (Fig 7.)

![Fig 7.](assets/monte_carlo_simulation.png)

- Distribución del portafolio óptimo (Fig 8.)

![Fig 8.](assets/optimal_portfolio_distribution.png)

- Distribución del valor en euros (Fig 8A.)

![Fig 8A.](assets/distribution_value_euros.png)

- Comparación con benchmark (Fig 11.)

![Fig 11.](assets/comparison_benchmark.png)

- Análisis de sensibilidad (Fig 12.)

![Fig 12.](assets/sensitivity_analysis.png)

- Análisis de escenarios (Fig 13.)

![Fig 13.](assets/scenario_analysis.png)

### Métricas
- Retorno esperado
- Volatilidad
- Ratio de Sharpe
- Ratio de Sortino

## Estructura del Proyecto

```
Python_Portafolio/
├── assets/
│   ├── logo.png
│   └── Fig.png
├── streamlit_portfolio_v9.py
├── requirements.txt
└── README.md

```

## Dependencias Principales

- streamlit==1.32.0
- pandas==2.2.1
- numpy==1.26.4
- yfinance==0.2.37
- matplotlib==3.8.3
- seaborn==0.13.2
- scipy==1.12.0
- requests==2.31.0

## Notas

- La aplicación utiliza datos de Yahoo Finance para obtener precios históricos
- Se recomienda usar símbolos válidos de Yahoo Finance
- El análisis se realiza sobre datos históricos y no constituye asesoramiento financiero
- Los resultados son para fines educativos y de investigación

## Contribuciones

Las contribuciones son bienvenidas. Por favor, abre un issue primero para discutir los cambios que te gustaría realizar.

## Licencia

Este proyecto está bajo la Licencia Apache-2.0 - ver el archivo LICENSE para más detalles.

## Autor

codi-web

## Versión

v9.1.1 
