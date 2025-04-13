# Análisis de Portafolio v9

Esta aplicación web desarrollada con Streamlit permite realizar análisis de portafolio de inversión, incluyendo optimización de cartera, análisis de riesgo y visualización de datos financieros.

## Características

- Análisis de múltiples activos financieros
- Optimización de portafolio usando el método de Monte Carlo
- Cálculo de métricas de riesgo (Ratio de Sharpe, Ratio de Sortino, CVaR)
- Visualización de datos mediante diferentes tipos de gráficos
- Comparación con benchmark
- Análisis de sensibilidad y escenarios
- Interfaz de usuario intuitiva y responsive

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
streamlit run streamlit_portfolio_v9.py
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
- Gráfico de precios históricos
- Rentabilidad simple acumulativa
- Histograma de retornos
- Análisis de volatilidad
- Matriz de correlación
- Simulación de Monte Carlo
- Distribución del portafolio óptimo
- Distribución del valor en euros
- Comparación con benchmark
- Análisis de sensibilidad
- Análisis de escenarios

### Métricas
- Retorno esperado
- Volatilidad
- Ratio de Sharpe
- Ratio de Sortino
- CVaR (Conditional Value at Risk)

## Estructura del Proyecto

```
Python_Portafolio/
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

Este proyecto está bajo la Licencia MIT - ver el archivo LICENSE para más detalles.

## Autor

Edwardyv

## Versión

v9.0.0 