# 🏠 California Housing Price Predictor



**ML-приложение для прогнозирования стоимости недвижимости в Калифорнии с интерпретацией результатов через SHAP**


##  О проекте

Этот проект решает задачу регрессии для предсказания медианной стоимости жилья в районах Калифорнии на основе данных переписи 1990 года.

**Ключевые особенности:**
-  **Полный ML-пайплайн**: от EDA до деплоя
-  **Сравнение 4 моделей**: Linear Regression, Random Forest, XGBoost, LightGBM
-  **Интерпретация**: SHAP-анализ для объяснения предсказаний
-  **Web-интерфейс**: интерактивное приложение на Streamlit
-  **Визуализация**: Plotly для интерактивных графиков

# Результаты моделей 
| Модель | Метрики |
|--------|---------|
| **Linear Regression** | $R²:$ 0.625, $RMSE:$ $70,060, $MAE:$ $50,670, $MAPE:$ 29.19% |
| **Random Forest (tuned)** | $R²:$ 0.818, $RMSE:$ $48,844, $MAE:$ $31,507, $MAPE:$ 17.67% |
| **XGBoost**  | $R²:$ 0.819, $RMSE:$ $48,763, $MAE:$ $32,612, $MAPE:$ 18.36% |
| **LightGBM** | $R²:$ 0.818, $RMSE:$ $48,826, $MAE:$ $32,816, $MAPE:$ 18.54% |

##  Технологии

| Категория | Инструменты |
|-----------|-------------|
| **Язык** | Python 3.8+ |
| **ML/DL** | scikit-learn, XGBoost, LightGBM |
| **Интерпретация** | SHAP |
| **Визуализация** | Plotly, Matplotlib, Seaborn |
| **Web-фреймворк** | Streamlit |
| **Обработка данных** | pandas, numpy |
| **Сериализация** | joblib |

##  Установка

1. Клонируйте репозиторий

    ```bash
    git clone https://github.com/yourusername/house-price-predictor.git
    cd house-price-predictor
   
2. Создайте виртуальное окружение
    ```
    python -m venv venv
    # Windows
   venv\Scripts\activate
   
    # macOS/Linux
    source venv/bin/activate

3. Установите зависимости
    ```bash
   pip install -r requirements.txt

4. Скачайте данные

   Датасет housing.csv можно скачать:

   - Kaggle: California Housing Prices
   - Или используйте встроенный в sklearn: from sklearn.datasets import fetch_california_housing
   
   Поместите файл в корень проекта или папку data/.


# Быстрый старт

Запуск Streamlit-приложения
    
```bash
  streamlit run app.py
   ```

    




