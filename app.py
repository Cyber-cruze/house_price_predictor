import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import re

# Настройка страницы
st.set_page_config(
    page_title="🏠 California Housing Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)



# Заголовок
st.markdown('<h3 style="color: #1f77b4; font-weight: 600; margin-bottom: 8px;">🏠 California Housing Price Predictor</h3>', unsafe_allow_html=True)
st.markdown("### Прогнозирование стоимости недвижимости с помощью ML и интерпретацией через SHAP")
st.markdown("---")


# Загрузка модели и данных (кэшируем)
@st.cache_resource
def load_model():
    return joblib.load('xgb_model.pkl')


@st.cache_data
def load_data():
    df = pd.read_csv('housing.csv')
    df_prep = pd.get_dummies(df, columns=['ocean_proximity'])
    clean_names = lambda x: re.sub(r'[<>\[\]]', '_', str(x))
    df_prep = df_prep.rename(columns=clean_names)
    return df, df_prep


try:
    model = load_model()
    df_raw, df_prep = load_data()
    st.success("Модель и данные загружены успешно!")
except Exception as e:
    st.error(f"Ошибка загрузки: {e}")
    st.stop()

# Сайдбар с параметрами
st.sidebar.header("⚙️ Параметры дома")
st.sidebar.markdown("---")

median_income = st.sidebar.slider("💰 Медианный доход района ($1000)", 0.5, 15.0, 3.5, 0.1)
housing_median_age = st.sidebar.slider("🏗️ Возраст жилья (лет)", 1, 52, 25, 1)
total_rooms = st.sidebar.number_input("🚪 Всего комнат", min_value=1, max_value=10000, value=1000, step=50)
total_bedrooms = st.sidebar.number_input("🛏️ Всего спален", min_value=1, max_value=2000, value=200, step=10)
population = st.sidebar.number_input("👥 Население", min_value=1, max_value=10000, value=500, step=50)
households = st.sidebar.number_input("🏡 Домохозяйства", min_value=1, max_value=2000, value=200, step=10)
latitude = st.sidebar.slider("📍 Широта", 32.5, 42.0, 34.0, 0.1)
longitude = st.sidebar.slider("📍 Долгота", -124.5, -114.0, -118.0, 0.1)

ocean_proximity = st.sidebar.selectbox(
    "🌊 Близость к океану",
    ["NEAR BAY", "<1H OCEAN", "NEAR OCEAN", "INLAND", "ISLAND"]
)

st.sidebar.markdown("---")


# Создание DataFrame для предсказания
input_data = pd.DataFrame({
    'longitude': [longitude],
    'latitude': [latitude],
    'housing_median_age': [housing_median_age],
    'total_rooms': [total_rooms],
    'total_bedrooms': [total_bedrooms],
    'population': [population],
    'households': [households],
    'median_income': [median_income],
    f'ocean_proximity_{ocean_proximity.replace(" ", "_").replace("<1H", "_1H").replace("<", "_")}': [1]
})

# Добавляем отсутствующие one-hot колонки
for col in df_prep.columns:
    if col.startswith('ocean_proximity_') and col not in input_data.columns:
        input_data[col] = 0

# Заполняем нулями недостающие колонки
for col in df_prep.columns:
    if col not in input_data.columns and col != 'median_house_value':
        input_data[col] = 0

# Оставляем только нужные колонки в правильном порядке
feature_cols = [col for col in df_prep.columns if col != 'median_house_value']
input_data = input_data[feature_cols]

# Предсказание
prediction = model.predict(input_data)[0]

# Отображение прогноза
col1, col2, col3 = st.columns(3)
with col1:
    st.metric(" Прогнозируемая цена", f"${prediction:,.0f}", f"{(prediction / 1000):.1f}k")
with col2:
    st.metric(" Медианная цена в датасете", f"${df_raw['median_house_value'].median():,.0f}", "База")
with col3:
    diff = ((prediction - df_raw['median_house_value'].median()) / df_raw['median_house_value'].median() * 100)
    st.metric(" Отклонение от медианы", f"{diff:+.1f}%")

st.markdown("---")

# SHAP анализ для текущего предсказания
st.header("🔍 Интерпретация прогноза (SHAP)")

# Инициализация SHAP
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(input_data)

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader(" Влияние признаков на цену")

    # Создаём фигуру для SHAP
    fig, ax = plt.subplots(figsize=(10, 8))

    # Получаем данные для текущего предсказания
    shap.initjs()

    # Простой bar plot важности для текущего предсказания
    shap_abs = np.abs(shap_values[0])
    top_features = 10

    fig_bar = go.Figure(data=[
        go.Bar(
            x=shap_values[0][:top_features],
            y=input_data.columns[:top_features],
            orientation='h',
            marker=dict(
                color=shap_values[0][:top_features],
                colorscale='RdBu',
                showscale=True
            )
        )
    ])
    fig_bar.update_layout(
        title='Влияние признаков на прогноз',
        xaxis_title='влияние на цену',
        yaxis_title='Признак',
        height=400
    )
    st.plotly_chart(fig_bar, use_container_width=True)

with col2:
    st.subheader("🎯 Топ влияния")

    # Топ признаков по абсолютному влиянию
    shap_abs = np.abs(shap_values[0])
    top_idx = np.argsort(shap_abs)[-5:][::-1]

    for idx in top_idx:
        feat_name = input_data.columns[idx]
        value = shap_values[0][idx]
        feat_val = input_data.iloc[0][idx]

        color = "🔼" if value > 0 else "🔽"
        st.metric(
            f"{color} {feat_name}",
            f"${value:,.0f}",
            f"Значение: {feat_val:.2f}" if isinstance(feat_val, float) else f"Значение: {feat_val}"
        )

st.markdown("---")

# Визуализация данных
st.header(" Разведочный анализ данных")

tab1, tab2, tab3 = st.tabs(["🗺️ География", "📈 Распределение цен", "🔗 Корреляции"])

with tab1:
    st.subheader("Распределение домов по Калифорнии")
    fig_map = px.scatter_mapbox(
        df_raw,
        lat="latitude",
        lon="longitude",
        color="median_house_value",
        size="population",
        color_continuous_scale=px.colors.sequential.Plasma,
        zoom=5,
        mapbox_style="carto-positron",
        hover_data=['ocean_proximity', 'median_income']
    )
    fig_map.update_layout(height=600)
    st.plotly_chart(fig_map, use_container_width=True)

with tab2:
    st.subheader("Распределение цен по категориям близости к океану")
    fig_hist = px.histogram(
        df_raw,
        x="median_house_value",
        color="ocean_proximity",
        nbins=50,
        opacity=0.7,
        labels={'median_house_value': 'Цена дома ($)', 'count': 'Количество'}
    )
    fig_hist.update_layout(height=500)
    st.plotly_chart(fig_hist, use_container_width=True)

with tab3:
    st.subheader("Корреляционная матрица признаков")
    corr_matrix = df_prep.corr()
    fig_corr = px.imshow(
        corr_matrix,
        text_auto='.2f',
        aspect='auto',
        color_continuous_scale='RdBu_r',
        zmin=-1,
        zmax=1
    )
    fig_corr.update_layout(height=600)
    st.plotly_chart(fig_corr, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p> Датасет: California Housing Prices (1990 Census)</p>
</div>
""", unsafe_allow_html=True)