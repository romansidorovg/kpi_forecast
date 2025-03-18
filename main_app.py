import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime
from dateutil.relativedelta import relativedelta

st.title("Прогноз KPI")

# Функция для загрузки данных
def load_data():
    uploaded_file = st.file_uploader("Загрузите ваш CSV файл", type="csv")
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    return None

# Функция для извлечения признаков
def get_features(df):
    new_df = df.copy(deep=True)
    new_df['year'] = new_df['date_trunc'].dt.year
    new_df['month'] = new_df['date_trunc'].dt.month
    new_df['day'] = new_df['date_trunc'].dt.day
    new_df['day_of_week'] = new_df['date_trunc'].dt.dayofweek
    new_df['week_of_year'] = new_df['date_trunc'].dt.isocalendar().week
    new_df['day_of_year'] = new_df['date_trunc'].dt.dayofyear
    new_df['quarter'] = new_df['date_trunc'].dt.quarter
    new_df['is_holiday'] = new_df['date_trunc'].isin(holidays['ds']).astype(int)

    return new_df

# Загрузка данных
df = load_data()
if df is not None:
    df.rename(columns={'ds': 'date_trunc'}, inplace=True)
    df['date_trunc'] = pd.to_datetime(df['date_trunc'])
    df = df.sort_values('date_trunc').reset_index(drop=True)

    # Список праздников
    holidays = pd.DataFrame({
    'holiday': ['Новый год', 'Рождество Христово', 'День защитника Отечества', 'Международный женский день', 
                'Пасха', 'День труда', 'День Победы', 'Россия День независимости', 'День народного единства',
                'новогодние 1', 'новогодние 2', 'новогодние 3', 'новогодние 4', 'новогодние 5', 'новогодние 6', 'новогодние 7',
                'Новый год', 'Рождество Христово', 'День защитника Отечества', 'Международный женский день', 
                'Пасха', 'День труда', 'День Победы', 'Россия День независимости', 'День народного единства',
                'новогодние 1', 'новогодние 2', 'новогодние 3', 'новогодние 4', 'новогодние 5', 'новогодние 6', 'новогодние 7',
                'Новый год', 'Рождество Христово', 'День защитника Отечества', 'Международный женский день',
                'Пасха', 'День труда', 'День Победы', 'Россия День независимости', 'День народного единства',
                'новогодние 1', 'новогодние 2', 'новогодние 3', 'новогодние 4', 'новогодние 5', 'новогодние 6', 'новогодние 7'],
    'ds': pd.to_datetime([
        '2023-01-01', '2023-01-07', '2023-02-23', '2023-03-08',  # 2023
        '2023-04-16', '2023-05-01', '2023-05-09', '2023-06-12', '2023-11-04', '2023-12-30', '2023-12-31', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05', '2024-01-06',
        '2024-01-01', '2024-01-07', '2024-02-23', '2024-03-08',  # 2024
        '2024-04-07', '2024-05-01', '2024-05-09', '2024-06-12', '2024-11-04', '2024-12-30', '2024-12-31', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05', '2024-01-06',
        '2025-01-01', '2025-01-07', '2025-02-23', '2025-03-08',  # 2025
        '2025-04-20', '2025-05-01', '2025-05-09', '2025-06-12', '2025-11-04', '2025-12-30', '2025-12-31', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05', '2024-01-06'
    ])
})

    # Функция для получения списка кварталов
    def get_quarters():
        current_year = datetime.now().year
        quarters = []
        for year in range(current_year - 1, current_year + 2):  # От прошлого года до следующего
            for quarter in range(1, 5):  # 1-4 кварталы
                quarters.append(f"{quarter}Q {year}")
        return quarters
    
    # Функция для определения следующего квартала
    def get_next_quarter():
        now = datetime.now()
        current_quarter = (now.month - 1) // 3 + 1  # Текущий квартал (1-4)
        current_year = now.year

        if current_quarter == 4:
            next_quarter = 1
            next_year = current_year + 1
        else:
            next_quarter = current_quarter + 1
            next_year = current_year

        return f"{next_quarter}Q {next_year}"
    
    def get_current_quarter_start():
        now = datetime.now()
        quarter = (now.month - 1) // 3 + 1  # Определяем номер квартала
        date_str = f"{now.year}-{(quarter - 1) * 3 + 1:02d}-01"
        return pd.Timestamp(date_str)  # Конвертируем в Timestamp

    # Функция для получения первого дня квартала
    def get_quarter_start_date(year, quarter):
        return pd.to_datetime(f"{year}-{int(quarter)*3-2}-01")
    
    def convert_to_quarter(date):
        # Получаем номер квартала (1, 2, 3, 4)
        quarter = (date.month - 1) // 3 + 1
        # Формируем строку вида 'Q1 2025'
        return f"Q{quarter} {date.year}"

    # Получаем список кварталов и следующий квартал по умолчанию
    quarters = get_quarters()
    default_quarter = get_next_quarter()

    # Выбор квартала пользователем
    selected_quarter = st.selectbox(
        "Выберите квартал для прогнозирования", 
        quarters, 
        index=quarters.index(default_quarter)  # Устанавливаем следующий квартал по умолчанию
    )

    # Преобразование выбора пользователя в дату
    quarter, year = selected_quarter.split('Q ')
    quarter = int(quarter)
    year = int(year)

    # Определяем start_date и end_date выбранного квартала
    start_date = get_quarter_start_date(year, quarter)  # Первый день выбранного квартала
    end_date = pd.to_datetime(f"{year}-{int(quarter)*3}-30")  # Последний день выбранного квартала

    start_date_strftime = start_date.strftime('%d.%m.%Y')
    end_date_strftime = end_date.strftime('%d.%m.%Y')

    current_quarter = get_current_quarter_start()

    def get_last_completed_quarter_start_date(start_date, current_quarter):
        return min(start_date, current_quarter) - relativedelta(months=3)

    def get_previous_quarter_start_date(start_date, current_quarter):
        return min(start_date, current_quarter)

    last_quarter_end = get_last_completed_quarter_start_date(start_date, current_quarter)
    next_quarter_start = get_previous_quarter_start_date(start_date, current_quarter)

    last_quarter_end_date = last_quarter_end.strftime('%d.%m.%Y')
    next_quarter_start_date = next_quarter_start.strftime('%d.%m.%Y')

    val_quarter = convert_to_quarter(last_quarter_end)

    # Фильтрация данных
    current_df = df[(df['date_trunc'] >= '2022-06-01') & (df['date_trunc'] < start_date)]
    df = get_features(df)

    train = df[(df['date_trunc'] >= '2022-06-01') & (df['date_trunc'] < last_quarter_end)]
    val = df[(df['date_trunc'] >= last_quarter_end) & (df['date_trunc'] < next_quarter_start)]

    # Обучение модели
    filtered_features = ['year', 'month', 'day', 'day_of_week', 'week_of_year', 'day_of_year', 'quarter', 'is_holiday']
    target = ['y']
    X = train[filtered_features]
    y = train[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = XGBRegressor(random_state=42)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    X_forecast = val[filtered_features]
    val['y_forecast'] = model.predict(X_forecast)

    # Визуализация результатов
    val = val[['date_trunc', 'y', 'y_forecast']]
    val.rename(columns={'y': 'y_fact'}, inplace=True)
    val['diff'] = np.abs(val['y_forecast'] - val['y_fact'])

    fact_metric = val['y_fact'].sum()
    forecast_metric = int(val['y_forecast'].sum())
    percentage_diff = np.round((forecast_metric/fact_metric)*100-100, 2)

    def get_percentage_diff_type(x):
        if x >= 0:
            return 'выше'
        elif x < 0:
            return 'ниже'
        
    def get_percentage_diff_summary(x):
        x = abs(x)
        if x >= 75:
            message = 'За последний квартал, данные критически разошлись с фактическими. Поэтому, в случае нашей метрики, на них не стоит обращать внимание.'
            color = "#FF0000"
            return [message, color]
        elif x >= 50:
            message = 'За последний квартал, прогностические данные очень сильно разошлись с фактическими. Поэтому, на них если стоит обращать внимание, то лишь минимально.'
            color = "#FFB6C1"
            return [message, color]
        elif x >= 25:
            message = 'За последний квартал, прогностические данные довольно ощутимо разошлись с фактическими. Доверять стоит с крайней опаской.'
            color = "#FFD700"
            return [message, color]
        elif x >= 10:
            message = 'За последний квартал, прогностические данные незначительно, но разошлись с фактическими. Не смотря на это, прогнозу, в целом, доверять можно.'
            color = "#90EE90"
            return [message, color]
        elif x >= 0:
            message = 'За последний квартал, прогностические данные практически не разошлись с фактическими. Прогнозу можно доверять.'
            color = "#008000"
            return [message, color]
        
    percentage_diff_type = get_percentage_diff_type(percentage_diff)
    percentage_diff_summary = get_percentage_diff_summary(percentage_diff)

    # Визуализация с Plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=val['date_trunc'], y=val['y_fact'], mode='lines', name='Факт'))
    fig.add_trace(go.Scatter(x=val['date_trunc'], y=val['y_forecast'], mode='lines', name='Прогноз'))
    # Настроим заголовки и подписи осей
    fig.update_layout(
        title=f"Факт vs Прогноз ({last_quarter_end_date} - {next_quarter_start_date})", 
        title_font=dict(size=14, color='gray'),  # Размер и цвет заголовка
        xaxis_title="Дата",
        xaxis_title_font=dict(size=12, color='gray'),  # Размер и цвет подписи оси X
        yaxis_title="Значение целевой метрики",
        yaxis_title_font=dict(size=12, color='gray'),  # Размер и цвет подписи оси Y
    )
    st.plotly_chart(fig)

    final_message = f'За срок с {last_quarter_end_date} по {next_quarter_start_date}, прогноcтические значение ({forecast_metric}) оказались {percentage_diff_type} на {abs(percentage_diff)}% относительно фактических ({fact_metric}). {percentage_diff_summary[0]}'
    st.markdown(f"<p style='color:{percentage_diff_summary[1]}; font-size:20px; font-weight:bold;'>{final_message}</p>", unsafe_allow_html=True)

    # Прогнозирование на будущие даты
    future_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    future = pd.DataFrame({'date_trunc': future_dates})
    future_df = get_features(future)
    forecast_future = model.predict(future_df[filtered_features])

    if len(future_dates) != len(forecast_future):
        raise ValueError(f"Количество дат ({len(future_dates)}) не совпадает с количеством данных ({len(forecast_future)})")
    else:
        future_df = pd.DataFrame({
            'date_trunc': future_df['date_trunc'],
            'y_forecast': forecast_future.flatten()
        })

    future_df['month_date'] = future_df['date_trunc'].dt.to_period('M').dt.to_timestamp()

    # Вычисление среднего и стандартного отклонения
    mean_yhat = future_df['y_forecast'].mean() * val.shape[0]
    std_yhat = future_df['y_forecast'].std() * val.shape[0]

    sigma_1_positive = mean_yhat + std_yhat
    sigma_1_negative = mean_yhat - std_yhat

    sigma_2_positive = mean_yhat + 2 * std_yhat
    sigma_2_negative = mean_yhat - 2 * std_yhat

    sigma_3_positive = mean_yhat + 3 * std_yhat
    sigma_3_negative = mean_yhat - 3 * std_yhat

    # Визуализация гистограммы с сигмами
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(x=future_df['y_forecast'] * future_df.shape[0], nbinsx=10, name='Распределение y_forecast'))
    fig_hist.add_vline(x=mean_yhat, line_dash="dash", line_color="red", annotation_text=f"Среднее: {mean_yhat:.2f}")
    #fig_hist.update_layout(title="Гистограмма распределения прогностических значений", xaxis_title="y_forecast", yaxis_title="Частота")
    fig_hist.update_layout(
        title="Гистограмма распределения прогностических значений", 
        title_font=dict(size=14, color='gray'),  # Размер и цвет заголовка
        xaxis_title="y_forecast",
        xaxis_title_font=dict(size=12, color='gray'),  # Размер и цвет подписи оси X
        yaxis_title="Частота",
        yaxis_title_font=dict(size=12, color='gray'),  # Размер и цвет подписи оси Y
    )
    st.plotly_chart(fig_hist)

    st.caption(f"Среднее значение: {mean_yhat:.2f}")
    st.caption(f"Стандартное отклонение: {std_yhat:.2f}")
    st.caption(f"1 сигма: {sigma_1_negative:.2f} (отрицательная), {sigma_1_positive:.2f} (положительная)")
    st.caption(f"2 сигма: {sigma_2_negative:.2f} (отрицательная), {sigma_2_positive:.2f} (положительная)")
    st.caption(f"3 сигма: {sigma_3_negative:.2f} (отрицательная), {sigma_3_positive:.2f} (положительная)")

    def styled_message(text, border_color, text_color, background):
        return f"""
        <div style="
            border: 3px solid {border_color}; 
            color: {text_color};
            padding: 10px; 
            border-radius: 10px; 
            background-color: {background};
            font-size: 18px;
            font-weight: bold;
        ">
            {text}
        </div>
        """
    
    st.markdown(f"### Прогноз ({start_date_strftime} - {end_date_strftime})")
    st.markdown(styled_message(f"Пессимистичный прогноз: {sigma_1_negative:.2f}", "#FF0000", "#8B0000", "#FFEBEB"), unsafe_allow_html=True)
    st.markdown(styled_message(f"Реалистичный прогноз: {mean_yhat:.2f}", "#FFD700", "#8B7500", "#FFF8DC"), unsafe_allow_html=True)
    st.markdown(styled_message(f"Оптимистичный прогноз: {sigma_1_positive:.2f}", "#008000", "#006400", "#E6FFE6"), unsafe_allow_html=True)

    # Объединение данных для графика
    current_df['source'] = 'fact'
    future_df['source'] = 'forecast'
    future_df.rename(columns={'y_forecast': 'y'}, inplace=True)
    combined_df = pd.concat([current_df, future_df]).sort_values(by='date_trunc')

    # Построение графика факта и прогноза
    fig_combined = go.Figure()
    fig_combined.add_trace(go.Scatter(x=combined_df[combined_df['source'] == 'fact']['date_trunc'], 
                             y=combined_df[combined_df['source'] == 'fact']['y'], 
                             mode='lines', name='Факт', line=dict(color='blue')))
    fig_combined.add_trace(go.Scatter(x=combined_df[combined_df['source'] == 'forecast']['date_trunc'], 
                             y=combined_df[combined_df['source'] == 'forecast']['y'], 
                             mode='lines', name='Прогноз', line=dict(color='deepskyblue')))
    #fig_combined.update_layout(title="Факт vs Прогноз", xaxis_title="Дата", yaxis_title="Значение целевой метрики")
    fig_combined.update_layout(
        title="График распределения целевой переменной", 
        title_font=dict(size=14, color='gray'),  # Размер и цвет заголовка
        xaxis_title="Дата",
        xaxis_title_font=dict(size=12, color='gray'),  # Размер и цвет подписи оси X
        yaxis_title="Значение целевой метрики",
        yaxis_title_font=dict(size=12, color='gray'),  # Размер и цвет подписи оси Y
    )
    st.plotly_chart(fig_combined)

    positive_diff = sigma_1_positive/mean_yhat
    negative_diff = (sigma_1_negative/mean_yhat)
    forecast_grouped_df = future_df.groupby(['month_date'], as_index=False).agg(key_metric=('y', 'sum'))
    forecast_grouped_df["Пессимистичный сценарий"] = (forecast_grouped_df['key_metric'] * negative_diff).astype(int)
    forecast_grouped_df["Оптимистичный сценарий"] = (forecast_grouped_df['key_metric'] * positive_diff).astype(int)
    forecast_grouped_df['key_metric'] = (forecast_grouped_df['key_metric']).astype(int)

    # Преобразуем в нужный формат: 'Январь, 2025'
    forecast_grouped_df['month_date'] = forecast_grouped_df['month_date'].dt.strftime('%B, %Y')

    # Создаем словарь с русскими названиями месяцев
    months_in_russian = {
        'January': 'Январь', 'February': 'Февраль', 'March': 'Март', 'April': 'Апрель', 'May': 'Май', 'June': 'Июнь',
        'July': 'Июль', 'August': 'Август', 'September': 'Сентябрь', 'October': 'Октябрь', 'November': 'Ноябрь', 'December': 'Декабрь'
    }

    # Заменяем английские месяцы на русские
    forecast_grouped_df['month_date'] = forecast_grouped_df['month_date'].replace(months_in_russian, regex=True)
    forecast_grouped_df.rename(columns={'key_metric' : 'Реалистичный сценарий', 'month_date' : 'Месяц'}, inplace=True)

    forecast_grouped_df = forecast_grouped_df[['Месяц', 'Пессимистичный сценарий', 'Реалистичный сценарий', 'Оптимистичный сценарий']]
    
    st.dataframe(forecast_grouped_df)
