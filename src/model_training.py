import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly
import pickle as pkl


def model_training(obj):
    obj.data['DATE'] = pd.to_datetime(obj.data['YEAR'], format='%Y')
    annual_rainfall_ts = obj.data.set_index('DATE')['ANNUAL']
    prophet_data = annual_rainfall_ts.reset_index()
    prophet_data.columns = ['ds', 'y']
    prophet_model = Prophet()
    prophet_model.fit(prophet_data)
    with open('models/profet_model.pkl', 'wb') as file:
        pkl.dump(prophet_model, file)
    print('model saved successfully')
    future = prophet_model.make_future_dataframe(periods=20, freq='Y')
    forecast = prophet_model.predict(future)
    return forecast, prophet_model


def forecast_plot(prophet_model, forecast):
    fig_forecast = plot_plotly(prophet_model, forecast)
    fig_forecast.update_layout(
        title='Annual Rainfall Forecast Using Prophet',
        xaxis_title='Year',
        yaxis_title='Rainfall (mm)',
        template='plotly_white',
        height=500
    )

    return fig_forecast
