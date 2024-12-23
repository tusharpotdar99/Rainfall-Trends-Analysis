import plotly.graph_objects as go
import plotly.express as px
from src.data_preprocessing import data_preprocess
from src.anomaly_detection import anomaly_detection, season_relationship
from src.clustering import clustering


def object_creation(file_path):
    obj = data_preprocess(file_path)
    seasonal_correlations = obj.seasonal_correlations
    drought_years = obj.drought_years
    std_dev_rainfall = obj.std_dev_rainfall
    mean_rainfall = obj.mean_rainfall
    extreme_rainfall_years = obj.extreme_rainfall_years
    seasonal_avg = obj.seasonal_avg
    lowest_rainfall_month = obj.lowest_rainfall_month
    highest_rainfall_month = obj.highest_rainfall_month
    monthly_avg = obj.monthly_avg
    data = obj.data
    month_stats = obj.month_stats
    season_stats = obj.season_stats
    print(f"""
    \ndata                    : \n{data.head()}
    \ndrought_years           : {drought_years}
    \nstd_dev_rainfall        : {std_dev_rainfall}
    \nmean_rainfall           : {mean_rainfall}
    \nextreme_rainfall_years  : {extreme_rainfall_years}
    \nseasonal_avg            : {seasonal_avg}
    \nlowest_rainfall_month   : {lowest_rainfall_month}
    \nhighest_rainfall_month  : {highest_rainfall_month}
    \nmonthly_avg             : {monthly_avg}
    \nmonth_stats             : {month_stats}
    \nseason_stats            : {season_stats}
    \nseasonal_correlation    : {seasonal_correlations}
    """)
    return obj


def trends_in_annual_rainfall(obj):
    annual_rainfall = obj.data[['YEAR', 'ANNUAL']]
    fig_annual = go.Figure()
    fig_annual.add_trace(go.Scatter(
        x=annual_rainfall['YEAR'],
        y=annual_rainfall['ANNUAL'],
        mode='lines',
        name='Annual Rainfall',
        line=dict(color='blue', width=2),
        opacity=0.7
    ))
    fig_annual.add_trace(go.Scatter(
        x=annual_rainfall['YEAR'],
        y=[annual_rainfall['ANNUAL'].mean()] * len(annual_rainfall),
        mode='lines',
        name='Main Rainfall',
        line=dict(color='red', dash='dash')
    ))

    fig_annual.update_layout(
        title='Trend in Annual Rainfall in India (1901-2015)',
        xaxis_title='Year',
        yaxis_title='Rainfall (mm)',
        template='plotly_white',
        legend=dict(title="Legend"),
        height=500
    )

    return fig_annual


def trends_in_monthly_rainfall(obj):
    fig_monthly = px.bar(
        x=obj.monthly_avg.index,
        y=obj.monthly_avg.values,
        labels={'x': 'Month', 'y': 'Rainfall (mm)'},
        title='Average Monthly Rainfall in India (1901-2015)',
        text=obj.monthly_avg.values
    )
    fig_monthly.add_hline(
        y=obj.monthly_avg.mean(),
        line_dash="dash",
        line_color="red",
        annotation_text="Mean Rainfall",
        annotation_position="top right"
    )
    fig_monthly.update_traces(marker_color='skyblue', marker_line_color='black', marker_line_width=1)
    fig_monthly.update_layout(template='plotly_white', height=500)
    return fig_monthly


def trends_in_seasonal_rainfall(obj):
    fig_seasonal = px.bar(
        x=obj.seasonal_avg.index,
        y=obj.seasonal_avg.values,
        labels={'x': 'Season', 'y': 'Rainfall (mm)'},
        title='Seasonal Rainfall Distribution in India (1901-2015)',
        text=obj.seasonal_avg.values,
        color=obj.seasonal_avg.values,
        color_continuous_scale=['gold', 'skyblue', 'green', 'orange']
    )

    fig_seasonal.update_traces(marker_line_color='black', marker_line_width=1)
    fig_seasonal.update_layout(
        template='plotly_white',
        height=500,
        coloraxis_colorbar=dict(title='mm')
    )
    fig_seasonal.update_layout(template='plotly_white', height=500)
    return fig_seasonal


def rolling_avg(obj):
    fig_climate_change = go.Figure()

    fig_climate_change.add_trace(go.Scatter(
        x=obj.data['YEAR'],
        y=obj.data['ANNUAL'],
        mode='lines',
        name='Annual Rainfall',
        line=dict(color='blue', width=2),
        opacity=0.6
    ))

    fig_climate_change.add_trace(go.Scatter(
        x=obj.data['YEAR'],
        y=obj.data['10-Year Rolling Avg'],
        mode='lines',
        name='10-Year Rolling Avg',
        line=dict(color='red', width=3)
    ))

    fig_climate_change.update_layout(
        title='Impact of Climate Change on Rainfall Patterns (1901-2015)',
        xaxis_title='Year',
        yaxis_title='Rainfall (mm)',
        template='plotly_white',
        legend=dict(title="Legend"),
        height=500
    )

    return fig_climate_change


def anomalous_rainfall_years(obj):
    data, monthly_columns, annual_anomalies, monthly_data, monthly_anomalies, monthly_anomalies_data, monthly_anomalies_data_long = anomaly_detection(obj)
    print(monthly_anomalies_data_long)
    fig_annual_anomalies = go.Figure()
    fig_annual_anomalies.add_trace(go.Scatter(
        x=data['YEAR'],
        y=data['ANNUAL'],
        mode='lines',
        name='Annual Rainfall',
        line=dict(color='blue', width=2),
        opacity=0.6
    ))

    fig_annual_anomalies.add_trace(go.Scatter(
        x=annual_anomalies['YEAR'],
        y=annual_anomalies['ANNUAL'],
        mode='markers',
        name='Anomalous Years',
        marker=dict(color='red', size=8, symbol='circle')
    ))

    fig_annual_anomalies.add_hline(
        y=data['ANNUAL'].mean(),
        line_dash='dash',
        line_color='green',
        annotation_text='Mean Rainfall',
        annotation_position='bottom right'
    )

    fig_annual_anomalies.update_layout(
        title='Annual Rainfall Anomalies in India (1901-2015)',
        xaxis_title='Year',
        yaxis_title='Rainfall (mm)',
        template='plotly_white',
        legend=dict(title="Legend"),
        height=500
    )
    fig_monthly_anomalies = px.line(
        data,
        x='YEAR',
        y=monthly_columns,
        labels={'YEAR': 'Year', 'value': 'Rainfall (mm)', 'variable': 'Month'},
        title='Monthly Rainfall Anomalies in India (1901-2015)',
        color_discrete_sequence=px.colors.qualitative.Set3
    )

    fig_monthly_anomalies.add_trace(go.Scatter(
        x=monthly_anomalies_data_long['Year'],
        y=monthly_anomalies_data_long['Rainfall'],
        mode='markers',
        name='Anomalous Months',
        marker=dict(color='red', size=5, symbol='circle')
    ))

    fig_monthly_anomalies.update_layout(
        template='plotly_white',
        legend=dict(title="Legend"),
        height=500
    )

    return fig_annual_anomalies, fig_monthly_anomalies, annual_anomalies, monthly_anomalies


def season_correlation_rainfall(obj):
    seasonal_columns, monsoon_column, relationships, correlation_data = season_relationship(obj)
    fig = px.bar(
        correlation_data,
        x='Season',
        y='Correlation Coefficient',
        title='Correlation Between Monsoon (Jun-Sep) Rainfall and Other Seasons',
        labels={'Season': 'Season', 'Correlation Coefficient': 'Correlation Coefficient'},
        text='Correlation Coefficient',
        color='Correlation Coefficient',
        color_continuous_scale='Blues'
    )

    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color="red",
        annotation_text="No Correlation",
        annotation_position="bottom left"
    )

    fig.update_traces(marker_line_color='black', marker_line_width=1, texttemplate='%{text:.2f}')
    fig.update_layout(
        template='plotly_white',
        height=500
    )

    return fig


def cluster_rainfall_years(obj):
    data = clustering(obj)
    fig = px.scatter(
        data,
        x='YEAR',
        y='ANNUAL',
        color='Rainfall_Category',
        title='Clustering of Years Based on Rainfall Patterns',
        labels={'YEAR': 'Year', 'ANNUAL': 'Annual Rainfall (mm)', 'Rainfall_Category': 'Rainfall Category'},
        color_discrete_sequence=px.colors.qualitative.Set2,
        hover_data={'Rainfall_Cluster': True, 'Rainfall_Category': True}
    )

    fig.update_layout(
        template='plotly_white',
        legend_title='Rainfall Category',
        height=500
    )

    return fig


if __name__ == '__main__':
    path = '../data/rainfall_area-wt_India_1901-2015.csv'
    obj = object_creation(path)
    anomalous_rainfall_years(obj)
