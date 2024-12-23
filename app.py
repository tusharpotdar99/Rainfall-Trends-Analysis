import streamlit as st
from src.data_preprocessing import data_preprocess
from src.exploratory_data_analysis import trends_in_annual_rainfall, trends_in_monthly_rainfall, \
    trends_in_seasonal_rainfall, rolling_avg, anomalous_rainfall_years, season_correlation_rainfall, \
    cluster_rainfall_years
from src.model_training import model_training, forecast_plot
# CSV file path
path = 'data/rainfall_area-wt_India_1901-2015.csv'
obj = data_preprocess(path)


# Function for Home Page
def home_page():
    st.title("Rainfall Trend Analysis")
    st.markdown("""
    Welcome to the **Rainfall Trend Analysis** App. This tool provides insights into historical rainfall trends across India.
    Use the **Data Exploration** section to view and analyze the dataset.
    """)


# Function for Data Exploration
def data_exploration():
    return obj.data


def statistical_analysis():
    return obj.month_stats, obj.season_stats, obj.mean_rainfall, obj.seasonal_avg, obj.seasonal_correlations, obj.drought_years, obj.std_dev_rainfall, obj.mean_rainfall, obj.extreme_rainfall_years, obj.seasonal_avg, obj.lowest_rainfall_month, obj.highest_rainfall_month, obj.monthly_avg


# Function to create the Plotly chart
def create_rainfall_trend_chart(obj):
    annual_rainfall_trends = trends_in_annual_rainfall(obj)
    monthly_rainfall_trends = trends_in_monthly_rainfall(obj)
    seasonal_rainfall_trends = trends_in_seasonal_rainfall(obj)
    return annual_rainfall_trends, monthly_rainfall_trends, seasonal_rainfall_trends


def rolling_average(obj):
    figure_avg = rolling_avg(obj)
    return figure_avg


def anomalous_detection(obj):
    fig_annual, fig_month, annul_, month_ = anomalous_rainfall_years(obj)
    return fig_annual, fig_month, annul_, month_


def season_correlation():
    season_corr = season_correlation_rainfall()


def clusters():
    groups = cluster_rainfall_years()


# Set up Streamlit Sidebar Navigation
st.sidebar.title("Navigation Bar")
page = st.sidebar.radio("Select a page", ["Home", "Data Representation", "Rainfall Trends", "Rolling Average", "Anomalous Rainfall Years", 'Forecasting Rainfall for 20 Years'])

# Display content based on page selection
if page == "Home":
    home_page()

elif page == "Data Representation":
    st.subheader("Explore the Dataset")
    data = data_exploration()

    if data is not None and not data.empty:
        st.dataframe(data)
        st.write("Fields   : ", data.shape[0])
        st.write("Features : ", data.shape[1])

elif page == 'Rainfall Trends':
    st.subheader("Rainfall Trend")
    annual_rainfall_trends, monthly_rainfall_trends, seasonal_rainfall_trends = create_rainfall_trend_chart(obj)
    st.subheader("Annual Rainfall Trend")
    st.plotly_chart(annual_rainfall_trends)

    st.subheader("Monthly Rainfall Trend")
    st.plotly_chart(monthly_rainfall_trends)

    st.subheader("Seasonal Rainfall Trend")
    st.plotly_chart(seasonal_rainfall_trends)

elif page == 'Rolling Average':
    st.subheader("Rainfall Rolling Average for 10 Years")
    r_avg = rolling_average(obj)
    st.plotly_chart(r_avg)

elif page == 'Anomalous Rainfall Years':
    st.header("Anomalous Rainfall Years")
    fig_annual, fig_month, annul_, month_ = anomalous_detection(obj)
    st.subheader("Annual Rainfall Anomalies")
    st.plotly_chart(fig_annual)
    st.subheader("Monthly Rainfall Anomalies")
    st.plotly_chart(fig_month)
    st.table(annul_)

elif page == 'Forecasting Rainfall for 20 Years':
    st.header('Rainfall Forecasting for 20 Years')
    forecast, prophet_model = model_training(obj)
    forecast_figure = forecast_plot(prophet_model, forecast)
    st.plotly_chart(forecast_figure)