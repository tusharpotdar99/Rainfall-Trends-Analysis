from sklearn.ensemble import IsolationForest
import pandas as pd
from scipy.stats import pearsonr


def anomaly_detection(obj):
    isolation_forest = IsolationForest(contamination=0.05, random_state=42)
    obj.data['Annual_Anomaly'] = isolation_forest.fit_predict(obj.data[['ANNUAL']])
    annual_anomalies = obj.data[obj.data['Annual_Anomaly'] == -1]
    monthly_data = obj.data[['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']]
    monthly_anomalies = isolation_forest.fit_predict(monthly_data)
    monthly_columns = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
    obj.data['Monthly_Anomaly'] = monthly_anomalies
    monthly_anomalies_data = obj.data[obj.data['Monthly_Anomaly'] == -1][['YEAR'] + monthly_columns]
    monthly_anomalies_ = []
    for column in monthly_columns:
        for _, row in monthly_anomalies_data.iterrows():
            monthly_anomalies_.append({'Year': row['YEAR'], 'Month': column, 'Rainfall': row[column]})

    monthly_anomalies_data_long = pd.DataFrame(monthly_anomalies_)
    return obj.data, monthly_columns, annual_anomalies, monthly_data, monthly_anomalies, monthly_anomalies_data, monthly_anomalies_data_long


# identifying drought and extreme rainfall years
def analyze_years(obj):
    mean_rainfall = obj.data['ANNUAL'].mean()
    std_dev_rainfall = obj.data['ANNUAL'].std()

    drought_years = obj.data[obj.data['ANNUAL'] < (mean_rainfall - 1.5 * std_dev_rainfall)]
    extreme_rainfall_years = obj.data[obj.data['ANNUAL'] > (mean_rainfall + 1.5 * std_dev_rainfall)]

    # correlating seasonal rainfall with annual rainfall totals
    seasonal_columns = ['Jan-Feb', 'Mar-May', 'Jun-Sep', 'Oct-Dec']
    seasonal_correlations = {
        season: pearsonr(obj.data[season], obj.data['ANNUAL'])[0] for season in seasonal_columns
    }

    # displaying results for drought/extreme years and correlations
    drought_years_summary = drought_years[['YEAR', 'ANNUAL']].reset_index(drop=True)
    extreme_rainfall_years_summary = extreme_rainfall_years[['YEAR', 'ANNUAL']].reset_index(drop=True)
    seasonal_correlations_summary = pd.DataFrame.from_dict(seasonal_correlations, orient='index',
                                                           columns=['Correlation'])

    return drought_years_summary, extreme_rainfall_years_summary, seasonal_correlations_summary


def season_relationship(obj):
    seasonal_columns = ['Jan-Feb', 'Mar-May', 'Jun-Sep', 'Oct-Dec']
    monsoon_column = 'Jun-Sep'
    relationships = {}

    for season in seasonal_columns:
        if season != monsoon_column:
            corr, _ = pearsonr(obj.data[monsoon_column], obj.data[season])
            relationships[season] = corr

    correlation_data = pd.DataFrame({
        'Season': list(relationships.keys()),
        'Correlation Coefficient': list(relationships.values())
    })
    return seasonal_columns, monsoon_column, relationships, correlation_data
