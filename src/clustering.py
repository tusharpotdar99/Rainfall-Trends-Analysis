from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def clustering(obj):
    rainfall_features = obj.data[['Jan-Feb', 'Mar-May', 'Jun-Sep', 'Oct-Dec', 'ANNUAL']]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(rainfall_features)
    kmeans = KMeans(n_clusters=3, random_state=42)
    obj.data['Rainfall_Cluster'] = kmeans.fit_predict(scaled_features)
    cluster_labels = {0: 'Dry', 1: 'Normal', 2: 'Wet'}
    obj.data['Rainfall_Category'] = obj.data['Rainfall_Cluster'].map(cluster_labels)
    return obj.data