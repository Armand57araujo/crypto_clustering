# crypto_clustering

## Technology Used 

| Technology Used | Resource URL | 
|------------------|:------------:| 
| Python           | [https://www.python.org/](https://www.python.org/) | 
| Pandas           | [https://pandas.pydata.org/](https://pandas.pydata.org/) | 
| scikit-learn     | [https://scikit-learn.org/](https://scikit-learn.org/) | 
| Matplotlib       | [https://matplotlib.org/](https://matplotlib.org/) | 

## Description 
In this project, I applied the K-means clustering algorithm and principal component analysis (PCA) to classify cryptocurrencies based on their price fluctuations over various timeframes. The analyzed intervals included 24 hours, 7 days, 30 days, 60 days, 200 days, and 1 year. 

The project workflow involved:
1. Preprocessing cryptocurrency market data.
2. Normalizing the data using `StandardScaler` from scikit-learn.
3. Exploring optimal cluster numbers via the Elbow Method and generating an elbow curve.
4. Applying PCA for dimensionality reduction to visualize the results effectively.

## Code Example 

### #1 Preprocessing Data
```python
market_data_df = pd.read_csv("Resources/crypto_market_data.csv", index_col="coin_id")
market_data_df.describe()
```

### #2 Normalizing Data
```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(market_data_df)
market_data_scaled = pd.DataFrame(X_scaled, columns=market_data_df.columns, index=market_data_df.index)
market_data_scaled.head()
```

### #3 Elbow Method for Optimal Clusters
```python
k_values = range(1, 12)
inertia = []

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(market_data_scaled)
    inertia.append(kmeans.inertia_)

elbow_df = pd.DataFrame({'K': list(k_values), 'Inertia': inertia})
elbow_df.plot.line(x="K", y="Inertia", title="Elbow Curve", legend=False)
plt.xlabel("k")
plt.ylabel("Inertia")
plt.show()
```

### #4 PCA for Dimensionality Reduction
```python
pca = PCA(n_components=3)
principal_components = pca.fit_transform(market_data_scaled)
pca_df = pd.DataFrame(data=principal_components, columns=['PCA1', 'PCA2', 'PCA3'], index=market_data_scaled.index)
pca_df.head()
```

## Challenges
Usingthe the Elbow Method and PCA, trying to understand the trade-offs between the concepts of dimensionality reduction while preserving variance was a bit of a challenge. All while reading and comprehending the clustering results in reduced dimensions pushed me really look at the information as thoroughly as possible.

## Learning Points 
This project deepened my understanding of unsupervised machine learning techniques, particularly K-means clustering and PCA. It also honed my skills in data preprocessing, scaling, and visualizing high-dimensional data in a way that supports actionable insights.

## Author Info
**Armand Araujo**  
Location: Las Vegas, NV  

* [LinkedIn](https://www.linkedin.com/in/armand-araujo-a82ba2291/)  
* [Github](https://github.com/Armand57araujo)  

## Credits 
Resources and tools utilized include:
- [W3 Schools](https://www.w3schools.com/)
- [GeeksForGeeks](https://www.geeksforgeeks.org/)
- ChatGPT  
- MDN  
- BCS Support Staff  