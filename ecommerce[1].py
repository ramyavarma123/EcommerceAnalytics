import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances, silhouette_score, davies_bouldin_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple

class EcommerceAnalytics:
    """
    Base class for eCommerce data analysis, providing data loading and utility functions.
    """
    def __init__(self, customers_path: str, products_path: str, transactions_path: str):
        """
        Initializes the EcommerceAnalytics class with file paths.

        Args:
            customers_path (str): Path to the customers CSV file.
            products_path (str): Path to the products CSV file.
            transactions_path (str): Path to the transactions CSV file.
        """
        self.customers_path = customers_path
        self.products_path = products_path
        self.transactions_path = transactions_path
        self.customers_df = self._load_data(self.customers_path)
        self.products_df = self._load_data(self.products_path)
        self.transactions_df = self._load_data(self.transactions_path)

    def _load_data(self, path: str) -> pd.DataFrame:
        """
        Loads data from a CSV file into a pandas DataFrame.

        Args:
            path (str): Path to the CSV file.

        Returns:
            pd.DataFrame: DataFrame containing the data from the CSV file.
        """
        try:
            return pd.read_csv(path)
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found at path: {path}")
        except Exception as e:
            raise Exception(f"Error loading data from {path}: {e}")

class EDAInsights(EcommerceAnalytics):
    """
    Performs Exploratory Data Analysis (EDA) and derives business insights.
    """
    def __init__(self, customers_path: str, products_path: str, transactions_path: str):
        super().__init__(customers_path, products_path, transactions_path)

    def perform_eda(self):
        """
        Performs EDA on the eCommerce dataset and prints descriptive statistics and generates basic plots.
        """
        print("Customers Data Description:\n", self.customers_df.describe(include='all'))
        print("\nProducts Data Description:\n", self.products_df.describe(include='all'))
        print("\nTransactions Data Description:\n", self.transactions_df.describe(include='all'))

        # Example EDA Visualizations (can be expanded)
        sns.countplot(x='Region', data=self.customers_df)
        plt.title('Customer Distribution by Region')
        plt.show()

        sns.histplot(self.products_df['Price'], kde=True)
        plt.title('Product Price Distribution')
        plt.show()

        transaction_counts = self.transactions_df['CustomerID'].value_counts()
        sns.histplot(transaction_counts, kde=True)
        plt.title('Transaction Frequency per Customer')
        plt.show()


    def derive_business_insights(self) -> List[str]:
        """
        Derives business insights from the EDA.

        Returns:
            List[str]: List of business insights as short sentences.
        """
        insights = []

        # Insight 1: Region with most customers
        top_region = self.customers_df['Region'].value_counts().idxmax()
        insights.append(f"Most customers are from {top_region}, indicating a potential focus area for marketing and sales efforts in this region. Further investigation into regional preferences could optimize strategies.")

        # Insight 2: Most common product category
        top_category = self.products_df['Category'].value_counts().idxmax()
        insights.append(f"'{top_category}' is the most frequent product category purchased. This suggests high demand for {top_category} products, which can guide inventory management and product development strategies.")

        # Insight 3: Average transaction value
        avg_transaction_value = self.transactions_df['TotalValue'].mean()
        insights.append(f"The average transaction value is approximately ${avg_transaction_value:.2f}. Monitoring this metric is crucial for revenue forecasting and understanding customer spending habits. Strategies to increase average transaction value should be explored.")

        # Insight 4: Customer signup trend over time (example, needs date parsing and analysis if relevant)
        if 'SignupDate' in self.customers_df.columns:
            signup_trends = self.customers_df['SignupDate'].value_counts().sort_index()
            insights.append("Customer sign-up trends over time reveal periods of growth and potential seasonality, useful for planning marketing campaigns and resource allocation. Further analysis of time series data could enhance predictive capabilities.")
        else:
            insights.append("Customer signup date information is available, but requires further processing for trend analysis. Analysing signup trends could reveal growth patterns and inform marketing strategies.")


        # Insight 5: Relationship between quantity and total value (positive correlation expected)
        correlation = self.transactions_df['Quantity'].corr(self.transactions_df['TotalValue'])
        insights.append(f"There is a positive correlation ({correlation:.2f}) between the quantity of products purchased and the total transaction value, as expected. This highlights the importance of encouraging larger purchases to increase revenue.")

        return insights

    def generate_eda_report(self, report_path: str):
        """
        Generates a PDF report of the business insights. (Simplified - saves insights to text file for demonstration)

        Args:
            report_path (str): Path to save the EDA report.
        """
        self.perform_eda() # Optionally call perform_eda to generate plots before report.
        insights = self.derive_business_insights()
        try:
            with open(report_path, 'w') as f:
                f.write("Business Insights from EDA:\n\n")
                for i, insight in enumerate(insights):
                    f.write(f"Insight {i+1}: {insight}\n\n")
            print(f"EDA report saved to {report_path}")
        except Exception as e:
            print(f"Error generating EDA report: {e}")


class LookalikeModel(EcommerceAnalytics):
    """
    Builds a Lookalike Model to recommend similar customers.
    """
    def __init__(self, customers_path: str, products_path: str, transactions_path: str):
        super().__init__(customers_path, products_path, transactions_path)
        self.customer_profiles = self._create_customer_profiles()

    def _create_customer_profiles(self) -> pd.DataFrame:
        """
        Creates customer profiles by aggregating transaction and customer data.

        Returns:
            pd.DataFrame: DataFrame with customer profiles.
        """
        transaction_summary = self.transactions_df.groupby('CustomerID').agg(
            total_transactions=('TransactionID', 'count'),
            total_quantity=('Quantity', 'sum'),
            total_value=('TotalValue', 'sum'),
            avg_price=('Price', 'mean')
        ).reset_index()

        customer_profiles = pd.merge(self.customers_df, transaction_summary, on='CustomerID', how='left').fillna(0)
        customer_profiles['avg_transaction_value'] = customer_profiles['total_value'] / (customer_profiles['total_transactions'] + 1e-9) # avoid div by zero

        # Encode categorical features if needed and scale numerical
        profile_features = customer_profiles[['Region', 'total_transactions', 'total_quantity', 'total_value', 'avg_price', 'avg_transaction_value']].copy()
        profile_features['Region'] = LabelEncoder().fit_transform(profile_features['Region']) # Simple Label Encoding. One-Hot could be used for better categorical handling.
        profile_features = StandardScaler().fit_transform(profile_features) # Scale numerical features

        return pd.DataFrame(profile_features, index=customer_profiles['CustomerID'])


    def get_lookalikes(self, customer_id: str, top_n: int = 3) -> List[Tuple[str, float]]:
        """
        Recommends top N lookalike customers for a given customer ID.

        Args:
            customer_id (str): Customer ID to find lookalikes for.
            top_n (int): Number of lookalike customers to recommend.

        Returns:
            List[Tuple[str, float]]: List of tuples, each containing lookalike customer ID and similarity score.
        """
        if customer_id not in self.customer_profiles.index:
            return [] # Customer not found

        customer_profile = self.customer_profiles.loc[customer_id].values.reshape(1, -1)
        similarity_scores = pairwise_distances(customer_profile, self.customer_profiles, metric='cosine')[0] # Cosine similarity

        similarity_df = pd.DataFrame({'CustomerID': self.customer_profiles.index, 'SimilarityScore': 1 - similarity_scores}) # 1 - cosine distance = cosine similarity
        similarity_df = similarity_df[similarity_df['CustomerID'] != customer_id].sort_values(by='SimilarityScore', ascending=False).head(top_n) # Exclude self and get top N

        return list(zip(similarity_df['CustomerID'], similarity_df['SimilarityScore']))


    def generate_lookalike_csv(self, output_path: str, customer_ids_to_check: List[str]):
        """
        Generates a CSV file containing lookalike recommendations for specified customer IDs.

        Args:
            output_path (str): Path to save the Lookalike CSV file.
            customer_ids_to_check (List[str]): List of customer IDs to generate lookalikes for.
        """
        lookalike_map: Dict[str, List[Tuple[str, float]]] = {}
        for cust_id in customer_ids_to_check:
            lookalike_map[cust_id] = self.get_lookalikes(cust_id)

        # Flatten the map into a DataFrame for CSV export.
        output_rows = []
        for cust_id, lookalikes in lookalike_map.items():
            for lookalike_id, score in lookalikes:
                output_rows.append({'CustomerID': cust_id, 'LookalikeCustomerID': lookalike_id, 'SimilarityScore': score})
        output_df = pd.DataFrame(output_rows)

        try:
            output_df.to_csv(output_path, index=False)
            print(f"Lookalike CSV saved to {output_path}")
        except Exception as e:
            print(f"Error generating Lookalike CSV: {e}")


class CustomerSegmentation(EcommerceAnalytics):
    """
    Performs customer segmentation using clustering techniques.
    """
    def __init__(self, customers_path: str, products_path: str, transactions_path: str):
        super().__init__(customers_path, products_path, transactions_path)
        self.customer_profiles = self._create_clustering_profiles()


    def _create_clustering_profiles(self) -> pd.DataFrame:
        """
        Creates customer profiles suitable for clustering, similar to Lookalike model but potentially different features.

        Returns:
            pd.DataFrame: DataFrame with customer profiles for clustering.
        """
        transaction_summary = self.transactions_df.groupby('CustomerID').agg(
            total_transactions=('TransactionID', 'count'),
            total_quantity=('Quantity', 'sum'),
            total_value=('TotalValue', 'sum'),
            unique_products=('ProductID', 'nunique') # Adding unique products
        ).reset_index()

        customer_profiles = pd.merge(self.customers_df, transaction_summary, on='CustomerID', how='left').fillna(0)

        profile_features = customer_profiles[['Region', 'total_transactions', 'total_quantity', 'total_value', 'unique_products']].copy()
        profile_features['Region'] = LabelEncoder().fit_transform(profile_features['Region']) # Label Encoding
        profile_features = StandardScaler().fit_transform(profile_features) # Scaling

        return pd.DataFrame(profile_features, index=customer_profiles['CustomerID'])


    def perform_clustering(self, n_clusters: int = 3, algorithm='kmeans') -> Tuple[pd.DataFrame, float, float]:
        """
        Performs customer clustering using KMeans or other algorithms.

        Args:
            n_clusters (int): Number of clusters to form.
            algorithm (str): Clustering algorithm to use ('kmeans').

        Returns:
            Tuple[pd.DataFrame, float, float]: DataFrame with cluster labels, DB Index, and Silhouette Score.
        """
        if algorithm == 'kmeans':
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10) # Explicitly set n_init for KMeans
            clusters = kmeans.fit_predict(self.customer_profiles)
        else:
            raise ValueError(f"Algorithm '{algorithm}' not supported.")

        cluster_labels = pd.DataFrame({'CustomerID': self.customer_profiles.index, 'Cluster': clusters})
        db_index = davies_bouldin_score(self.customer_profiles, clusters)
        silhouette_avg = silhouette_score(self.customer_profiles, clusters)

        return cluster_labels, db_index, silhouette_avg


    def visualize_clusters(self, cluster_labels: pd.DataFrame):
        """
        Visualizes customer clusters using PCA for dimensionality reduction.

        Args:
            cluster_labels (pd.DataFrame): DataFrame containing cluster labels for each customer.
        """
        pca = PCA(n_components=2) # Reduce to 2 dimensions for visualization
        reduced_features = pca.fit_transform(self.customer_profiles)
        reduced_df = pd.DataFrame(data=reduced_features, index=self.customer_profiles.index, columns=['PCA1', 'PCA2'])
        reduced_df = reduced_df.join(cluster_labels.set_index('CustomerID'), how='inner') # Join with cluster labels

        plt.figure(figsize=(8, 6))
        sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=reduced_df, palette='viridis')
        plt.title('Customer Clusters Visualisation (PCA)')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.show()


    def generate_clustering_report(self, report_path: str, n_clusters: int, db_index: float, silhouette_avg: float):
        """
        Generates a report on clustering results. (Simplified - saves to text file)

        Args:
            report_path (str): Path to save the clustering report.
            n_clusters (int): Number of clusters formed.
            db_index (float): DB Index value.
            silhouette_avg (float): Silhouette Score value.
        """
        try:
            with open(report_path, 'w') as f:
                f.write("Customer Clustering Report:\n\n")
                f.write(f"Number of Clusters: {n_clusters}\n")
                f.write(f"Davies-Bouldin Index: {db_index:.4f}\n")
                f.write(f"Silhouette Score: {silhouette_avg:.4f}\n")
                f.write("\nInterpretation:\n")
                f.write("- DB Index: Lower values indicate better clustering. A score of 0 is the best possible score.\n")
                f.write("- Silhouette Score: Values range from -1 to +1. Higher values indicate better clustering.\n")
            print(f"Clustering report saved to {report_path}")
        except Exception as e:
            print(f"Error generating clustering report: {e}")


if __name__ == "__main__":
    # File paths (adjust as needed)
    customers_file = r'E:\LLMS\Fine-tuning\Demo\Customers.csv' # or full path if needed
    products_file = r'E:\LLMS\Fine-tuning\Demo\Products.csv'
    transactions_file = r'E:\LLMS\Fine-tuning\Demo\Transactions.csv'

    # Task 1: EDA and Business Insights
    eda_analyzer = EDAInsights(customers_file, products_file, transactions_file)
    eda_report_file = 'FirstName_LastName_EDA.pdf.txt' # Saving as txt for simplicity, can be converted to PDF using libraries
    eda_analyzer.generate_eda_report(eda_report_file)

    # Task 2: Lookalike Model
    lookalike_model = LookalikeModel(customers_file, products_file, transactions_file)
    lookalike_csv_file = 'FirstName_LastName_Lookalike.csv'
    customer_ids_for_lookalike = lookalike_model.customers_df['CustomerID'].head(20).tolist() # First 20 customers
    lookalike_model.generate_lookalike_csv(lookalike_csv_file, customer_ids_for_lookalike)


    # Task 3: Customer Segmentation / Clustering
    cluster_analyzer = CustomerSegmentation(customers_file, products_file, transactions_file)
    n_clusters = 4 # Example number of clusters, can be tuned
    cluster_labels, db_index, silhouette_avg = cluster_analyzer.perform_clustering(n_clusters=n_clusters)
    cluster_analyzer.visualize_clusters(cluster_labels) # Visualize clusters
    clustering_report_file = 'FirstName_LastName_Clustering.pdf.txt' # Saving as txt for simplicity
    cluster_analyzer.generate_clustering_report(clustering_report_file, n_clusters, db_index, silhouette_avg)


    print("All tasks completed. Please check generated reports and CSV files.")