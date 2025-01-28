import unittest
import pandas as pd
import os
from ecommerce import EcommerceAnalytics, EDAInsights, LookalikeModel, CustomerSegmentation 


customers_test_data = """CustomerID,CustomerName,Region,SignupDate
C0001,Alice,North America,2023-01-01
C0002,Bob,Europe,2023-01-05
C0003,Charlie,Asia,2023-01-10
"""

products_test_data = """ProductID,ProductName,Category,Price
P001,Product A,Electronics,100
P002,Product B,Books,20
P003,Product C,Clothing,50
"""

transactions_test_data = """TransactionID,CustomerID,ProductID,TransactionDate,Quantity,TotalValue,Price
T001,C0001,P001,2023-01-02,1,100,100
T002,C0001,P002,2023-01-06,2,40,20
T003,C0002,P001,2023-01-07,1,100,100
T004,C0003,P003,2023-01-08,1,50,50
"""

TEST_CUSTOMERS_FILE = 'Customers_test.csv'
TEST_PRODUCTS_FILE = 'Products_test.csv'
TEST_TRANSACTIONS_FILE = 'Transactions_test.csv'
TEST_OUTPUT_FILE = 'test_output.csv'
TEST_EDA_REPORT_FILE = 'test_eda_report.pdf.txt'
TEST_CLUSTERING_REPORT_FILE = 'test_clustering_report.pdf.txt'


# Helper function to create test CSV files
def create_test_files():
    with open(TEST_CUSTOMERS_FILE, 'w') as f:
        f.write(customers_test_data)
    with open(TEST_PRODUCTS_FILE, 'w') as f:
        f.write(products_test_data)
    with open(TEST_TRANSACTIONS_FILE, 'w') as f:
        f.write(transactions_test_data)

def delete_test_files():
    if os.path.exists(TEST_CUSTOMERS_FILE):
        os.remove(TEST_CUSTOMERS_FILE)
    if os.path.exists(TEST_PRODUCTS_FILE):
        os.remove(TEST_PRODUCTS_FILE)
    if os.path.exists(TEST_TRANSACTIONS_FILE):
        os.remove(TEST_TRANSACTIONS_FILE)
    if os.path.exists(TEST_OUTPUT_FILE):
        os.remove(TEST_OUTPUT_FILE)
    if os.path.exists(TEST_EDA_REPORT_FILE):
        os.remove(TEST_EDA_REPORT_FILE)
    if os.path.exists(TEST_CLUSTERING_REPORT_FILE):
        os.remove(TEST_CLUSTERING_REPORT_FILE)


class TestEcommerceAnalytics(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        create_test_files()

    @classmethod
    def tearDownClass(cls):
        delete_test_files()

    def setUp(self):
        self.analytics = EcommerceAnalytics(TEST_CUSTOMERS_FILE, TEST_PRODUCTS_FILE, TEST_TRANSACTIONS_FILE)

    def test_load_data(self):
        self.assertIsInstance(self.analytics.customers_df, pd.DataFrame)
        self.assertIsInstance(self.analytics.products_df, pd.DataFrame)
        self.assertIsInstance(self.analytics.transactions_df, pd.DataFrame)
        self.assertEqual(len(self.analytics.customers_df), 3)
        self.assertEqual(len(self.analytics.products_df), 3)
        self.assertEqual(len(self.analytics.transactions_df), 4)

    def test_load_data_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            EcommerceAnalytics("non_existent_file.csv", TEST_PRODUCTS_FILE, TEST_TRANSACTIONS_FILE)


class TestEDAInsights(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        create_test_files()

    @classmethod
    def tearDownClass(cls):
        delete_test_files()

    def setUp(self):
        self.eda_insights = EDAInsights(TEST_CUSTOMERS_FILE, TEST_PRODUCTS_FILE, TEST_TRANSACTIONS_FILE)

    def test_derive_business_insights(self):
        insights = self.eda_insights.derive_business_insights()
        self.assertIsInstance(insights, list)
        self.assertTrue(len(insights) > 0)

    def test_generate_eda_report(self):
        report_path = TEST_EDA_REPORT_FILE
        self.eda_insights.generate_eda_report(report_path)
        self.assertTrue(os.path.exists(report_path))
        with open(report_path, 'r') as f:
            report_content = f.read()
            self.assertTrue("Business Insights" in report_content)


class TestLookalikeModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        create_test_files()

    @classmethod
    def tearDownClass(cls):
        delete_test_files()

    def setUp(self):
        self.lookalike_model = LookalikeModel(TEST_CUSTOMERS_FILE, TEST_PRODUCTS_FILE, TEST_TRANSACTIONS_FILE)

    def test_create_customer_profiles(self):
        profiles_df = self.lookalike_model._create_customer_profiles()
        self.assertIsInstance(profiles_df, pd.DataFrame)
        self.assertEqual(len(profiles_df), 3)
        self.assertIn('C0001', profiles_df.index)

    def test_get_lookalikes(self):
        lookalikes = self.lookalike_model.get_lookalikes('C0001')
        self.assertIsInstance(lookalikes, list)
        if lookalikes: # check if lookalikes are found in test data
            self.assertEqual(len(lookalikes), min(2, len(self.lookalike_model.customer_profiles)-1)) # should be at max 2 for test data and top 3 requested
            for lookalike in lookalikes:
                self.assertIsInstance(lookalike, tuple)
                self.assertEqual(len(lookalike), 2)
                self.assertIsInstance(lookalike[0], str)
                self.assertIsInstance(lookalike[1], float)

    def test_get_lookalikes_customer_not_found(self):
        lookalikes = self.lookalike_model.get_lookalikes('NonExistentCustomer')
        self.assertEqual(lookalikes, [])

    def test_generate_lookalike_csv(self):
        output_path = TEST_OUTPUT_FILE
        customer_ids_to_check = ['C0001', 'C0002']
        self.lookalike_model.generate_lookalike_csv(output_path, customer_ids_to_check)
        self.assertTrue(os.path.exists(output_path))
        df = pd.read_csv(output_path)
        self.assertIn('CustomerID', df.columns)
        self.assertIn('LookalikeCustomerID', df.columns)
        self.assertEqual(len(df), 4) # 2 customers * top 2 lookalikes (max for this test data)


class TestCustomerSegmentation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        create_test_files()

    @classmethod
    def tearDownClass(cls):
        delete_test_files()

    def setUp(self):
        self.clustering_model = CustomerSegmentation(TEST_CUSTOMERS_FILE, TEST_PRODUCTS_FILE, TEST_TRANSACTIONS_FILE)

    def test_create_clustering_profiles(self):
        profiles_df = self.clustering_model._create_clustering_profiles()
        self.assertIsInstance(profiles_df, pd.DataFrame)
        self.assertEqual(len(profiles_df), 3)
        self.assertIn('C0001', profiles_df.index)

    def test_perform_clustering(self):
        cluster_labels, db_index, silhouette_avg = self.clustering_model.perform_clustering(n_clusters=2)
        self.assertIsInstance(cluster_labels, pd.DataFrame)
        self.assertIn('Cluster', cluster_labels.columns)
        self.assertIsInstance(db_index, float)
        self.assertIsInstance(silhouette_avg, float)

    def test_generate_clustering_report(self):
        report_path = TEST_CLUSTERING_REPORT_FILE
        cluster_labels, db_index, silhouette_avg = self.clustering_model.perform_clustering(n_clusters=2)
        self.clustering_model.generate_clustering_report(report_path, 2, db_index, silhouette_avg)
        self.assertTrue(os.path.exists(report_path))
        with open(report_path, 'r') as f:
            report_content = f.read()
            self.assertTrue("Clustering Report" in report_content)
            self.assertIn("Davies-Bouldin Index", report_content)
            self.assertIn("Silhouette Score", report_content)


if __name__ == '__main__':
    unittest.main()