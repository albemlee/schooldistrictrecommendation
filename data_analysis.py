# Load required libraries
import pandas as pd
from sklearn.pipeline import Pipeline
from helper import *

# Load raw data into dataframes
finance_data = load_data('finance')
dropout_data = load_data('dropout')
universe_data = load_data('universe')
directory_data = load_data('directory', year='2015')
# print(len(finance_data))
# print(len(dropout_data))
# print(len(universe_data))
# print(len(directory_data))

# Load school demographic data
school_columns = load_columns('school')
demographic_data = aggregate_school_data(school_columns)
# print(len(demographic_data))
demographic_data.to_csv("output/demographic_data.csv", index=False)

# Combine raw data using outer joins
# demographic_data = pd.read_csv("output/demographic_data.csv", dtype='str')
merged_data = merge_data(finance_data, dropout_data, universe_data, demographic_data)
# print(len(merged_data))
merged_data.to_csv("output/merged_data.csv", index=False)

# Wrangle merged data by removing unnecessary column and rows
# merged_data = pd.read_csv("output/merged_data.csv")
wrangled_data = wrangle_data(merged_data, directory_data)
# print(len(wrangled_data))
wrangled_data.to_csv("output/wrangled_data.csv", index=False)

# Rename and reduce columns in wrangled_data
# wrangled_data = pd.read_csv("output/wrangled_data.csv")
wrangled_data_ii = wrangle_data_ii(wrangled_data)
# print(len(wrangled_data_ii))
wrangled_data_ii.to_csv("output/wrangled_data_ii.csv", index=False)

# Prepare dataset for clustering
# wrangled_data_ii = pd.read_csv('output/wrangled_data_ii.csv', dtype='str', na_values='NaN')
X_train, y_train, X_dev, y_dev, X_test, y_test = prep_columns_cluster(wrangled_data_ii)

# Perform clustering
clusters = find_clusters(wrangled_data_ii, X_test, y_test)
clusters.to_csv("output/cluster_results.csv", index=False)

# Save bar chart of clusters
def percentage_under(values):
    return 100 * sum(values)/len(values)
clusters[['cluster', 'label']].groupby('cluster').agg(percentage_under).plot.barh()
plt.xlim(0, 100)
plt.savefig('output/clusters.jpg')

# Prepare dataset for classification
# wrangled_data_ii = pd.read_csv('output/wrangled_data_ii.csv', dtype='str', na_values='NaN')
X_train, y_train, X_dev, y_dev, X_test, y_test = prep_columns_classification(wrangled_data_ii)

# Perform Classification
clf_pipeline, train_score, development_score, test_score = classify_school_districts(X_train, y_train, X_dev, y_dev, X_test, y_test)
