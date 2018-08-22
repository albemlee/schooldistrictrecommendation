import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import Imputer, MaxAbsScaler, MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from tpot.builtins import OneHotEncoder, StackingEstimator
from xgboost import XGBClassifier

# NOTE: Make sure that the class is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1).values
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'].values, random_state=42)

imputer = Imputer(strategy="median")
imputer.fit(training_features)
training_features = imputer.transform(training_features)
testing_features = imputer.transform(testing_features)

# Score on the training set was:0.9578456717783361
exported_pipeline = make_pipeline(
    make_union(
        StackingEstimator(estimator=make_pipeline(
            MinMaxScaler(),
            StackingEstimator(estimator=XGBClassifier(learning_rate=0.1, max_depth=3, min_child_weight=11, n_estimators=100, nthread=1, subsample=0.15000000000000002)),
            MaxAbsScaler(),
            RandomForestClassifier(bootstrap=False, criterion="entropy", max_features=0.3, min_samples_leaf=5, min_samples_split=15, n_estimators=100)
        )),
        StackingEstimator(estimator=DecisionTreeClassifier(criterion="gini", max_depth=4, min_samples_leaf=19, min_samples_split=14))
    ),
    OneHotEncoder(minimum_fraction=0.25, sparse=False),
    ExtraTreesClassifier(bootstrap=True, criterion="entropy", max_features=0.6500000000000001, min_samples_leaf=1, min_samples_split=5, n_estimators=100)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
