import numpy as np
import pandas as pd
from sklearn.kernel_approximation import Nystroem
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import Imputer
from tpot.builtins import StackingEstimator
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

# Score on the training set was:1.0
exported_pipeline = make_pipeline(
    make_union(
        make_pipeline(
            Nystroem(gamma=0.9, kernel="poly", n_components=4),
            StackingEstimator(estimator=BernoulliNB(alpha=1.0, fit_prior=False)),
            Nystroem(gamma=0.9500000000000001, kernel="polynomial", n_components=1)
        ),
        StackingEstimator(estimator=XGBClassifier(learning_rate=1.0, max_depth=7, min_child_weight=11, n_estimators=100, nthread=1, subsample=0.9000000000000001))
    ),
    GaussianNB()
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
