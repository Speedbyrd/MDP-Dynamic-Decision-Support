"""
Spark MLlib RandomForest wrapper with sklearn-compatible interface.

Replaces sklearn's RandomForestClassifier.fit() and .predict_proba()
with distributed Spark MLlib operations. The wrapper converts pandas
data to Spark DataFrames, trains a distributed RF, and converts
predictions back to numpy arrays — matching sklearn's API so the
existing MDP training algorithm can use it as a drop-in replacement.
"""

import numpy as np
import pandas as pd
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import RandomForestClassifier as SparkRFC
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType


class SparkRFClassifier:
    """Sklearn-compatible wrapper around Spark MLlib RandomForestClassifier.

    Usage:
        clf = SparkRFClassifier(spark_session, random_state=0)
        clf.fit(X_pandas, y_pandas)
        proba = clf.predict_proba(X_pandas)  # returns numpy array
        preds = clf.predict(X_pandas)        # returns numpy array
    """

    def __init__(
        self,
        spark_session,
        random_state=0,
        num_trees=100,
        max_depth=10,
        chunk_size=50000,
        **kwargs,
    ):
        self.spark = spark_session
        self.random_state = random_state
        self.num_trees = num_trees
        self.max_depth = max_depth
        self.chunk_size = chunk_size
        self.model_ = None
        self.classes_ = None
        self._feature_cols = None
        self._label_col = "__label__"
        self._features_vec_col = "__features_vec__"
        self._indexed_label_col = "__label_idx__"

    def _pandas_to_spark(self, pdf):
        """Convert pandas DataFrame to Spark in chunks to avoid serialization limit."""
        if len(pdf) <= self.chunk_size:
            return self.spark.createDataFrame(pdf)
        chunks = [
            pdf.iloc[i : i + self.chunk_size]
            for i in range(0, len(pdf), self.chunk_size)
        ]
        sdfs = [self.spark.createDataFrame(chunk) for chunk in chunks]
        result = sdfs[0]
        for sdf in sdfs[1:]:
            result = result.unionByName(sdf, allowMissingColumns=True)
        return result

    def fit(self, X, y):
        """Train a distributed Random Forest on the full dataset.

        Parameters
        ----------
        X : pandas DataFrame or numpy array
            Feature matrix (n_samples, n_features)
        y : pandas Series/DataFrame or numpy array
            Target labels (n_samples,)

        Returns
        -------
        self
        """
        # Convert inputs to pandas DataFrame if needed
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
        if isinstance(X, pd.DataFrame):
            self._feature_cols = list(X.columns)
        else:
            self._feature_cols = [f"f{i}" for i in range(X.shape[1])]
            X = pd.DataFrame(X, columns=self._feature_cols)

        if isinstance(y, (pd.Series, pd.DataFrame)):
            y_vals = y.values.ravel()
        else:
            y_vals = np.asarray(y).ravel()

        # Store class labels (sorted for consistent ordering)
        self.classes_ = np.sort(np.unique(y_vals))

        # Build a combined pandas DF and convert to Spark
        train_pdf = X.copy()
        train_pdf[self._label_col] = y_vals.astype(float)
        train_sdf = self._pandas_to_spark(train_pdf)

        # Assemble features into a vector column
        assembler = VectorAssembler(
            inputCols=self._feature_cols,
            outputCol=self._features_vec_col,
            handleInvalid="skip",
        )
        train_sdf = assembler.transform(train_sdf)

        # Train Spark RF (distributed across executors)
        rf = SparkRFC(
            featuresCol=self._features_vec_col,
            labelCol=self._label_col,
            predictionCol="__prediction__",
            probabilityCol="__probability__",
            numTrees=self.num_trees,
            maxDepth=self.max_depth,
            seed=self.random_state,
        )
        self.model_ = rf.fit(train_sdf)
        return self

    def predict(self, X):
        """Predict class labels using the trained Spark RF model.

        Parameters
        ----------
        X : pandas DataFrame or numpy array

        Returns
        -------
        numpy array of predicted labels
        """
        if self.model_ is None:
            raise ValueError("Model has not been fitted yet.")

        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self._feature_cols)

        pred_sdf = self._pandas_to_spark(X)
        assembler = VectorAssembler(
            inputCols=self._feature_cols,
            outputCol=self._features_vec_col,
            handleInvalid="skip",
        )
        pred_sdf = assembler.transform(pred_sdf)
        result_sdf = self.model_.transform(pred_sdf)

        # Collect predictions
        preds = np.array(
            result_sdf.select("__prediction__").toPandas()["__prediction__"]
        )
        return preds

    def predict_proba(self, X):
        """Predict class probabilities using the trained Spark RF model.

        Parameters
        ----------
        X : pandas DataFrame or numpy array

        Returns
        -------
        numpy array of shape (n_samples, n_classes) with probability for each class
        """
        if self.model_ is None:
            raise ValueError("Model has not been fitted yet.")

        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self._feature_cols)

        pred_sdf = self._pandas_to_spark(X)
        assembler = VectorAssembler(
            inputCols=self._feature_cols,
            outputCol=self._features_vec_col,
            handleInvalid="skip",
        )
        pred_sdf = assembler.transform(pred_sdf)
        result_sdf = self.model_.transform(pred_sdf)

        # Extract probability vector into individual columns
        n_classes = len(self.classes_)
        for i in range(n_classes):
            result_sdf = result_sdf.withColumn(
                f"__prob_{i}__",
                result_sdf["__probability__"].getItem(i).cast(DoubleType()),
            )
        prob_cols = [f"__prob_{i}__" for i in range(n_classes)]
        proba_pdf = result_sdf.select(prob_cols).toPandas()
        return proba_pdf.values
