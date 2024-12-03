import numpy as np
import tensorflow as tf
from loguru import logger
from collections import Counter
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    recall_score,
    precision_score,
    roc_curve,
    precision_recall_curve,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)
from tensorflow.keras.preprocessing.image import NumpyArrayIterator


class ModelTester:
    def __init__(
        self,
        test: NumpyArrayIterator,
        y_test: np.ndarray,
        model: tf.keras.Model,
    ):
        self.test_dataset = test
        self.model = model
        self.y_test_prob = y_test

    def test(self):
        logger.info("Start testing")
        y_pred_prob = self.model.predict(self.test_dataset)
        y_pred = np.argmax(y_pred_prob, axis=1)
        y_test = np.argmax(self.y_test_prob, axis=1)
        logger.info(f"y_pred_prob: {y_pred_prob}")
        logger.info(f"y_test_prob: {self.y_test_prob}")
        logger.info(f"y_pred: {y_pred}")
        logger.info(f"y_test: {y_test}")

        # self.cm, self.rec, self.pr, self.acc = self.model_evaluate(y_test, y_pred)
        # self.log_metrics(model_name)
        # self.log_graphics(y_test, y_pred_prob, model_name)
        logger.info("Finish testing")

    def model_evaluate(self, y_test: np.ndarray, y_pred: np.ndarray):
        cm = confusion_matrix(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        pr = precision_score(y_test, y_pred)
        acc = accuracy_score(y_test, y_pred)
        return cm, rec, pr, acc

    def log_metrics(self, model_name: str):
        logger.info(f"{model_name} accuracy:{self.acc}")
        logger.info(f"{model_name} precision:{self.pr}")
        logger.info(f"{model_name} recall:{self.rec}")

    def log_graphics(
        self,
        y_test: np.ndarray,
        y_pred_prob: np.ndarray,
    ):
        precision, recall, thresholds = precision_recall_curve(y_test, y_pred_prob)
        pr_curve = PrecisionRecallDisplay(precision=precision, recall=recall).plot()
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
        roc_auc_curve = RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
        disp = ConfusionMatrixDisplay(confusion_matrix=self.cm)
        disp.plot()
        # log_figure(pr_curve.figure_, f"{model_name}_PR_curve.png")
        # log_figure(roc_auc_curve.figure_, f"{model_name}_ROC_curve.png")
        # log_figure(disp.figure_, f"{model_name}_confusion_matrix.png")
