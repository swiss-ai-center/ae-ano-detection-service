from common_code.config import get_settings
from common_code.logger.logger import get_logger, Logger
from common_code.service.models import Service
from common_code.service.enums import ServiceStatus
from common_code.common.enums import FieldDescriptionType, ExecutionUnitTagName, ExecutionUnitTagAcronym
from common_code.common.models import FieldDescription, ExecutionUnitTag
from common_code.tasks.models import TaskData
# Imports required by the service's model
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import io
import os

api_description = """
Anomaly detection of a time series with an autoencoder.
"""
api_summary = """
Anomaly detection of a time series with an autoencoder.
"""
api_title = "Autoencoder Anomaly Detection"
version = "1.0.0"

settings = get_settings()


class MyService(Service):
    """
    Autoencoder Anomaly detection
    """

    # Any additional fields must be excluded for Pydantic to work
    _model: object
    _logger: Logger

    def __init__(self):
        super().__init__(
            name="Autoencoder Anomaly Detection",
            slug="ae-anomaly-detection",
            url=settings.service_url,
            summary=api_summary,
            description=api_description,
            status=ServiceStatus.AVAILABLE,
            data_in_fields=[
                FieldDescription(name="text", type=[FieldDescriptionType.TEXT_CSV, FieldDescriptionType.TEXT_PLAIN]),
            ],
            data_out_fields=[
                FieldDescription(name="result", type=[FieldDescriptionType.IMAGE_PNG]),
            ],
            tags=[
                ExecutionUnitTag(
                    name=ExecutionUnitTagName.ANOMALY_DETECTION,
                    acronym=ExecutionUnitTagAcronym.ANOMALY_DETECTION
                ),
                ExecutionUnitTag(
                    name=ExecutionUnitTagName.TIME_SERIES,
                    acronym=ExecutionUnitTagAcronym.TIME_SERIES
                ),
            ],
            docs_url="https://docs.swiss-ai-center.ch/reference/services/ae-ano-detection/",
            has_ai=True
        )
        self._model = tf.keras.models.load_model(os.path.join(os.path.dirname(__file__), "..", "ae_model.h5"))
        self._logger = get_logger(settings)

    def process(self, data):
        raw = str(data["text"].data)[2:-1]
        raw = raw.replace('\\t', ',').replace('\\n', '\n').replace('\\r', '\n')
        X_test = pd.read_csv(io.StringIO(raw), dtype={"value": np.float64})

        # Use the model to reconstruct the original time series data
        reconstructed_X = self._model.predict(X_test)

        # Calculate the reconstruction error for each point in the time series
        reconstruction_error = np.square(X_test - reconstructed_X).mean(axis=1)

        err = X_test
        fig, ax = plt.subplots(figsize=(20, 6))

        a = err.loc[reconstruction_error >= np.max(reconstruction_error)]  # anomaly

        ax.plot(err, color='blue', label='Normal')

        ax.scatter(a.index, a, color='red', label='Anomaly')
        plt.legend()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)

        # NOTE that the result must be a dictionary with the keys being the field names set in the data_out_fields
        return {
            "result": TaskData(data=buf.read(), type=FieldDescriptionType.IMAGE_PNG)
        }
