import pandas as pd
import numpy as np

from src.DimondPricePrediction.logger import logging
from src.DimondPricePrediction.exception import customexception

import os
import sys
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from pathlib import Path

class DataIngestionConfig:
    raw_data_path = os.path.join("artifacts","raw.csv")
    train_data_path = os.path.join("artifacts","train.csv")
    test_data_path = os.path.join("artifacts","test.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Data Ingestion Started")

        try:
            data = pd.read_csv(Path(os.path.join("notebooks\data","gemstone.csv")))
            logging.info("Read dataset as Pandas dataframe")

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)
            data.to_csv(self.ingestion_config.raw_data_path,index=False)
            logging.info("Saved the raw dataset in artifacts folder")

            logging.info("Performing train test split")
            train_data,test_data = train_test_split(data,test_size=0.25)
            logging.info("Train test split completed")

            train_data.to_csv(self.ingestion_config.train_data_path,index=False)
            test_data.to_csv(self.ingestion_config.test_data_path,index=False)

            logging.info("Data Ingestion completed")

            return(self.ingestion_config.train_data_path,self.ingestion_config.test_data_path)

        except Exception as e:
            logging.info("Exception occurred during data ingestion stage")
            raise customexception(e,sys)
        




