# Final-Assignment
### Flight Delay Prediction
##### This repository contains a machine learning model for predicting flight delays based on weather data. The model is designed to help users to predict a flight being delayed due to weather conditions.


import os
from pathlib2 import Path
from zipfile import ZipFile
import time
import pandas as pd
import numpy as np
import subprocess
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score, auc



