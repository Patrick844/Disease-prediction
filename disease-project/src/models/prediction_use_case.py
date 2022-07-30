import sys
sys.path.append("..")
import pandas as pd
import src.models.predict as predict


df = sys.argv[0]
predict.predict(df)