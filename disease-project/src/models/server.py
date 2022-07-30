import subprocess
import os
import glob
import sys
sys.path.append("..")
 

list_of_files = glob.glob('mlruns/0/*') # * means all if need specific format then *.csv
latest_file = max(list_of_files, key=os.path.getctime)
print(latest_file)
os.system('mlflow models serve --model-uri {}/artifacts/SVC -p 1234'.format(latest_file))