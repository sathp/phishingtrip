import csv
from textIO import text_to_csv

# Reads file.csv for filepaths
paths = open('file.csv', 'r')

# Driver will output all results to csv file
for year in filepaths:
    for file in year:
        text_to_csv(file, "output.csv")
