import glob
import csv
from textIO import text_to_csv

# Testing out glob to recursively crawl through e-mail directory
years = []
for filename in glob.glob('spamdata_untroubled/*/', recursive=True):
    years.append(filename);


filepaths = []
for x in years:
    paths = []
    for filename in glob.glob(x +'**/*.txt', recursive=True):
        paths.append(filename);
    filepaths.append(paths)
    # Writing filepaths to .csv file
    csv_file = open('file.csv', 'a')
    with csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(paths)


# Driver will output all results to csv file
# for year in filepaths:
#     for file in year:
#         text_to_csv(file, "output.csv")

