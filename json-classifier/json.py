import csv
import json

def converter(csv, json):
    csvfile = open(csv)
    jsonfile = open(json)
    read = csv.DictReader(csvfile)
    input = json.dumps(read)
    jsonfile.write(input)

    return jsonfile
