import csv
import json

def converter(csv, json)
    csvfile = open(csv)
    jsonfile = open(json)
    read = csv.DictReader(csvfile)
    in = json.dumps(read)
    jsonfile.write(in)

    return jsonfile
