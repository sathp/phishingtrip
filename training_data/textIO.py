import csv


def text_to_csv(input_path, output_path):
    """
    This is a script that's meant to parse a .txt file and write its contents to a .csv file.
    Parameter:
    input_path -- path to input text file
    output_path -- path to output .csv file
    returns - nothing
    """

    # Parsing through .txt file
    text_file = open(input_path)
    text = text_file.encode('utf-8').readlines()
    output = [input_path]
    body = ""
    for i in range(len(text)):
        try:
            text[i] = text[i].encode('utf-8').strip("\n")
            if text[i].find("Subject:") != -1:
                output.append(text[i])
                for x in range(i + 1, len(text)):
                    temp = " " + text[x].strip("\n")
                    body += temp
                output.append(body)
                break
        except:
            pass

    # Print Output
    # print(output)

    # Writing Data to .csv file
    csv_file = open(output_path, 'a')
    with csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(output)



