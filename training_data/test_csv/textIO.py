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
    text = text_file.readlines()
    output = []
    body = []
    for i in range(len(text)):
        text[i] = text[i].strip("\n")
        if text[i].find("Subject:") != -1:
            temp = text[i]
            output.append(temp[8:])
            for x in range(i + 1, len(text)):
                temp = text[i].strip("\n")
                body.append(text[x])
            output.append(body)
            break
    print(output)

    # Writing Data to .csv file
    # csv = open(output_path)
