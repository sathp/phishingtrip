# phishingtrip

Final Project for CS196 - CS@Illinois.

Team Members: Krist Pregracke, Jingjin Wang, Sathwik Pochampally and Soumya Kuruvila.

This program uses spam and non-spam e-mail data to

### Uses Python v3.6.2

## Installation

### For Mac/Linux/Unix Users:

Mac:
```bash
brew install python3
pip3 install virtualenv
pip3 install tensorflow
brew update
brew install p7zip
```

Linux:
```bash
sudo apt-get install python3
sudo apt-get install python-pip
sudo pip install --upgrade pip
sudo pip install --upgrade virtualenv
sudo pip install tensorflow
sudo apt-get install p7zip
```

Check to make sure python, pip and virtualenv are installed correctly:
```bash
python3 --version
pip3 --version
virtualenv --version
```

Then activate the virtualenv:
```bash
cd /path/to/phishingtrip
source bin/activate
```

The prompt of your shell should now read:
```bash
(phishingtrip)Your-Computer:phishingtrip UserName$)
```

Installing dependencies:
```bash
pip install -r requirements.txt
```

Installing

## Downloading the training_data
We used two e-mail databases to train our model. They can be downloaded at:
```
http://untroubled.org/spam/
```
and
```
https://www.kaggle.com/kaggle/hillary-clinton-emails?login=true
```

### Unzipping .7z files from untroubled:
```bash
7z x -r /training_data/*.7z
```

### Parsing through raw e-mail data:
```bash
python3 training_data/file.py
python3 training_data/parser.py
```

Filepaths to e-mails will be stored in file.csv.

Selected e-mail data is stored in the form of a csv at output.csv.

## Checking whether an email is a spam:

First, copy and paste the email to spamfilter_wordbased/input.txt

Then, run

```bash
python3 spamfilter_wordbased/test.py
```


