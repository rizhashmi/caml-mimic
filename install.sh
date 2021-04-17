#/bin/bash

sudo apt-get install python3-venv build-essential python3.6

python3.6 -m venv ../venv
source ../venv/bin/activate
pip install -r requirement.txt


