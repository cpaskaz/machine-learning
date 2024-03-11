**Pre-Requisites**
Your personal laptop will need to have Python installed and we highly recommend using Python 3.10. You can use a tool like pyenv (mac) or pyenv-win (windows) to easily download and switch between Python versions.

pyenv install 3.10.11  # install
pyenv global 3.10.11  # set default

Once we have our Python version, we can create a virtual environment to install our dependencies. 

We'll download our Python dependencies after we clone our repository from git shortly.

python3 -m venv mlvenv  # create virtual environment
source mlvenv/bin/activate  # on Windows: venv\Scripts\activate

python3 -m pip install --upgrade pip setuptools wheel

install requirements from the requirements.txt file.

I ran this on a Mac M1 Pro, python 3.10.11