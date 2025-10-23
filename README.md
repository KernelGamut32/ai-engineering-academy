# AI Engineering Academy

## Environment Configuration

- Python 3.13.8
- Pip 25.2

```bash
sudo apt update && sudo apt upgrade -y

sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update

sudo apt install python3.13 python3.13-venv
python3.13 -m pip install --upgrade pip

# Optional step: create an alias in ~/.bash_aliases for python and pip
# alias python='python3.13'
# alias pip='pip3.13'
# Adjust statements below if aliases are created
# Also, feel free to change name from llm_venv to something else

python3.13 -m venv llm_venv
source llm_venv/bin/activate
pip3.13 install setuptools
pip3.13 install ipykernel
python3.13 -m ipykernel install --user --name=llm_venv --display-name 'llm_venv'

# This environment can now be selected as target kernel for Jupyter notebooks running in VS code
```
