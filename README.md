### Setup
I use `python=3.10.8` and `pytorch=1.13.0` to run models.
```shell
# create environment
conda env create -f env.yml
# install whisper
pip install git+https://github.com/openai/whisper.git
# or
git clone https://github.com/openai/whisper.git && cd whisper
pip install -e .

# install speechbrain
pip install speechbrain
# or
git clone https://github.com/speechbrain/speechbrain.git && cd speechbrain
pip install -r requirements.txt
pip install --editable .
```