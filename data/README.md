# Data
## Environment Requirements
* python 3.6
* tqdm
* [StanfordCoreNLP](https://stanfordnlp.github.io/CoreNLP/) \[[download](https://nlp.stanford.edu/software/stanford-corenlp-full-2018-10-05.zip)\]

## Usage
1. Download raw data and decompress it into `data` folder.
2. [Download CoreNlp](https://nlp.stanford.edu/software/stanford-corenlp-full-2018-10-05.zip), put it in `./data/` folder and then unzip it.
3. Run the following commands to convert the raw data to the specified format.
```shell
python preprocess.py
```
3. The conversion results are `train.json` and `test.json`.
4. In addition, create a file `relation2id.txt`.