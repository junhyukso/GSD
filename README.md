
## Download tokenizer
Our model use chameleon(from META)'s image tokenizer. Please download it from
`https://ai.meta.com/resources/models-and-libraries/chameleon-downloads/`

and put it into
`ckpts/chameleon/tokenizer`

The folder structure should be :

The files
`checklist.chk,text_tokenizer.json,vqgan.ckpt,vqgan.yaml`

should be in 
`ckpts/chameleon/tokenizer/`


## Install dependancies

We recommend to use pytorch>=2.3.0
```
pip install transformers==4.48.1
pip install sentencepiece
pip install accelerate>=0.26.0
pip install absl-py
```


## Test our GSD

```
python test_GSD.py
```


## Test the baseline (SJD; ICLR25)
```
python test_SJD.py
```


Thank you!
