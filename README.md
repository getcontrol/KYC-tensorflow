# Machine Learning Driven Identity Verification

This repo contains a Jupyter Notebook that utilizes a Tensorflow Model to identity and smartly crop an Identity Card, along with an accompanying Flask asp.

# Installation Instructions

### Jupyter Notebook

1. Create and activate a Python 3 Virtual environment

```python3 -m venv env```

```source env/bin/activate```

2. Install Requirements

```pip install -r requirements.txt```

3. Start Jupyter Notebook

```ipython notebook --ip=127.0.0.1```

4. Open & Run [Tensorflow - Verification.ipynb](https://github.com/getcontrol/tensorflow-verification/blob/master/Tensorflow%20-%20Verification.ipynb)

The mechanics of the Tensorflow model and OpenCV transforms is documented inline.

### Flask App
This demonstrates uploading the  identity document via a Flask app with the relevant pre-processing OpenCV steps. The machine learning model is optimized for images that are acquired via an Android or Samsung Phone. Use ngrok to share localhost URL for mobile browser testing.

Notes:

The Step 1 Form and Step 3 selfie or not fully functioning yet.

You must take a photo of ID card in Step 2 and selfie in Step 3 or app will break.

App.py behaves very similarily to the Jupyter Notebook , with a few exceptions documented as comments in the code.

1. Create and activate a Python 3 Virtual environment

```python3 -m venv env```

```source env/bin/activate```

2. Install Requirements

```pip install -r requirements.txt```

3. Start Flask app

```python app.py```

4. In a separate terminal tab start ngrok

```./ngrok http 5000```

5. Test ngrok URL on mobile browser. Final ID image is saved in tmp/FINAL.jpg

#TODO
1. Deploy Flask app & model to production (GCP)
2. Connect Step 1 form to database
3. Send app-final.jpg to Nanonets API and save JSON response to database
4. Send app-img_self.jpg to Face++ API and save JSON response to database

### References
(https://github.com/RRanddom/tf_doc_localisation)

(https://github.com/AdivarekarBhumit/ID-Card-Segmentation)

(https://zhuanlan.zhihu.com/p/56336225)

### Citation
Please cite this paper, if using midv dataset, link for dataset provided in paper

    @article{DBLP:journals/corr/abs-1807-05786,
      author    = {Vladimir V. Arlazarov and
                   Konstantin Bulatov and
                   Timofey S. Chernov and
                   Vladimir L. Arlazarov},
      title     = {{MIDV-500:} {A} Dataset for Identity Documents Analysis and Recognition
                   on Mobile Devices in Video Stream},
      journal   = {CoRR},
      volume    = {abs/1807.05786},
      year      = {2018},
      url       = {http://arxiv.org/abs/1807.05786},
      archivePrefix = {arXiv},
      eprint    = {1807.05786},
      timestamp = {Mon, 13 Aug 2018 16:46:35 +0200},
      biburl    = {https://dblp.org/rec/bib/journals/corr/abs-1807-05786},
      bibsource = {dblp computer science bibliography, https://dblp.org}
    }
