# Lyrics Thumbnailing
This repository contains the code needed to run and reproduce results reported in our RANLP2019 paper.

## Abstract
Given the peculiar structure of songs, applying generic text summarization methods to lyrics can lead to the generation of highly redundant and incoherent text. In this paper, we propose to enhance state-of-the-art text summarization approaches with a method inspired by audio thumbnailing. Instead of searching for the thumbnail clues in the audio of the song, we identify equivalent clues in the lyrics. We then show how these summaries that take into account the audio nature of the lyrics outperform the generic methods according to both an automatic evaluation and human judgments.

## Dependencies
- the packages present when successfully running the code are listed in the file [pip list --local](https://github.com/TuringTrain/lyrics_thumbnailing/blob/master/pip%20list%20--local)
- the methods rely on the TextRank implementation [summa](https://pypi.org/project/summa/). The input (and output) to our method are text lines. We adapted the source code of summa to work with lines instead of raw text and circumvent sentence splitting. Therefore the altered code of summa is required.

## Running the models
- Download the [topic model](https://mega.nz/#!KJhnECjZ!tVoo4_EHO6g5S5XL-wP7TkPBV_sEGJnckIgqIvFuVIw)
- See usage example in this [Jupyter Notebook](https://github.com/TuringTrain/lyrics_thumbnailing/blob/master/Showcase.ipynb)


### Citation
```
@inproceedings{fell:summarization,
  TITLE = {Song Lyrics Summarization Inspired by Audio Thumbnailing},
  AUTHOR = {Fell, Michael and Cabrio, Elena and Gandon, Fabien and Giboin, Alain},
  BOOKTITLE = {RANLP 2019 - Recent Advances in Natural Language Processing (RANLP)},
  ADDRESS = {Varna, Bulgaria},
  YEAR = {2019}
}
```
