# Document-Classification
Classification of given document using core NLP method and word2vec method


## Intoduction
The task was to classify documents based on words given. The training data was just simple words or phrases given with their category.
Unigram, Bigram and trigrams were made and diferent results were obtained.

![Results of word2vec model](https://raw.githubusercontent.com/AsimMessi/Document-Classification/master/4.png)

## Dependencies
- Python3
- Gensim(for word2vec)

## Usage
- Get word2vec pretrained model for word vectors from [here](https://github.com/alexandres/lexvec#pre-trained-vectors).
- Execute getdata to save pickle files of all the training words and categories.
- Enter your testing text in text_data.txt.
- Execute test to get results from core NLP method.
- Run word2vec to get results from word2vec model.

## Results

Results of core nlp and word2vec model using unigrams, bigrams and trigrams are shown in pie chart

![Results of word2vec model](https://raw.githubusercontent.com/AsimMessi/Document-Classification/master/1.png)

![Results of word2vec model](https://raw.githubusercontent.com/AsimMessi/Document-Classification/master/2.png)

![Results of word2vec model](https://raw.githubusercontent.com/AsimMessi/Document-Classification/master/3.png)
