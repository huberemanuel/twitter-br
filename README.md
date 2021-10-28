# Pre-trained language models for Tweets in Portuguese

The pre-trained language templates allow you to fine-tune several NLP tasks, allowing for easy adaptation with few lines of code and little time to fine-tune the final task. 
Although there are initiatives like the Multilingual BERT that allows text processing in Portuguese or even Bertimbau and AlbertPT that were exclusively trained in Portuguese, there is still no model focused on Twitter.
Tweets are complex, expressed in a few tokens, and users create unique words and expressions. Therefore, this project seeks to train language models in Twitter data and make them available on the Hugging Face Hub.

We hope that the development of the proposed contributions will help to minimize the lack of computational resources for Brazilian Portuguese. Additionally, the datasets include tweets related to the COVID-19 pandemic. Therefore we expect the pre-trained models to help the development of tools and systems to combat misinformation.

## Datasets

| Corpus                                                                           | Size |    Year   |                Theme               |
|----------------------------------------------------------------------------------|:----:|:---------:|:----------------------------------:|
| [TweetSentBR](https://bitbucket.org/HBrum/tweetsentbr/src/master/)                                                                      |  15k |    2018   |             Sentiments             |
| [Portuguese Tweets for Sentiment Analysis](https://www.kaggle.com/augustop/portuguese-tweets-for-sentiment-analysis)                                         | 800k |    2018   | Political, sentiments, and general |
| [Detecção de Casos de Violência Patrimonial a partir do Twitter](https://sol.sbc.org.br/index.php/brasnam/article/download/6456/6352/)                   | 200M | 2015/2016 |   Patrimonial violence, general.   |
| [Vamos falar sobre deficiencia?](https://sol.sbc.org.br/index.php/brasnam/article/download/3601/3560/)                                                   |  17k |    2018   |             Disability             |
| [A first public dataset from Brazilian twitter and news on COVID-19 in Portuguese](https://www.sciencedirect.com/science/article/pii/S2352340920310738) |  4M  |    2020   |              COVID-19              |
| [BraSNAM2018](https://github.com/danielkansaon/BraSNAM2018-Dataset-Analise-de-sentimentos-em-tweets-em-portugues-brasileiro)                                                                      |  12k |    2018   |             Sentiments             |
| [Covid-19 e Tweets no Brasil](https://sol.sbc.org.br/index.php/brasnam/article/view/16127)                                                      |  6M  |    2020   |              COVID-19              |

After anonimizing user names with @user, replacing URLs with @http, and removing duplicates the full dataset contains 175 M tweets.


## Current state of training

| Model      | Training script | Executing experiment | Deployed |
|------------|:---------------:|:--------------------:|:--------:|
| Roberta    |        🗹        |           🗹          |     ☐    |
| Albert     |        ☐        |           ☐          |     ☐    |
| DistilBERT |        ☐        |           ☐          |     ☐    |

