# bullet-arXiv

bullet-arXiv is an NLP keyword generator for research papers. By analyzing paper abstracts, bullet-arXiv can cluster by topic and generate meaningful summary keywords.

To see it in action, visit the [web app](http://bullet-env-2.eba-p3cickfb.us-west-2.elasticbeanstalk.com) where I apply bullet-arXiv to nearly 20,000 astronomy papers published on the arXiv between *2012-09-01* and *2022-09-01*. I used the excellent `arxivscraper` package found [here](https://github.com/Mahdisadjadi/arxivscraper). 

You may also pull the latest Docker release, see below.

## About
The goal of the project is to automatically generate meaningful keywords for research publications. Doing so allows for ease of finding publications relevant to you -- particularly important when an overwhelming number of papers are published each day. This project was inspired by [Eren et al (2020)](https://doi.org/10.1145/3395027.3419591), who use Latent Dirichlet Allocation on COVID-19 publications.


## Installation

To use the tools developed here or to recreate the results, it is recommended to set up a [conda](https://www.anaconda.com/products/individual) environment with `environment.yml` included in this repo:

```
$ git clone https://github.com/RaySSharma/bullet-arXiv.git
$ cd bullet-arXiv
$ conda env create -f environment.yml
$ conda activate bullet_arxiv
```
To install the package:
```
$ pip install .
```

If you want to simply run the webapp locally, you can pull the latest docker image:

```
$ docker pull rayssharma/bullet-arxiv:latest
$ docker run -p 8080:80 rayssharma/bullet-arxiv:latest 
```

then navigate to `localhost:8080`.

## Methodology

The steps in the NLP pipeline are summarized as follows:

- Separate the raw data into training (60%), validation (20%), and test (20%) sets.
    
- Format the incoming paper abstracts, stripping LaTeX equations, digits, and punctuation. Generate lemmatized tokens, comparing against a set of user-defined + common english stop-words.

- Vectorize text using a TF-IDF transformer, down-weighting tokens that appear frequently throughout the corpus. The output is a sparse matrix.
    
- Reduce the dimensionality of the sparse, vectorized text using Latent Semantic Analysis with n=100 components and normalizing the results.
    
- Run K-means clustering, where the number of clusters were chosen by maximizing the silhouette score against the validation set. Silhouette scores measure both the distance between points within a cluster, and the distance between clusters.
    
- On each cluster run Latent Dirichlet Allocation to generate latent topics. Latent topics are simple distributions over the words in the cluster. By gathering the most probable words within each distribution, I can identify keywords that best describe each topic. By combining up the keywords from each topic, I end up with a set of key words that describe the cluster as a whole.
    
- Run t-SNE to flatten the LSA-vectorized data down into 2 dimensions for visualization. Points are colored by their K-means cluster designation, and clusters are assigned the most probable keywords across their LDA topics.

## Looking Forward

Future iterations of bullet-arXiv will include:
- Support for N-gram keywords
- LDA hyperparameter tuning against the Topic Coherence metric
- Exploration of other clustering algorithms