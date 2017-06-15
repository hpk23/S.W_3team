from elasticsearch import Elasticsearch
import pickle as pkl


def update_elastic_data() :

    f = open('movie_review.pkl', 'rb')
    review_dict = pkl.load(f)
    es = Elasticsearch([{'host' : 'localhost', 'port' : 9200}])
    es.indices.delete(index='movie_info', ignore=[400, 404])
    for title, values in review_dict.items() :
        for img, review, sentiment in values :
            #print title, img, review, sentiment
            es.index(index='movie_info', doc_type='review',
                     body={"title" : title, "img" : img, "review" : review, "sentiment" : sentiment})