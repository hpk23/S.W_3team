#coding: utf-8
from bs4 import BeautifulSoup
import urllib
import time
import lstm
import pickle as pkl
import elastic


if __name__ == "__main__" :

    try :

        while(True) :

            html_text = urllib.urlopen("http://movie.naver.com/movie/point/af/list.nhn").read()
            soup = BeautifulSoup(html_text, 'lxml')
            duplicate = {}
            count = 0
            file = open('predict.txt', 'w')
            for option in soup.find_all('option') :
                value = option.get('value')
                if value == None or value == 'movie' or value == 'userid' : continue
                url = 'http://movie.naver.com/movie/point/af/list.nhn?st=mcode&sword={}&target=after&page='.format(value)
                for page in range(1, 10+1) :
                    connection = urllib.urlopen(url+str(page))
                    html_text = connection.read()
                    soup2 = BeautifulSoup(html_text, 'lxml')

                    img = None
                    for div in soup2.find_all('div') :
                        if div.get('class') is None : continue
                        if div.get('class')[0] == 'fl' :
                            try :
                                img = div.find('a').find('img').get('src')
                            except :
                                continue


                    for td in soup2.find_all('td') :
                        value = td.get('class')
                        if value is None : continue
                        if value[0] == 'title' :
                            attr = td.text.split('\n')
                            title = attr[1]
                            review = attr[2]

                            if title+review in duplicate : continue
                            elif title+review not in duplicate :
                                duplicate[title+review] = 1

                            label = '1'
                            if review == "" : continue
                            string = title.encode('utf-8') + '\t' + review.encode('utf-8').strip('\r\n') + '\t' + label.strip('\r\n') + '\t' + img + '\n'
                            file.write(string)
                            count += 1
            connection.close()
            file.close()
            lstm.predict_lstm()
            print('Done...')


            review_dict = {}
            lines = open('predict.out', 'r').readlines()
            for line in lines :
                tokens = line.split('\t')
                img = tokens[0]
                title = tokens[1]
                review = tokens[2]
                sentiment = tokens[3]

                if sentiment.strip('\n').strip('\r\n') == '1' : sentiment = 'Positive'
                else : sentiment = 'Negative'

                if title not in review_dict :
                    review_dict[title] = [(img, review, sentiment)]
                else :
                    review_dict[title].append((img, review, sentiment))

            f = open('movie_review.pkl', 'wb')
            pkl.dump(review_dict, f)
            f.close()

            elastic.update_elastic_data()
            time.sleep(3600)

    except KeyboardInterrupt :
        print('Parsing interupted')
    except Exception, e :
        print '',