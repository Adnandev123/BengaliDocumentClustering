# -*- coding: utf-8 -*-

import random
import string
import cherrypy
import sys
import json
import gensim
import time
from gensim.models.keyedvectors import KeyedVectors
import numpy
import multiprocessing
from multiprocessing import Pool, Manager, Array
import scipy
from scipy.cluster.hierarchy import ward, dendrogram
import re
import simplejson
import io
import os
import matplotlib
from datetime import datetime


global word_vector
global doc
global s0
matplotlib.use('Agg')


def calculateDistance2(j):
    s1 = doc[j]
    distance = word_vectors.wmdistance(s0.split(), s1.split())
    return distance


class ClusterMarge(object):

    def __init__(self):

        global word_vectors
        # word_2_vec_loc = os.environ['WORD2VEC']
        word_2_vec_loc = 'word2vec_model.txt'
       # word_vectors = KeyedVectors.load_word2vec_format("/home/adnan/NetBeansProjects/recentnewscluster/News Cluster Python/word2vec_model.txt", encoding='utf-8', binary=False,unicode_errors='strict')
        word_vectors = KeyedVectors.load_word2vec_format(word_2_vec_loc, encoding='utf-8', binary=False,unicode_errors='strict')

    def margeCluster(self, json_data):

        # with open(sys.argv[1]) as json_data:
        d = json_data

        # json_data = sys.argv[1]
        # d = json.loads(json_data)
        print len(d)

        docID = 0;
        global doc
        doc = []
        names = []
        loc = []

        stop_words = frozenset(
            [u"ও", u"এ", u"থেকে", u"করে", u"তাঁকে", u"জানান", u"হয়", u"এবং", u"করা", u"হয়েছে", u"না", u"জন্য", u"এই",
             u"এক", u"হবে", u"নিয়ে", u"তিনি", u"এর", u"একটি", u"করতে", u"করেন", u"বলেন", u"সঙ্গে", u"মধ্যে", u"হচ্ছে",
             u"বলে", u"তার", u"হয়ে", u"পর", u"গত", u"আর", u"করার", u"তাদের", u"দিয়ে", u"দিন", u"করেছে", u"যায়",
             u"প্রতি", u"ই", u"তবে", u"কোনো", u"তা", u"তারা", u"কিন্তু", u"যে", u"শুরু", u"বিভিন্ন", u"গেছে", u"সে",
             u"রয়েছে", u"করেছেন", u"ছিল", u"আগে", u"সব", u"দুই", u"পর্যন্ত", u"কাছে", u"নেই", u"টি", u"এখন", u"নয়",
             u"হিসেবে", u"ওই", u"কিছু", u"অনেক", u"হলে", u"কোন", u"পারে", u"আপনার", u"কারণে", u"দিতে", u"প্রায়",
             u"আমাদের", u"আরও", u"আমরা", u"হতে", u"নতুন", u"আছে", u"জানা", u"জানান", u"করছে", u"আগের", u"কে", u"ছিলেন",
             u"এসব", u"ওপর", u"এখানে", u"মতো", u"ধরে", u"আমার", u"বা", u"বিশেষ", u"মাধ্যমে", u"বলা", u"বড়", u"হওয়ার",
             u"এতে", u"দিকে", u"তাই", u"সকল", u"দেন", u"জন", u"কি", u"তিন", u"পরে", u"পরের", u"সেই", u"দেয়া", u"হয়নি",
             u"পাওয়া", u"একই", u"দেওয়া", u"বলেছেন", u"হলো", u"একজন", u"এমন", u"আলী", u"আরো", u"যাবে", u"ছাড়া", u"তাকে",
             u"হক", u"পড়ে", u"দিয়েছে", u"আমি", u"কর্তৃক", u"আজকের", u"চৌধুরী", u"গিয়ে", u"ফলে", u"যাচ্ছে", u"থাকে",
             u"দেয়", u"কারণ", u"করবে", u"শুধু", u"অন্য", u"করছেন", u"আবার", u"হলেও", u"সামনে", u"নাম", u"যা", u"টা",
             u"চেষ্টা", u"মানিক", u"সেখানে", u"নিতে", u"দিয়েছেন", u"নামে", u"এদিকে", u"শেষে", u"নাই", u"ভালো", u"দিনের",
             u"হাতে", u"যদি", u"কেউ", u"সুযোগ", u"তুলে", u"ইত্তেফাক", u"তৈরি", u"এটা", u"চলছে", u"বর্তমান", u"চলে",
             u"তখন", u"বেশ", u"করলে", u"ক্ষেত্রে", u"সকাল", u"এছাড়া", u"এম", u"বাইরে", u"হওয়া", u"সম্ভব", u"তো",
             u"ভোরের", u"ফিরে", u"দু", u"মোঃ", u"হয়েছিল", u"তাঁর", u"যেতে", u"চেয়ে", u"থাকা", u"করুন", u"রাতে",
             u"থাকবে", u"অনুযায়ী", u"গেলে", u"এসে", u"থেকেই", u"পাশাপাশি", u"একটা", u"সাড়ে", u"নানা", u"হন", u"মো",
             u"সকালে", u"মাত্র", u"কেন", u"সফর", u"সমূহ", u"এরপর", u"কী", u"যান", u"খুব", u"করবেন", u"রেখে", u"এটি",
             u"নেওয়া", u"হয়েছেন", u"পালন", u"যেন", u"যখন", u"কয়েক", u"তাহলে", u"রাখা", u"প্রয়োজন", u"নেয়া", u"টায়",
             u"যারা", u"দেওয়ার", u"দেয়ার", u"জানতে", u"ব্যবহার", u"আশা", u"করি", u"ঠিক", u"থাকার", u"পেয়ে", u"জানিয়েছে",
             u"এখনো", u"গড়ে", u"ঘটে", u"আসে", u"অন্যান্য", u"বিষয়", u"সবাই", u"আল", u"থাকতে", u"হল", u"হওয়ায়",
             u"অবস্থান", u"বাড়ি", u"নিয়েছে", u"কাছ", u"উল্লেখ", u"কয়েকটি", u"এগিয়ে", u"বের", u"দেখতে", u"চাই", u"উচিত",
             u"যাওয়া", u"পারেন", u"কবির", u"রাখতে", u"সাথে", u"দেখে", u"ছেলে"])

        kmeansNOC = len(d)

        for i in range(len(d)):  # len(d)
            document = d[i][u'news'][0][u'content']
            document = re.sub(u'\\,', ' ', document)
            document = re.sub(u'।', ' ', document)
            document = re.sub('-', ' ', document)
            document = re.sub(u'\\?', ' ', document)
            document = re.sub(u'\\!', ' ', document)
            document = re.sub('\'', ' ', document)
            document = re.sub(u'‘', ' ', document)
            document = re.sub(u'’', ' ', document)
            document = re.sub(u'\\(', ' ', document)
            document = re.sub(u'\\)', ' ', document)
            document = re.sub(u':', ' ', document)
            document = re.sub(u'—', ' ', document)
            document = re.sub(u';', ' ', document)
            document = ' '.join([word for word in document.split() if word not in stop_words])
            document = re.sub(' +', ' ', document)
            # print document

            doc.insert(docID, document)
            names.insert(docID, str(docID) + " " + d[i][u'news'][0][u'headline'])
            loc.insert(docID, docID)
            docID += 1

        # print('loading word2vec...')
        # word_vectors.init_sims(replace=True)


        # print('start calculating distance matrics...')
        t = time.time()
        dist = numpy.zeros(shape=(len(doc), len(doc)))

        for i in range(len(doc)):
            global s0
            s0 = doc[i]
            pool = Pool(processes=multiprocessing.cpu_count())
            distances = pool.map(calculateDistance2, range(i))
            # print distances

            for j in range(len(distances)):
                dist[i, j] = distances[j]
                dist[j, i] = distances[j]
                # print(str(i) + ' , ' + str(j) + ' : ' + str(dist[i, j]))
            pool.close()
            pool.join()

        print('total time : ' + str(time.time() - t))

        linkage_matrix = ward(dist)
        Z = dendrogram(linkage_matrix, orientation="right", labels=loc, count_sort='descendent',show_leaf_counts='true')

        count = 0
        total = len(Z.get('ivl'))
        threshold = 0.0
        margedClusters = []
        clusterPos = 0

        for i in range(len(Z.get('ivl')) - 1):
            loc1 = Z.get('ivl')[i]
            loc2 = Z.get('ivl')[i + 1]
            distance = dist.item(int(loc1), int(loc2))

            # print names[loc1]
            # print str(dist.item(int(loc1), int(loc2)))
            # print loc[loc1]

            if (i < (total / 3)):
                threshold = 25.0;
            else:
                threshold = 22.0;

            margedClusters.insert(clusterPos, loc[loc1])
            clusterPos += 1

            if (distance >= threshold):
                # print str(dist.item(int(loc1), int(loc2)))
                # count+=1
                margedClusters.insert(clusterPos, -100)
                clusterPos += 1
                # print '=================='+str(count)+'======================='

            if (i == (len(Z.get('ivl')) - 2)):
                # print names[loc2]
                # print loc[loc2]
                margedClusters.insert(clusterPos, loc[loc2])

        news = []
        newsCount = 0
        clusters = []
        clusterCount = 0
        categories = []
        for i in range(len(margedClusters)):

            if (margedClusters[i] == -100):
                clusters.insert(clusterCount, news)
                categories.insert(clusterCount, 'other')
                clusterCount += 1
                news = []
                # print "============================="
            else:
                for j in range(len(d[margedClusters[i]][u'news'])):
                    news.insert(newsCount, d[margedClusters[i]][u'news'][j])
                    newsCount += 1
                    # print margedClusters[i]

        final_cluster = [{"category": category, "news": cluster, "newsCount": len(cluster)} for category, cluster in
                         zip(categories, clusters)]
        sorted_array = sorted(final_cluster, key=lambda x: x['newsCount'], reverse=True)

        hcaNOC = len(sorted_array)

        with open("clusterlog.txt", "a") as myfile:
         myfile.write(str(datetime.now()) + " >> " + " Kmeans : " + str(kmeansNOC) + " , HCA : " + str(hcaNOC) + "\n")

        print(str(datetime.now()) + " >> " + " Kmeans : " + str(kmeansNOC) + " , HCA : " + str(hcaNOC) + "\n")

        #with io.open('cluster_output_from_python_dump.txt', 'w', encoding='utf-8') as f:
         #    f.write(str(sorted_array))

       # return json.dumps(sorted_array,  ensure_ascii=False).encode('utf8')
        jsonString = json.dumps(sorted_array)
        jsonlist = json.loads(jsonString)
        return jsonlist


@cherrypy.expose
class StringGeneratorWebService(object):
    @cherrypy.tools.accept(media='application/json')
    def GET(self):
        return cherrypy.session['mystring']

    @cherrypy.tools.json_out()
    @cherrypy.tools.json_in()
    def POST(self):
        input_json = cherrypy.request.json
        cMarge = ClusterMarge()
        final_cluster = cMarge.margeCluster(input_json)
        return final_cluster
       # return json.dumps(input_json,  ensure_ascii=False)

    #def PUT(self, another_string):
     #   cherrypy.session['mystring'] = another_string

 #   def DELETE(self):
      #  cherrypy.session.pop('mystring', None)

def accept_request():
    pass

    # conf = {
    #     '/': {
    #         'request.dispatch': cherrypy.dispatch.MethodDispatcher(),
    #         'tools.sessions.on': True,
    #         'tools.response_headers.on': True,
    #         'tools.response_headers.headers': [('Content-Type', 'text/plain')],
    #         'response.timeout' : 60 * 60 * 60,
    #     }
    # }

    # cherrypy.config.update({'server.socket_host': '0.0.0.0', 'server.socket_port': 8099})
    # cherrypy.engine.restart()
    # cherrypy.quickstart(StringGeneratorWebService(), '/d/', conf)

def process_cluster(file_name):

    cMarge = ClusterMarge()

    with open(file_name, 'rb') as file:

        data = file.read()
        data = json.loads(data)
        print('data loaded')
        final_cluster = cMarge.margeCluster(data)
        print('cluster is ready')

    with open('output.json', 'wb') as out_file:
        out_file.write(final_cluster)

if __name__ == '__main__':

    process_cluster('temp.json');