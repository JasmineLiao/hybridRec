import os
import numpy as np
import pickle
import pandas as pd
import csv
import random
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import time
from sklearn.metrics import jaccard_similarity_score


movie_path = '../ml-1m/movies.dat'
rating_path = '../ml-1m/ratings.dat'
user_path = '../ml-1m/users.dat'

def pre_user():
    new_path = '../ml-1m/users.csv'
    csvfile = open(new_path, 'w', newline='')
    userwriter = csv.writer(csvfile, dialect='excel')
    userwriter.writerow(['uid', 'gender', 'age', 'occu', 'code'])
    user = open(user_path, 'r', errors='ignore')
    for u in user:
        line = u.strip().split('::')
        userwriter.writerow(line)
    csvfile.close()


def pre_movie():
    new_path = '../ml-1m/movies.csv'
    csvfile = open(new_path, 'w', newline='')
    moviewriter = csv.writer(csvfile, dialect='excel')
    moviewriter.writerow(['mid', 'title', 'genre'])
    movie = open(movie_path, 'r', errors='ignore')
    for m in movie:
        line = m.strip().split('::')
        moviewriter.writerow(line)
    csvfile.close()


def pre_rating():
    train_path = '../ml-1m/train.csv'
    rfile = open(train_path, 'w', newline='')
    rwriter = csv.writer(rfile, dialect='excel')
    rwriter.writerow(['uid', 'mid', 'rating', 'time'])

    test_path = 'C../ml-1m/test.csv'
    sfile = open(test_path, 'w', newline='')
    swriter = csv.writer(sfile, dialect='excel')
    swriter.writerow(['uid', 'mid', 'rating', 'time'])

    user = open(rating_path, 'r', errors='ignore')
    split=0.2
    for u in user:
        line = u.strip().split('::')
        if random.random() < split:
            swriter.writerow(line)
        else:
            rwriter.writerow(line)

    rfile.close()
    sfile.close()


def pre_data():
    pre_movie()
    pre_rating()
    pre_user()


def gen_user_feat():
    user_path='../ml-1m/users.csv'
    dump_path='../ml-1m/cache/users.pkl'
    user=pd.read_csv(user_path)
    age_df=pd.get_dummies(user['age'], prefix='age')
    gender_df=pd.get_dummies(user['gender'], prefix='gender')
    occu_df=pd.get_dummies(user['occu'], prefix='occu')
    user=pd.concat([user['uid'], gender_df, age_df, occu_df], axis=1)
    pickle.dump(user, open(dump_path, 'wb'))


def gen_rating_feat_():
    rating_path='../ml-1m/train.csv'
    rating=pd.read_csv(rating_path)
    rating['uid']=rating['uid'].apply(lambda x: x-1)
    rating['mid'] = rating['mid'].apply(lambda x: x - 1)
    bias=rating.groupby(['uid'], as_index=False)['rating'].mean()
    bias.sort_values(by='uid')

    rating_matrix=np.zeros((6040,3952))


    for line in rating.itertuples():
        mid = int(line[2])
        uid=int(line[1])
        rate=int(line[3])
        rating_matrix[uid][mid]=rate

    for i in range(6040):
        rating_matrix[i]=rating_matrix[i]-bias.iloc[i, 1]
        if i%100==0:
            print(i)
            print(rating_matrix[i])

    matrix_path='../ml-1m/cache/matrix.pkl'
    pickle.dump(rating_matrix,open(matrix_path,'wb'))


def gen_movie_index():
    ''''
    电影文件有缺失，实际上并没有3952部电影，且缺失的地方不规律
    '''''
    movie = pd.read_csv('../ml-1m/movies.csv', encoding='gbk')
    movie['mid'] = movie['mid'].apply(lambda x: x - 1)
    movie_index = np.zeros(3952)
    for m in movie.itertuples():
        index = int(m[0])
        id = int(m[1])
        movie_index[id] = index

    pickle.dump(movie_index,open('C:/Users/lxya9/Downloads/ml-1m/cache/movie_index.pkl','wb'))


def knn(k):
    data_path='../ml-1m/cache/matrix.pkl'
    data=pickle.load(open(data_path,'rb'))

    neigh = NearestNeighbors(n_neighbors=k,algorithm='brute',metric='cosine')
    neigh.fit(data)
    distance,indices=neigh.kneighbors(data)
    neighbor_path='../ml-1m/new/knn_40.pkl'
    distance_path='../ml-1m/new/knn_distance_40.pkl'
    pickle.dump(distance,open(distance_path,'wb'))
    pickle.dump(indices,open(neighbor_path,'wb'))


def gen_movie_list():
    movie = pd.read_csv('../ml-1m/movies.csv', encoding='gbk')
    mid=movie['mid'].drop_duplicates()
    movie_list=np.zeros(3883)
    for i in range(3883):
        movie_list[i]=mid.iloc[i, 0]
    pickle.dump(movie_list,open( '../ml-1m/cache/movie_list.pkl','wb'))


def gen_user_list():
    test_path = '../ml-1m/test.csv'

    test=pd.read_csv(test_path)
    user = test['uid'].drop_duplicates()
    ulist = np.zeros(len(user))
    for i in range(len(user)):
        ulist[i] = int(user.iloc[i]) - 1

    ulist.sort()
    return ulist


def predict(n):
    #选择n个近邻进行预测
    ISOTIMEFORMAT = '%Y-%m-%d %X'
    print('start ' + time.strftime(ISOTIMEFORMAT, time.localtime()))
    matrix_path = '../ml-1m/cache/matrix.pkl'
    bias_path = '../ml-1m/cache/user_bias.pkl'
    neighbor_path = '../ml-1m/new/knn_40.pkl'
    test_path='../ml-1m/test.csv'
    sim_path='../ml-1m/new/knn_distance_40.pkl'

    movie_list = pickle.load(open('C:/Users/lxya9/Downloads/ml-1m/cache/movie_list.pkl', 'rb'))

    test=pd.read_csv(test_path)
    user=test['uid'].drop_duplicates()
    ulist=np.zeros(len(user))
    for i in range(len(user)):
        ulist[i]=int(user.iloc[i])-1

    ulist.sort()

    neighbors=pickle.load(open(neighbor_path, 'rb'))
    sim = pickle.load(open(sim_path, 'rb'))

    predict_score=np.zeros((len(user),3952))

    rating_matrix=pickle.load(open(matrix_path,'rb'))
    bias=pickle.load(open(bias_path,'rb'))

    print(len(user))

    for u in range(len(user)):
        uid = int(ulist[u])
        neighbor_u = neighbors[uid, 1:n+1]
        sim_u = sim[uid, 1:n+1]
        b = bias.iloc[uid, 1]

        if u % 100 == 0:
            print(u)
            print('start ' + time.strftime(ISOTIMEFORMAT, time.localtime()))


        for mid in movie_list:
            m = int(mid)-1
            if rating_matrix[uid][m] == -b:
                r = 0
                s = 0
                for i in range(n):
                    v = int(neighbor_u[i])
                    if rating_matrix[v][m] != -bias.iloc[v, 1]:
                        r += rating_matrix[v][m] * (1 - sim_u[i])
                        s += (1-sim_u[i])
                if s == 0:
                    s = 1
                predict_score[u][m] = b+r/s

            else:
                predict_score[u][m] = 0

    print('predict done' + time.strftime(ISOTIMEFORMAT, time.localtime()))
    predict_res_path= '../ml-1m/new/predict_%s.pkl'%(n)
    pickle.dump(predict_score,open(predict_res_path,'wb'))


def gen_test_set():
    test_path='../ml-1m/test.csv'
    dump_path='../ml-1m/cache/test_matrix.pkl'
    test=pd.read_csv(test_path)

    user=test['uid'].drop_duplicates().apply(lambda x: x-1)
    ulist=np.zeros(6040)
    for i in range(len(user)):
        ulist[int(user.iloc[i])-1]=i

    test_matrix = np.zeros((len(user), 3952))

    for line in test.itertuples():
        uid=int(line[1])-1
        mid=int(line[2])-1
        u=int(ulist[uid])
        test_matrix[u][mid]=int(line[3])

    pickle.dump(test_matrix,open( dump_path,'wb'))



def run():
    ISOTIMEFORMAT ='%Y-%m-%d %X'

    print('start '+ time.strftime( ISOTIMEFORMAT, time.localtime()))

    list_num=[3,5,7,9,11,13,15]
    top=[3,5,8,10,10,14,16,18,20]


    for e in top:
        print('genre')
        gen_percision(2, e)
        print(time.strftime(ISOTIMEFORMAT, time.localtime()))
        print('--------------------------------------')
        gen_percision(3, e)
        print(time.strftime(ISOTIMEFORMAT, time.localtime()))
        print('--------------------------------------')
        gen_percision(4, e)
        print(time.strftime(ISOTIMEFORMAT, time.localtime()))
        print('--------------------------------------')
        gen_percision(4, e)
        print(time.strftime(ISOTIMEFORMAT, time.localtime()))



def run_knn():
    ISOTIMEFORMAT ='%Y-%m-%d %X'

    print('start '+ time.strftime( ISOTIMEFORMAT, time.localtime()))

    list_num=[3,5,7,9,11,13,15]
    top=[3,5,8,10,10,14,16,18,20]
    for num in list_num:
        for n in top:
            print(time.strftime(ISOTIMEFORMAT, time.localtime()))
            percision(num,n)



def user_list():
    test_path = '../ml-1m/test.csv'
    test = pd.read_csv(test_path)
    user = test['uid'].drop_duplicates()
    ulist = np.zeros(len(user))
    for i in range(len(user)):
        ulist[i] = int(user.iloc[i]) - 1

    ulist.sort()
    path= '../ml-1m/ulist.pkl'
    pickle.dump(ulist,open(path,'wb'))


def original_rate():
    ISOTIMEFORMAT = '%Y-%m-%d %X'
    print('start ' + time.strftime(ISOTIMEFORMAT, time.localtime()))
    rating_path = '../ml-1m/train.csv'
    dump_path = '../ml-1m/rating.pkl'
    rating = pd.read_csv(rating_path)
    rating['uid'] = rating['uid'].apply(lambda x: x - 1)
    rating['mid'] = rating['mid'].apply(lambda x: x - 1)

    rate = np.zeros((6040,3952))

    for line in rating.itertuples():
        uid  =int(line[1])
        mid=  int(line[2])
        rate[uid][mid] = int(line[3])

    print(rate[:3])
    print(time.strftime(ISOTIMEFORMAT, time.localtime()))
    pickle.dump(rate, open(dump_path, 'wb'))


def gen_cf(n):
    sim_path2='../ml-1m/new_method/genre_sim.pkl'
    sim2 = pickle.load(open(sim_path2, 'rb'))
    bias = pickle.load(open('../ml-1m/cache/user_bias.pkl', 'rb'))
    movie_list = pickle.load(open('../ml-1m/cache/movie_list.pkl', 'rb'))
    rating = pickle.load(open('../ml-1m/cache/matrix.pkl', 'rb'))
    u_path='../ml-1m/ulist.pkl'

    ulist=pickle.load(open(u_path,'rb'))
    res = np.zeros((6037, 3952))

    for u in range(6037):
        uid = int(ulist[u])
        b = bias.iloc[uid, 1]
        nei = np.argsort(sim2[uid])[-n:]

        for mid in movie_list:
            m = int(mid) - 1
            if rating[u][m] == -b:
                s = 0
                r = 0
                for v in nei:
                    if rating[v][m] != -bias.iloc[v, 1]:
                        r += rating[v][m] * sim2[u][v]
                        s += sim2[u][v]
                if s == 0:
                    s = 1
                res[u][m] = b + r / s
            else:
                res[u][m] = 0

    dump_path='../ml-1m/new_method/genre_predict_%s.pkl'%(n)

    pickle.dump(res,open(dump_path,'wb'))


def gen_percision(num,n):
    path = '../ml-1m/new_method/genre_predict_%s.pkl' % (num)
    pre = pickle.load(open(path, 'rb'))
    true_rate = pickle.load(open('../ml-1m/cache/test_matrix.pkl', 'rb'))
    bias = pickle.load(open('../ml-1m/cache/user_bias.pkl', 'rb'))
    ulist = pickle.load(open('../ml-1m/ulist.pkl', 'rb'))
    count = 0
    c=0
    emp1=0
    emp2=0

    for i in range(6037):
        uid = int(ulist[i])
        p1 = np.argsort(pre[i])[-n:]
        p2 = np.argsort(pre[i])[:n]

        for m in p1:
            if true_rate[i][int(m)] > bias.iloc[uid, 1]:
                count = count + 1
            if true_rate[i][int(m)]==0:
                emp1=emp1+1
        for m in p2:
            if true_rate[i][int(m)] > bias.iloc[uid, 1]:
                c = c + 1
            if true_rate[i][int(m)] == 0:
                emp2 = emp2 + 1

    print(num,n)
    print(count,emp1)
    print(c,emp2)
    print('---------------------------------')


def percision_ua_c(path1,path2,n):
    pre1=pickle.load(open(path1, 'rb'))
    pre2=pickle.load(open(path2, 'rb'))

    true_rate = pickle.load(open('..s/ml-1m/cache/test_matrix.pkl', 'rb'))
    bias = pickle.load(open('../ml-1m/cache/user_bias.pkl', 'rb'))
    ulist=pickle.load(open('../ml-1m/ulist.pkl','rb'))
    count=0
    pre=pre1+pre2

    for i in range(6037):
        uid = int(ulist[i])
        p1=np.argsort(pre[i])[-n:]

        for m in p1:
            if true_rate[i][int(m)]>bias.iloc[uid,1]:
                count=count+1

    print(n,count)
    print('---------------------------------')
    return count


def percision(path, n):
    pre = pickle.load(open(path, 'rb'))

    true_rate = pickle.load(open('../ml-1m/cache/test_matrix.pkl', 'rb'))
    bias = pickle.load(open('../ml-1m/cache/user_bias.pkl', 'rb'))
    ulist = pickle.load(open('../ml-1m/ulist.pkl', 'rb'))
    count = 0


    for i in range(6037):
        uid = int(ulist[i])
        p1 = np.argsort(pre[i])[-n:]

        for m in p1:
            if true_rate[i][int(m)] > bias.iloc[uid, 1]:
                count = count + 1

    print(n, count)
    print('---------------------------------')
    return count



def item_rate_sim_1m():
    rate_path='../ml-1m/rating.pkl'
    r=pd.read_csv('../ml-1m/train.csv')
    avg=r.groupby(['mid'],as_index=False)['rating'].mean()

    avg_m=np.zeros(3952)
    for line in avg.itertuples():
        id=int(line[1])-1
        avg[id]=line[2]

    rate_matrix=pickle.load(open(rate_path,'rb'))

    for u in range(6040):
        for m in range(3952):
            rate_matrix[u][m]-=avg_m[m]
    print('avg complete')

    rate=rate_matrix.T

    mode=np.zeros(3952)

    for id in range(3952):
        mode[id]=np.sum(np.multiply(rate[id],rate[id]))**0.5
    print('mode complete')

    sim=np.zeros((3952,3952))
    for i in range(3952):
        if i%200==0:
            print(i)
        for j in range(i):
            sim[i][j]=sim[j][i]
        for j in range(i+1,3092):
            if mode[i]==0 or mode[j]==0:
                sim[i][j]=0
            else:
                sim[i][j]=np.sum(np.multiply(rate[i],rate[j]))/(mode[i]*mode[j])

    dump_path='../ml-1m/item/item_sim.pkl'
    pickle.dump(sim,open(dump_path,'wb'))


def user_attr_sim():
    user = pickle.load(open('../ml-1m/cache/users.pkl', 'rb'))
    user = user.values
    user_attr_s = np.zeros((6040, 6040))

    for i in range(6040):
        if i % 300 == 0:
            print(i)
        for j in range(i):
            user_attr_s[i][j] = user_attr_s[j][i]
        for j in range(i+1, 6040):
            user_attr_s[i][j] = jaccard_similarity_score(user[i], user[j])

    print(user_attr_s[:5,:5])

    dump_path='C:/Users/lxya9/Downloads/ml-1m/user_attr_sim.pkl'
    pickle.dump(user_attr_s, open(dump_path, 'wb'))


def item_rate_predict_1m(n):
    path = '../ml-1m/item/item_sim.pkl'
    sim=pickle.load(open(path, 'rb'))
    u_path = '../ml-1m/ulist.pkl'
    #bias = pickle.load(open('C:/Users/lxya9/Downloads/ml-1m/cache/user_bias.pkl', 'rb'))
    ulist = pickle.load(open(u_path, 'rb'))
    predict=np.zeros((6037,3952))
    rate = pickle.load(open('C:/Users/lxya9/Downloads/ml-1m/rating.pkl', 'rb'))

    for u in range(6037):
        uid = int(ulist[u])
        for mid in range(3952):
            if rate[uid,mid]==0:
                nei=np.argsort(-sim[mid])[:n]
                s=0
                r=0
                for m in nei:
                    r+=rate[uid][m]*sim[mid][m]
                    s+=sim[mid][m]
                if s==0:
                    s=1
                predict[u][mid]=r/s

    dump_path='../ml-1m/19/predict_rate_%s.pkl'%(n)
    pickle.dump(predict,open(dump_path,'wb'))


def user_attr_predict_1m(n):
    path = '../ml-1m/user_attr_sim.pkl'
    sim=pickle.load(open(path, 'rb'))
    u_path = '../ml-1m/ulist.pkl'
    bias = pickle.load(open('../ml-1m/cache/user_bias.pkl', 'rb'))
    ulist=pickle.load(open(u_path, 'rb'))
    rate=pickle.load(open('../ml-1m/rating.pkl', 'rb'))

    predict=np.zeros((6037,3952))

    for  u in range(6037):
        uid = int(ulist[u])
        if uid%300==0:
            print(uid)
        b = bias.iloc[uid, 1]
        nei=np.argsort(-sim[uid])[:n]
        for mid in range(3952):
            r=0
            s=0
            if rate[uid][mid]==0:
                for v in nei:
                    r+=(rate[v][mid]-bias.iloc[v,1])*sim[uid][v]
                    s+=sim[uid][v]
            if s==0:
                s=1
            predict[u][mid]=b+r/s
    dump_path='../ml-1m/19/user_attr_predict_%s.pkl'%(n)
    pickle.dump(predict,open(dump_path,'wb'))


def ua_percision():
    new_path = '../ml-1m/new/result.csv'
    csvfile = open(new_path, 'w', newline='')
    writer = csv.writer(csvfile, dialect='excel')
    writer.writerow(['kind', 'nei_i', 'nei_j', 'n', 'count', 'precision', 'recall'])

    nei_num = [3, 5, 7, 9, 10, 11, 13, 15]
    n_list=[3, 5, 8, 10, 14, 16, 18, 20]
    for i in nei_num:
        for j in nei_num:
            for n in n_list:
                path_a = '../ml-1m/19/user_attr_predict_%s.pkl' % i
                path_c = '../ml-1m/new/predict_%s.pkl' % j
                print(i, j, "ua_c")
                count = percision_ua_c(path_a, path_c, n)
                writer.writerow(['ua_c', i, j ,n, count, count/(n*6037), count/2000276])
                print(i, j, "ua")
                count = percision(path_a, n)
                writer.writerow(['ua', i, j, n, count, count/(n*6037), count/2000276])
                print(i, j, "c")
                count = percision(path_c, n)
                writer.writerow(['c', i, j, n, count, count/(n*6037), count/2000276])

    csvfile.close()



def user_100k():
    path= '../ml-100k/u.user'
    new_path = '../ml-100k/user.csv'
    csvfile = open(new_path, 'w', newline='')
    writer = csv.writer(csvfile, dialect='excel')
    writer.writerow(['uid', 'age', 'gender', 'occu','zip'])
    user = open(path, 'r', errors='ignore')
    for u in user:
        line=u.strip().split('|')
        writer.writerow(line)
    csvfile.close()


def data_100k():
    path= '../ml-100k/u.data'
    new_path = '../ml-100k/data.csv'
    csvfile = open(new_path, 'w', newline='')
    writer = csv.writer(csvfile, dialect='excel')
    writer.writerow(['uid', 'mid', 'rating', 'time'])
    user = open(path, 'r', errors='ignore')
    for d in user:
        line=d.split(' ')
        for l in line:
            l=l.strip()
        writer.writerow(line)
    csvfile.close()


def user_feat_100k():
    path = '../ml-100k/user.csv'
    user=pd.read_csv(path)
    user['uid']=user['uid'].apply(lambda x: x-1)
    age_df = pd.get_dummies(user['age'], prefix='age')
    gender_df = pd.get_dummies(user['gender'], prefix='gender')
    occu_df = pd.get_dummies(user['occu'], prefix='occu')
    user = pd.concat([user['uid'], gender_df, age_df, occu_df], axis=1)
    dump_path='../ml-100k/cache/user.pkl'
    pickle.dump(user, open(dump_path, 'wb'))


def r_100k():
    data_fields = ['uid', 'mid', 'rating', 'time']
    path = '../ml-100k/u.data'
    data = pd.read_table(path, names=data_fields)

    train_path = '../ml-100k/train.csv'
    rfile = open(train_path, 'w', newline='')
    rwriter = csv.writer(rfile, dialect='excel')
    rwriter.writerow(['uid', 'mid', 'rating'])

    test_path = '../ml-100k/test.csv'
    sfile = open(test_path, 'w', newline='')
    swriter = csv.writer(sfile, dialect='excel')
    swriter.writerow(['uid', 'mid', 'rating'])

    split = 0.2
    for line in data.itertuples():
        l=[line[1], line[2], line[3]]
        if random.random() < split:
            swriter.writerow(l)
        else:
            rwriter.writerow(l)

    rfile.close()
    sfile.close()


def rating_100k():
    path= '../ml-100k/train.csv'
    data=pd.read_csv(path)
    data['uid']=data['uid'].apply(lambda x: x-1)
    data['mid']=data['mid'].apply(lambda x: x-1)

    org_rating=np.zeros((943,1682))
    kbias=data.groupby(['uid'], as_index=False)['rating'].mean()
    rating=np.zeros((943,1682))

    for line in data.itertuples():
        uid=int(line[1])
        mid=int(line[2])
        org_rating[uid][mid] = int(line[3])

    for i in range(943):
        rating[i] = org_rating[i]-kbias.iloc[i, 1]

    print(rating[:5])

    dump1='../ml-100k/cache/org_rating.pkl'
    dump2='../ml-100k/cache/rating.pkl'

    bias_path='../ml-100k/cache/bias.pkl'
    pickle.dump(org_rating, open(dump1, 'wb'))
    pickle.dump(rating, open(dump2, 'wb'))
    pickle.dump(kbias, open(bias_path, 'wb'))


def rating_sim_100k():
    path='../ml-100k/cache/rating.pkl'
    rate = pickle.load(open(path, 'rb'))
    mode = np.zeros(943)
    sim = np.zeros((943, 943))
    for i in range(943):
        mode[i] = np.sum(np.multiply(rate[i], rate[i]))**0.5

    for i in range(943):
        for j in range(i):
            sim[i][j] = sim[j][i]
        for j in range(i+1, 943):
            sim[i][j] = np.sum(np.multiply(rate[i], rate[j]))/(mode[i]*mode[j])

    print('rating', sim[:5, :5])

    dump_path='../ml-100k/cache/rating_sim.pkl'
    pickle.dump(sim, open(dump_path, 'wb'))



def ua_sim_100k():
    user = pickle.load(open('../ml-100k/cache/user.pkl', 'rb'))
    user = user.values
    user_attr_s = np.zeros((943, 943))

    for i in range(943):
        if i % 300 == 0:
            print(i)
        for j in range(i):
            user_attr_s[i][j] = user_attr_s[j][i]
        for j in range(i + 1, 943):
            user_attr_s[i][j] = jaccard_similarity_score(user[i], user[j])

    print('attr', user_attr_s[:5, :5])

    dump_path = '../ml-100k/cache/user_attr_sim.pkl'
    pickle.dump(user_attr_s, open(dump_path, 'wb'))


def predict_rate_100k(n):
    path = '../ml-100k/cache/rating_sim.pkl'
    sim = pickle.load(open(path, 'rb'))
    bias = pickle.load(open('../ml-100k/cache/bias.pkl', 'rb'))
    rate = pickle.load(open('../ml-100k/cache/rating.pkl', 'rb'))

    predict = np.zeros((943, 1682))

    for u in range(943):
        if u % 100 == 0:
            print(u)
        b = bias.iloc[u, 1]
        nei = np.argsort(-sim[u])[:n]
        for mid in range(1682):
            r = 0
            s = 0
            if rate[u][mid] == -b:
                for v in nei:
                    if rate[v][mid] != -bias.iloc[v, 1]:
                        r += rate[v][mid] * sim[u][v]
                        s += sim[u][v]
                if s == 0:
                    s = 1
                predict[u][mid] = b + r / s

    dump_path = '../ml-100k/cache/user_rate_predict_%s.pkl' % (n)
    pickle.dump(predict, open(dump_path, 'wb'))




def predict_attr_100k(n):
    path = '../ml-100k/cache/user_attr_sim.pkl'
    sim = pickle.load(open(path, 'rb'))
    bias = pickle.load(open('../ml-100k/cache/bias.pkl', 'rb'))
    rate = pickle.load(open('../ml-100k/cache/rating.pkl', 'rb'))

    p = np.zeros((943, 1682))

    for u in range(943):
        if u % 100 == 0:
            print(u)
        b = bias.iloc[u, 1]
        nei = np.argsort(-sim[u])[:n]
        for mid in range(1682):
            r = 0
            s = 0
            if rate[u][mid] == -b:
                for v in nei:
                    if rate[v][mid] != -bias.iloc[v, 1]:
                        r += rate[v][mid] * sim[u][v]
                        s += sim[u][v]
                if s == 0:
                    s = 1
                p[u][mid] = b + r / s

    dump_path = '../ml-100k/cache/user_attr_predict_%s.pkl' % (n)
    print(np.sort(p[:5]))
    pickle.dump(p, open(dump_path, 'wb'))


def gen_test_100k():
    path = '../ml-100k/test.csv'
    data = pd.read_csv(path)
    data['uid'] = data['uid'].apply(lambda x: x-1)
    data['mid'] = data['mid'].apply(lambda x: x-1)
    print(len(data['rating']))

    rating=np.zeros((943,1682))

    for line in data.itertuples():
        uid = int(line[1])
        mid = int(line[2])
        rating[uid][mid] = int(line[3])

    print(rating[:5])

    dump_path='../ml-100k/cache/test.pkl'
    pickle.dump(rating,open(dump_path, 'wb'))


def run_100k():
    nlist = [3, 5, 7, 9, 10, 11, 13, 15]
    for n in nlist:
        predict_attr_100k(n)
        print("attr", n)
        print('_____________________')
        predict_rate_100k(n)
        print("rate", n)
        print('_____________________')


def percision_ua_c_100k(path1, path2, n):
    pre1 = pickle.load(open(path1, 'rb'))
    pre2 = pickle.load(open(path2, 'rb'))

    true_rate = pickle.load(open('../ml-100k/cache/test.pkl', 'rb'))
    bias = pickle.load(open('../ml-100k/cache/bias.pkl', 'rb'))
    count = 0

    pre = pre1 + pre2

    for i in range(943):
        p1 = np.argsort(pre[i])[-n:]
        for m in p1:
            if true_rate[i][int(m)] > bias.iloc[i, 1]:
                count = count+1

    print(n, count)
    print('---------------------------------')
    return count

def percision_100k(path, n):
    pre = pickle.load(open(path, 'rb'))

    true_rate = pickle.load(open('../ml-100k/cache/test.pkl', 'rb'))
    bias = pickle.load(open('../ml-100k/cache/bias.pkl', 'rb'))
    count = 0

    for i in range(943):
        p1 = np.argsort(pre[i])[-n:]

        for m in p1:
            if true_rate[i][int(m)] > bias.iloc[i, 1]:
                count = count + 1

    print(n, count)
    print('---------------------------------')
    return count


def ua_percision_100k():
    #run_100k()
    new_path = '../ml-100k/cache/resultk.csv'
    csvfile = open(new_path, 'w', newline='')
    writer = csv.writer(csvfile, dialect='excel')
    writer.writerow(['kind', 'nei_i', 'nei_j', 'n', 'count', 'precision', 'recall'])

    nei_num = [3, 5, 7, 9, 10, 11, 13, 15]
    n_list = [23,25,27,30,35,40]
    for i in nei_num:
        for j in nei_num:
            for n in n_list:
                path_a = '../ml-100k/cache/user_attr_predict_%s.pkl' % i
                path_c = '../ml-100k/cache/user_rate_predict_%s.pkl' % j
                print(i, j, "ua_c")
                count = percision_ua_c_100k(path_a, path_c, n)
                writer.writerow(['ua_c', i, j, n, count, count/(n*942), count/20090])
                print(i, j, "ua")
                count = percision_100k(path_a, n)
                writer.writerow(['ua', i, j, n, count, count/(n*942), count/20090])
                print(i, j, "c")
                count = percision_100k(path_c, n)
                writer.writerow(['c', i, j, n, count, count/(n*942), count/20090])

    csvfile.close()


if __name__ == '__main__':
    pass