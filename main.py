import os
import glob
import numpy as np
import math
from numba import jit

#global
_feature = 196
_max = 180
_outputfolder = "out"
_maharanobisuvalue = 1701
_hiragana = ["あ","い","う","え","お","か","き","く","け","こ","さ","し","す","せ","そ","た","ち","つ","て","と","な","に","ぬ","ね","の","は","ひ","ふ","へ","ほ","ま","み","む","め","も","や","ゆ","よ","ら","り","る","れ","ろ","わ","を","ん"]


###ファルダを作成
def makeFolder():
    try:
        print("/" + _outputfolder + "を生成します")
        os.mkdir(_outputfolder)
    except:
        print("/" + _outputfolder + "はすでに存在します.")

###ファイル読み込み
def openData():
    lists = []
    files = []
    for file in glob.glob('data/*.txt'):
        lines = open(file, 'r').readlines()
        floatData = [float(d.strip("\n")) for d in lines]
        splitData = [a for a in zip(*[iter(floatData)] * _feature)]
        lists.append(splitData)
        files.append(file)
    return (lists,files)

###平均値
def average(data):
    list = [0 for i in range(_feature)]
    for i in range(0, _feature):
        for j in range(0, _feature):
            list[j] += data[i][j]
    list = [d / _feature for d in list]
    return list

###分散共分散行列を計算
def varianceMatrix(data, ave):
    matrix = [[0 for i in range(_max)] for j in range(_max)]
    for i in range(0, _max):
        for j in range(0, _max):
            total = 0
            for a in range(0, _max):
                total += data[a][i] * data[a][j]
            matrix[i][j] = (total / _max) - (ave[i] * ave[j])
    return matrix

###二次元配列を書き出すメソッド
def writeVector(headStr, file, matrixList):
    outfile = file.replace("data/c", _outputfolder+"/"+headStr)
    with open(outfile, "w") as fp:
        for i in range(0, len(matrixList)):
            for j in range(0, len(matrixList[0])):
                fp.write(str(matrixList[i][j])+"\n")
            fp.write("\n")
    print("write "+outfile)

###リストを書き出すメソッド
def writeValue(headStr, file, valueList):
    outfile = file.replace("data/c", _outputfolder+"/"+headStr)
    with open(outfile, "w") as fp:
        for i in range(0, len(valueList)):
            fp.write(str(valueList[i])+"\n")
    print("write "+outfile)

###ヤコビ行列のソート
@jit
def sortJacobi(values, vectors):
    mergeMatrix =  np.c_[values.T, vectors]
    eigenValue, eigenVector = np.hsplit(mergeMatrix[mergeMatrix[:, 0].argsort()], [1])
    return (eigenValue[:, 0].tolist(), eigenVector.tolist())

###マハラノビス距離
@jit
def maharanobisu(x, m, e, l):
    d = 0
    x2 = np.array(x)
    for n in range(180):
        a = x2-m
        aa = np.array(a).T
        b = np.dot(aa, e[n])
        d += (b*b)/(l[n] + _maharanobisuvalue)
    return d

###認識
# @jit
def recognize(dataList, aveList, eigenValues, eigenVectors):
    probability = [0] * 46
    for strNum, data in enumerate(dataList):
        p = 0
        for i in range(179, 199):
            dmin = maharanobisu(data[i], aveList[0], eigenVectors[0], eigenValues[0])
            nummin = 0
            for j in range(46):
                d = maharanobisu(data[i], aveList[j], eigenVectors[j], eigenValues[j])
                if dmin > d:
                    nummin = j
                    dmin = d
            if nummin == strNum:
                p += 1

        print(_hiragana[strNum]+"の認識率 : "+str(p/20))
        probability.append(p / 20)
    return sum(probability)/46

def main():
    #フォルダの作成
    makeFolder()
    #ファイル読み込み
    dataList,files = openData()
    #初期化
    eigenValues = []
    eigenVectors = []
    aveList = []

    print("c+番号.txt: 分散・共分散行列")
    print("v+番号.txt: 固有ベクトル")
    print("d+番号.txt: 固有値")

    #計算
    for file, data in zip(files, dataList):
        ###平均
        ave = np.array(average(data))

        ###分散共分散行列
        matrix = np.cov(data, rowvar=0, bias=1)
        # matrix = varianceMatrix(data, ave)

        ###固有値・固有ベクトル
        va, ve = np.linalg.eig(matrix)
        eigenValue, eigenVector = sortJacobi(va, ve)
        # eigenValue, eigenVector = jacobi(matrix)

        ###append
        eigenValues.append(eigenValue)
        eigenVectors.append(eigenVector)
        aveList.append(ave)

        ###output
        writeVector("c", file, matrix)
        writeVector("v", file, eigenVector)
        writeValue("d", file, eigenValue)
    #認識率
    per = recognize(dataList, aveList, eigenValues, eigenVectors)
    print("全体の認識率 : " + str(per))

if __name__ == '__main__':
    main()