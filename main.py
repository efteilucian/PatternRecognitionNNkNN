import math
import numpy as np
import cv2 as cv
from numpy import linalg as la
import time
from statistics import mode

def NN(img, matrix, norm):
    z = np.zeros(len(matrix[0]))
    for i in range(0, 320):
        if norm != 'cosine':
            z[i] = la.norm(matrix[:, i] - img, norm)
        else:
            dot_prod = np.dot(matrix[:, i], img)
            norm_prod = la.norm(matrix[:, i]) * la.norm(img)
            z[i] = 1 - dot_prod / norm_prod
    position = z.argmin()
    return position//8+1, position

def KNN(img, matrix, norm, k):
    z = np.zeros(len(matrix[0]))
    for i in range(0,320):
        if norm != 'cosine':
            z[i] = la.norm(matrix[:, i] - img, norm)
        else:
            dot_prod = np.dot(matrix[:, i], img)
            norm_prod = la.norm(matrix[:, i]) * la.norm(img)
            z[i] = 1 - dot_prod/norm_prod
    positions = np.argsort(z)
    positions = positions[:k]
    persons = positions//8 + 1
    found_person = mode(persons)
    return found_person, (found_person-1)*8

def testImages(k, norm):
    numCorrect = 0
    totalQueryTime = 0

    for i in range(1, numPersons + 1):
        pathT = folderPath + "\\s" + str(i)
        for j in range(numTrainingPictures + 1, numPicturesPerPerson + 1):
            imgPathT = pathT + "\\" + str(j) + ".pgm"
            imgT = cv.imread(imgPathT, 0)
            imgT = imgT.reshape(-1, )
            t0 = time.perf_counter()
            person, position = KNN(imgT, A, norm, k)
            t1 = time.perf_counter()
            totalQueryTime += t1 - t0
            if person == i:
                numCorrect += 1

    rr = numCorrect / numTests
    aqt = totalQueryTime / numTests
    rr_results.append(rr)
    aqt_results.append(aqt)
    print(f"Statistics for k = {k} and norm = {norm}")
    print(f"Recognition rate: {rr*100:.5f}%")
    print(f"Average query time: {aqt:.8f} s")
    print()


if __name__ == '__main__':
    folderPath = "C:\\Users\\Lucian\\Downloads\\ORL"

    A = np.zeros((112*92, 320))

    numPersons = 40
    numPicturesPerPerson = 10
    numTrainingPictures = 8
    numTests = numPersons * (numPicturesPerPerson - numTrainingPictures)

    for i in range(1, 41):
        path = folderPath + "\\s" + str(i)
        for j in range(1, 9):
            imgPath = path + "\\" + str(j) + ".pgm"
            img = cv.imread(imgPath, 0)
            img = img.reshape(-1, )
            A[:, (i - 1) * 8 + (j - 1)] = img

    rr_results = []
    aqt_results = []
    k_test = [1,3,5,7,9]
    norm_test = [1, 2, np.inf,'cosine']
    for norm in norm_test:
        for k in k_test:
            testImages(k, norm)
        np.savetxt(f"ORL_kNN_norm{norm}.txt", rr_results, fmt="%.8f")