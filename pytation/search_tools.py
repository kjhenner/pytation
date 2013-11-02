import numpy
import itertools
from numpy import log, array
from scipy import dot, linalg


def tfifd(array):
    array = array
	c, r = array.shape
        for column in xrange(c):
            ctotal = reduce(lambda x, y: x+y, array[column])
            for row in xrange(r):
                array[row, column] = float(array[row, column])
                if array[row, column] != 0:
                    termDocumentOccurrences = self.getTermDocumentOccurrences(row)
                    termFrequency = self.matrix[column][row] / float(wordTotal)
                    inverseDocumentFrequency = log(abs(columns / float(termDocumentOccurrences)))
                    self.matrix[column][row]=termFrequency*inverseDocumentFrequency
    return 

class MatrixMaker(object):

    def __init__(self, columnList, columnLabels):
        self.columnList = columnList
        self.columnLabels = columnLabels
        self.itemList = list(itertools.chain.from_iterable(columnList))
        self.uniqueItems = list(set(self.itemList))
        self.vectorKeywordIndex = self.getVectorKeywordIndex()

        self.matrix = self.makeMatrix()

    def getVectorKeywordIndex(self):
        vectorIndex = {}
        offset = 0
        for item in self.uniqueItems:
            vectorIndex[item] = offset
            offset += 1
        return vectorIndex #(keyword:position)

    def makeVector(self, column):
        vector = [0] * len(self.vectorKeywordIndex)
        for item in column:
            if item in self.uniqueItems:
                vector[self.vectorKeywordIndex[item]] += 1
        return vector

    def makeMatrix(self):
        matrix = [0] * len(self.columnList)
        for i, column in enumerate(self.columnList):
            matrix[i] = self.makeVector(column)
        return matrix

    #Term Frequency Inverse Document Frequency transformation

    def tfidfTransform(self):

        columns = len(self.matrix)
        rows = len(self.matrix[0])

        for column in xrange(0, columns):

            wordTotal = reduce(lambda x, y: x+y, self.matrix[column])

            for row in xrange(0, rows): #For each term

                self.matrix[column][row] = float(self.matrix[column][row])

                if self.matrix[column][row] != 0:

                    termDocumentOccurrences = self.getTermDocumentOccurrences(row)
                    termFrequency = self.matrix[column][row] / float(wordTotal)
                    inverseDocumentFrequency = log(abs(columns / float(termDocumentOccurrences)))
                    self.matrix[column][row]=termFrequency*inverseDocumentFrequency

    #Returns the number of documents in which a given term, specified by row
    #index, occurs. That is, the number of columns for which a given row has
    #an entry greater than zero.

    def smoothMatrix(self):

        columns = len(self.matrix)
        rows = len(self.matrix[0])

        for column in xrange(0, columns):
            for row in xrange(0, rows):
                if self.matrix[column][row] == 0:
                    self.matrix[column][row] += 1


    def getTermDocumentOccurrences(self, row):

        term_document_occurrences = 0

        columns = len(self.matrix)
        rows = len(self.matrix[0])

        for n in xrange(0, columns):
            if self.matrix[n][row] > 0:
                term_document_occurrences += 1

        return term_document_occurrences

    def distance(self, vector1, vector2):
        d = 0
        for i in range(len(vector1)):
            d += (vector1[i] - vector2[i])**2
        return math.sqrt(d)

    def cosine(self, vector1, vector2):
        return float(dot(vector1, vector2) / (linalg.norm(vector1)*linalg.norm(vector2)))

    def printColumns(self):
        print self.columnLabels
        for i in range(len(self.uniqueItems)):
            print self.uniqueItems[i], ' ', [self.matrix[y][i] for y in range(len(self.matrix))]
