import math
import glob
import os.path
import re


def tokenize(doc):
    ''' splits a given string into tokens around all non-alphabetical characters
    
    args:
        doc: a string representing an entire document (can contain linebreaks)
    returns:
        a list of alphabetical tokens (all non-empty)
    '''
    token_list = []
    word = ''
    for char in doc:
        # get the ordinal (integer) representation of the character (ASCII code)
        # (comparing with this is faster than searching an explicit list of all 
        #  52 characters each time)
        ch_ord = ord(char)
        if (ch_ord >= 65 and ch_ord <= 90) or (ch_ord >= 97 and ch_ord <= 122):
            # character is alphabetical (A-Z, or a-z), add to current word
            word += char
            # note that you can also use str.isalpha() to do all of this
            # however, that also accepts letters with diacritics, like 'â'
        elif len(word) > 0:
            # non-alphabetical character delimits current word, store it
            token_list.append(word)
            word = ''
    # store final token (would otherwise be missed in loop)
    if len(word) > 0:
        token_list.append(word)
    return token_list
    # much shorter solution with regular expressions (requires module import)
    # return re.findall('[a-zA-Z]+', doc)
    
    # examples for tokens that are not handled properly:
    # "Peter's", "12/21/2020", "365 5th Avenue", "New York", "U.S.A."


def normalize(token_list):
    ''' puts all tokens in a given list in lower case and returns the list 
    (changes can happen in place, i.e., the input itself may change) '''
    for i in range(len(token_list)):
        token_list[i] = token_list[i].lower()
    return token_list
    
    # tokens not normalized to the same although the arguably should be:
    # (several of these are not tokens as produced by the above function
    #  but they could well be tokens in other implementations)
    # "Men"/"Man"
    # "is"/"be"
    # "12/21/2020"/"December 12, 2020"
    # "Lauchstädt"/"Lauchstaedt"
    # "well-being"/"wellbeing"


def getVocabulary(term_lists):
    ''' determines the list of distinct terms for a given list of term lists
    
    args:
        term_lists: a list of lists of normalized tokens / terms (i.e., strings)
    returns:
        a sorted list of all distinct terms in the input, i.e., the index terms
    '''
    # use a set to keep track of terms seen so far
    # (this performs the check in O(1) for each new term; dict can also do this;
    #  better than using a list, which needs O(n))
    vocab = set()
    for term_list in term_lists:
        for term in term_list:
            if term not in vocab:
                vocab.add(term)
    vocab = sorted(list(vocab))
    return vocab


def getInverseVocabulary(vocab):
    ''' produces a mapping from index terms to indices in the vocabulary 
    
    args: 
        vocab: the list of index terms, the vocabulary
    results:
        a dictionary term2id such that vocab[term2id[term]] = term for all terms
    '''
    term2id = {}
    for i, term in enumerate(vocab):
        term2id[term] = i
    # shorter solution with list comprehension
    # return {term: i for i, term in enumerate(vocab)}
    return term2id


def getTermFrequencies(term_list, term2id):
    ''' determines the frequencies of all terms in a given term list
    
    able to handle terms in the list that are not in the vocabulary
    
    args:
        term_list: a list of normalized tokens produced from a document
        term2id: the inverse vocabulary produced by getInverseVocabulary
    returns:
        a vector (list) tfs of term frequencies, including zero entries
        tfs[i] refers to the term for which term2id[term] = i, for all i
    '''
    tfs = [0.0] * len(term2id)
    for term in term_list:
        if term in term2id:
            tfs[term2id[term]] += 1.0
    return tfs


def getInverseDocumentFrequencies(matrix):
    ''' determines the idf of all terms based on counts in given matrix
    
    args:
        matrix: the 2d weight matrix of the document collection (intermediate)
            matrix[i] returns a list of all weights for document i
            matrix[i][j] returns the weight for term j in document i
    returns:
        list of inverse document frequencies, one per term
    '''
    M = len(matrix[0]) # vocabulary size
    N = len(matrix) # number of documents
    idfs = []
    
    # loop over all columns (terms) and rows (documents) of the matrix
    for j in range(M):
        df = 0.0
        for i in range(N):
            if matrix[i][j] > 0.0:
                # count new occurrence of term j
                df += 1.0
        idfs.append(math.log10(N/df))
    # shorter solution with list comprehension
    # idfs = [math.log10(N/sum([1.0 if matrix[i][j] > 0.0 else 0.0 
    #                           for i in range(N)]))
    #         for j in range(M)]
    return idfs


def logTermFrequencies(tfs):
    ''' turns given list of term freq. into log term freq. and returns it
    (changes can happen in place, i.e., the input itself may change) '''
    for i in range(len(tfs)):
        tfs[i] = 1.0 + math.log10(tfs[i]) if tfs[i] > 0.0 else 0.0
    # shorter solution with list comprehension
    # tfs = [1.0 + math.log10(tf) if tf > 0.0 else 0.0 for tf in tfs]
    return tfs


def getTfIdf(tfs, idfs):
    ''' returns tf.idf weights for given document's term freq. and given idfs 
    
    args:
        tfs: term frequencies of one document, i.e. one row in the matrix
        idfs: inverse document frequencies for the collection
    returns:
        list of tf.idf weights, i.e., elementwise product of the two input lists 
    '''
    tf_idfs = []
    for i in range(len(tfs)):
        tf_idfs.append(tfs[i] * idfs[i])
    # shorter solution with list comprehension
    # tf_idfs = [tfs[i] * idfs[i] for i in range(len(tfs))]
    return tf_idfs


def normalizeVector(vector):
    ''' normalizes a vector by dividing each element by the L2 norm 
    (changes can happen in place, i.e., the input itself may change)
    
    args:
        vector: a list of numerical values, e.g. log term frequencies
    returns:
        the same vector, normalized
    '''
    # compute L2 norm of the vector for normalization
    L2 = 0.0
    for w in vector:
        L2 += w * w
    L2 = math.sqrt(L2)
    # shorter solution with list comprehension
    # L2 = math.sqrt(sum([w*w for w in vector]))
    for i in range(len(vector)):
        vector[i] = vector[i] / L2 if L2 > 0.0 else 0.0
    # shorter solution with list comprehension
    # vector = [w / L2 if L2 > 0.0 else 0.0 for w in vector]
    return vector


def dotProduct(v1, v2):
    ''' returns the dot product of two input vectors '''
    dp = 0.0
    for i in range(len(v1)):
        dp += v1[i] * v2[i]
    return dp
    # shorter solution with list comprehension
    # return sum([v1[i] * v2[i] for i in range(len(v1))])


def runQuery(query, k, matrix, term2id):
    ''' executes a given query using a given weight matrix 
    
    processes the query to obtain a vector of normalized log term frequencies,
    then returns the top k documents 
    
    args:
        query: a string to process for document retrieval
        k: the (maximum) number of documents to return
        matrix: the 2d weight matrix of the document collection
            matrix[i] returns all weights for document i
            matrix[i][j] returns the weight for term j in document i
        term2id: a mapping from terms to indices in the second matrix dimension 
    returns:
        up to k document indices ranked by the match score between the documents 
        and the query; only documents with non-zero score should be returned 
        (so it can be fewer than k)
    '''
    term_list = normalize(tokenize(query))
    print(term_list)
    q = getTermFrequencies(term_list, term2id)
    q = logTermFrequencies(q)
    q = normalizeVector(q)
    scores = []
    for i, d in enumerate(matrix):
        scores.append((dotProduct(q, d), i))
    # shorter solution with list comprehension
    # scores = [(dotProduct(q, d), i) for i, d in enumerate(matrix)] 
    scores.sort(reverse=True)
    results = []
    # note the use of min in case k > N
    for i in range(min(len(scores), k)):
        if scores[i][0] > 0.0:
            results.append(scores[i][1])
        else:
            break
    print(scores[:k])
    # shorter solution with list comprehension
    # results = [v[1] for v in scores[:k] if v[0] > 0.0]
    return results


def main():
    # process all files (tokenization and token normalization)
    term_lists = []
    file_names = []
    for txtFile in glob.glob(os.path.join('data/', '*.txt')):
        with open(txtFile) as tf:
            term_lists.append(normalize(tokenize('\n'.join(tf.readlines()))))
            file_names.append(txtFile)
    # determine the vocabulary and the inverse mapping
    vocab = getVocabulary(term_lists)
    term2id = getInverseVocabulary(vocab)
    print('vocabulary size:', len(vocab))
    
    # compute the weight matrix
    matrix = [[0.0 for i in range(len(vocab))] for j in range(len(term_lists))]
    for i, term_list in enumerate(term_lists):
        matrix[i] = getTermFrequencies(term_list, term2id)
    idfs = getInverseDocumentFrequencies(matrix) 
    for i in range(len(matrix)):
        matrix[i] = logTermFrequencies(matrix[i])
        matrix[i] = getTfIdf(matrix[i], idfs)
        matrix[i] = normalizeVector(matrix[i])
    
    # run some test queries
    docs = runQuery('god', 3, matrix, term2id)
    print([file_names[i] for i in docs], end='\n\n\n')
    docs = runQuery('liberty freedom justice', 3, matrix, term2id)
    print([file_names[i] for i in docs], end='\n\n\n')
    docs = runQuery('Though passion may have strained it must not break our '
                    'bonds of affection', 3, matrix, term2id)
    print([file_names[i] for i in docs], end='\n\n\n')
    docs = runQuery('carnage', 3, matrix, term2id)
    print([file_names[i] for i in docs], end='\n\n\n')
    
    
if __name__ == '__main__':
    main()