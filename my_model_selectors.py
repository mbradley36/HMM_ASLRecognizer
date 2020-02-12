import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        bestScore = float('inf')
        bestModel = GaussianHMM()
        bestNumVariables = None
        n = self.min_n_components
        while n <= self.max_n_components:
            try:
                model= GaussianHMM(n_components=n, covariance_type="diag", random_state=self.random_state, n_iter=1000).fit(self.X, self.lengths)
                logL = model.score(self.X, self.lengths)
                #(num states^2)+(2num_states num_data_points)-1
               # num_states = len(self.sequences)
                num_dataPts = len(self.X[0])
                #parameters = (num_states*num_states) + (2*num_states*num_dataPts)-1
                parameters = (n*n) + (2*n*num_dataPts)-1
                score = -2*logL + parameters*math.log(n)
                if score < bestScore:
                    bestScore = score
                    bestModel = model
            except:
                print("")
            n += 1

        return bestModel


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        bestScore = float('-inf')
        bestNumVariables = None
        n = self.min_n_components
        while n <= self.max_n_components:
            try:
                model= GaussianHMM(n_components=n, covariance_type="diag", random_state=self.random_state, n_iter=1000).fit(self.X, self.lengths)
                logL = model.score(self.X, self.lengths)
                #DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
                wordScores = {}
                for word, (X, lengths) in self.hwords.items():
                    #print(word)
                    if word != self.this_word:
                        try:
                           # X_word, X_lengths = combine_sequences(cv_train_idx, self.sequences)
                            score = model.score(X, lengths)
                            wordScores[word] = score
                        except:
                            pass
                DICscore = logL - np.mean([wordScores[currword] for currword in wordScores.keys()])
                if DICscore > bestScore:
                    bestScore = DICscore
                    bestNumVariables = n
            except:
                 pass
            n += 1

        try:
            bestModel = GaussianHMM(n_components=bestNumVariables, covariance_type="diag", random_state=self.random_state, n_iter=1000).fit(self.X, self.lengths)
        except:
            best_num_components = self.n_constant
            return self.base_model(best_num_components)
        return bestModel


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds
        The hmmlearn library may not be able to train or score all models.
        Implement try/except contructs as necessary to eliminate non-viable models from consideration.



    training = asl.build_training(features_ground)  # Experiment here with different feature sets
    word = 'VEGETABLE'  # Experiment here with different words
    word_sequences = training.get_word_sequences(word)
    split_method = KFold()
    for cv_train_idx, cv_test_idx in split_method.split(word_sequences):
        print("Train fold indices:{} Test fold indices:{}".format(cv_train_idx, cv_test_idx))

    Tip: In order to run hmmlearn training using the X,lengths tuples on the new folds, subsets must be combined
    based on the indices given for the folds. A helper utility has been provided in the asl_utils module named
    combine_sequences for this purpose.

    SelectorCV(sequences, Xlengths, word,
               min_n_components=2, max_n_components=15, random_state=14)
    '''
    allLogLs = []
    finalModel = GaussianHMM()
    finalAvg = float('inf')
    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        bestScore = float('-inf')
        bestModel = GaussianHMM()
        states = self.max_n_components - self.min_n_components
        # TODO implement model selection using CV
        n_splits = 3
        if len(self.sequences) < n_splits:
            n_splits = len(self.sequences)
        if n_splits == 1:
            #implement score w/o kfold
            best_num_components = self.n_constant
            return self.base_model(best_num_components)
        split_method = KFold(n_splits)

        for num_states in range(self.min_n_components, self.max_n_components+1):
            for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                try:
                    X, lengths = combine_sequences(cv_train_idx, self.sequences)
                    XTest, lengthsTest = combine_sequences(cv_test_idx, self.sequences)
                    model= GaussianHMM(n_components=num_states, n_iter=1000).fit(X, lengths)
                    logL = model.score(XTest, lengthsTest)
                    avg = float('inf')
                    self.allLogLs.append(logL)
                    if logL > bestScore:
                        bestScore = logL
                        #retrain the model over all training sets, including test
                        X = X + XTest
                        lengths = lengths + lengthsTest
                        model = GaussianHMM(n_components=num_states, n_iter=1000).fit(X, lengths)
                        bestModel = model
                    #print("Train fold indices:{} Test fold indices:{}".format(cv_train_idx, cv_test_idx))
                except:
                     pass
            #update avg
            if len(self.allLogLs)>0:
                logLs = np.array(self.allLogLs)
                self.finalAvg = abs(logLs.mean())
            #if the best score is better than the avg, update
            if bestScore<self.finalAvg:
                self.finalModel = bestModel

        return self.finalModel
