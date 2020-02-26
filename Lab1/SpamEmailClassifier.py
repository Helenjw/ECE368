#%%
import os.path
import numpy as np
import matplotlib.pyplot as plt
import util
#%%

#%%
def learn_distributions(file_lists_by_category):
    """
    Estimate the parameters p_d, and q_d from the training set
    
    Input
    -----
    file_lists_by_category: A two-element list. The first element is a list of 
    spam files, and the second element is a list of ham files.

    Output
    ------
    probabilities_by_category: A two-element tuple. The first element is a dict 
    whose keys are words, and whose values are the smoothed estimates of p_d;
    the second element is a dict whose keys are words, and whose values are the 
    smoothed estimates of q_d 
    """
    
    # Build complete set of vocabulary from both ham and spam training sets
    vocabulary = dict()
    for file_list in file_lists_by_category:
        for file in file_list:
            vocabulary.update( (el,0) for el in util.get_words_in_file(file) )
    
    # Populate complete set of vocabulary with frequencies by category
    vocabulary_by_category = []
    for file_list in file_lists_by_category:
        cat_vocab = dict.copy(vocabulary)
        cat_vocab.update( util.get_word_freq(file_list) )
        vocabulary_by_category.append(cat_vocab)

    # N: training samples per category, D: total unique words
    N = [ sum(vocabulary_by_category[0].values()), sum(vocabulary_by_category[1].values()) ]
    D = len(vocabulary)


    # Calculate smoothed estimates for each word in vocabulary
    smoothed_estimates = []
    for vocab, n in zip(vocabulary_by_category, N):
        for word, freq in vocab.items():
            vocab[word] = (freq + 1)/(n+D)
        smoothed_estimates.append(vocab)

    return smoothed_estimates
#%%

def classify_new_email(filename, probabilities_by_category, prior_by_category, threshold):
    """
    Use Naive Bayes classification to classify the email in the given file.

    Inputs
    ------
    filename: name of the file to be classified
    probabilities_by_category: output of function learn_distributions
    prior_by_category: A two-element list as [\pi, 1-\pi], where \pi is the 
    parameter in the prior class distribution

    Output
    ------
    classify_result: A two-element tuple. The first element is a string whose value
    is either 'spam' or 'ham' depending on the classification result, and the 
    second element is a two-element list as [log p(y=1|x), log p(y=0|x)], 
    representing the log posterior probabilities
    """

    # Get probabilities of new email being SPAM and HAM
    posterior_prob = []
    for category_probability, prior in zip(probabilities_by_category, prior_by_category):
      temp_prob = np.log(prior);
      words = util.get_words_in_file( filename )
      
      for word in words:
          temp_prob += np.log( category_probability.get(word, 1.0) )
      
      posterior_prob.append( temp_prob )
    

    # Determine email label
    ratio = posterior_prob[0] / posterior_prob[1] 
    if ratio > threshold:
      label = "ham"
    else:
      label = "spam"
    
    return label, posterior_prob
#%%

#%%
if __name__ == '__main__':
    
    # folder for training and testing 
    spam_folder = "data/spam"
    ham_folder = "data/ham"
    test_folder = "data/testing"

    # generate the file lists for training
    file_lists = []
    for folder in (spam_folder, ham_folder):
        file_lists.append(util.get_files_in_folder(folder))
        
    # Learn the distributions    
    probabilities_by_category = learn_distributions(file_lists)
    
    # prior class distribution
    priors_by_category = [0.5, 0.5]
    
    # Store the classification results
    performance_measures = np.zeros([2,2])
    # explanation of performance_measures:
    # columns and rows are indexed by 0 = 'spam' and 1 = 'ham'
    # rows correspond to true label, columns correspond to guessed label
    # to be more clear, performance_measures = [[p1 p2]
    #                                           [p3 p4]]
    # p1 = Number of emails whose true label is 'spam' and classified as 'spam' 
    # p2 = Number of emails whose true label is 'spam' and classified as 'ham' (type 1)
    # p3 = Number of emails whose true label is 'ham' and classified as 'spam' (type 2)
    # p4 = Number of emails whose true label is 'ham' and classified as 'ham' 

    # Classify emails from testing set and measure the performance
    for filename in (util.get_files_in_folder(test_folder)):
        # Classify
        label,log_posterior = classify_new_email(filename, probabilities_by_category, priors_by_category, 1)
        
        # Measure performance (the filename indicates the true label)
        base = os.path.basename(filename)
        true_index = ('ham' in base) 
        guessed_index = (label == 'ham')
        performance_measures[int(true_index), int(guessed_index)] += 1

    template="You correctly classified %d out of %d spam emails, and %d out of %d ham emails."
    # Correct counts are on the diagonal
    correct = np.diag(performance_measures)
    # totals are obtained by summing across guessed labels
    totals = np.sum(performance_measures, 1)
    print(template % (correct[0],totals[0],correct[1],totals[1]))
    
    
    # Trading off type 1 and 2 errors
    type_1_errors = []
    type_2_errors = []
    
    for threshold in np.arange(0.9, 1.4, 0.01):
        performance_measures = np.zeros([2,2])
        
        for filename in (util.get_files_in_folder(test_folder)):
            # Classify
            label,log_posterior = classify_new_email(filename, probabilities_by_category, priors_by_category, threshold)
        
            # Measure performance (the filename indicates the true label)
            base = os.path.basename(filename)
            true_index = ('ham' in base) 
            guessed_index = (label == 'ham')
            performance_measures[int(true_index), int(guessed_index)] += 1

        type_1_errors.append( performance_measures[0,1] )
        type_2_errors.append( performance_measures[1,0] )
    
    
    plt.scatter(type_1_errors, type_2_errors, s=15)
    plt.grid()
    plt.xlabel('Number of Type 1 Errors')
    plt.ylabel('Number of Type 2 Errors')
    plt.title('Type 1 Error and Type 2 Error Tradeoff')
    plt.show()
