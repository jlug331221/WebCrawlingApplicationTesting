from gensim import corpora, models, similarities

def predict_LDA(forms_under_test_feature_vectors):
  base_saved_output = 'training_set_lda_model'

  dictionary = corpora.Dictionary.load_from_text(base_saved_output + '/training_set_dictionary.txt')
  corpus = corpora.MmCorpus(base_saved_output + '/training_corpus.mm')

  model_lda = models.LdaModel.load(base_saved_output + '/LDA.model')

  print(model_lda.show_topics(num_topics=len(corpus)))

  for feature_vector in forms_under_test_feature_vectors:
    print('Feature vector under test: ' + str(feature_vector))

    vec_bow = dictionary.doc2bow(feature_vector)
    vec_lda = model_lda[vec_bow]

    index = similarities.MatrixSimilarity(model_lda[corpus])

    # compare similarity between vec_lda and the rest of the corpus
    sims = index[vec_lda]
    # sort the sims descending order
    sims = sorted(enumerate(sims), key=lambda item: -item[1])

    most_similar = sims[0]

    my_topic = list(dictionary.token2id)[list(dictionary.token2id.values())
      .index(most_similar[0]+1)]

    print('Predicted probability: ' + str(most_similar))
    print('Predicted topic: ', my_topic)
    print()
