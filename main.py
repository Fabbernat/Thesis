import independent_scripts.tfidf.wic_tfidf_baseline_single as tfidf
obvious_questions = {
    'Does the word "dog" mean the same thing in sentences "The dog barked." and "The dog wagged its tail."?': 'Yes',
    'Does the word "apple" mean the same thing in sentences "I ate an apple." and "He owns Apple Inc."?': 'No'
}
similarities = tfidf.compute_sentence_similarity(obvious_questions)
plottable_results = tfidf.evaluate(similarities=similarities, labels=['N', 'N'], data=obvious_questions)
print('Accuracy:' + plottable_results.get_percentage)
print('Precision:' + plottable_results.get_tf)
print('Recall:' + plottable_results.get_tp)
print('F1:' + plottable_results.get_f1)
print('TP:' + plottable_results.get_tp, 'TN:' + plottable_results.get_tn, 'FP:' + plottable_results.get_fp, 'FN:' + plottable_results.get_fn)
plot(plottable_results)
