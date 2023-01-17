# Team Process Mapping Take-Home Task: Liancheng Gong.

Goal: In this pre-test, you will implement a feature extractor that detects sentiment from team conversations. Specifically, given an input of conversation data, you should output: (1) a sentiment label of ‘positive,’ ‘negative,’ or ‘neutral,’ alongside a score for each label, from 0-1. You will run your feature extractor on a dataset of team jury conversations.

You will then write a reflection on how well you think this feature extractor performed on the data. Please write your reflection in this README document.

## 1. What method(s) did you choose?
In 1-2 sentences each, describe your sentiment analysis method(s).

> The first model, cardiffnlp/twitter-roberta-base-sentiment-latest, is a fine-tuned version of the RoBERTa language model for sentiment analysis on ～124M tweets. The link of the model on huggingface is https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest

> The second model, philschmid/distilbert-base-multilingual-cased-sentiment-2, is a fine-tuned version of the DistilBERT language model for sentiment analysis on amazon_reviews_multi dataset. The link of the model on huggingface is https://huggingface.co/philschmid/distilbert-base-multilingual-cased-sentiment-2

## 2. Method evaluation
Next, we would like you to consider how you would evaluate your method. How do you know the classification or quantification of emotion is “right?” Try to think critically!

2a. Open up output/jury_output_chat_level.csv and look at the columns you generated. Do the values “make sense” intuitively? Why or why not?

> As for the first model, the sentiment generated mostly make sense. For example, model 1 returns 81% negative according to the message, "I can see how the family is upset because they feel the mother was disrespected but I can also understand the guy's feelings. Why should he have to work as interpreter for his mother in law?"; returns 79% positive according to the message, "After the edit he done it made it sound like he really loves his family"; returns 60% neutral according to the message, "I think he also tried to utilize other resources such as language learning apps to help her learn".

> As for the second model, the sentiment generated are also reasonable to some extent but may be slightly different from the model 1. To take the same 3 examples above, model 2 returns 42% neutral according to the message, "I can see how the family is upset because they feel the mother was disrespected but I can also understand the guy's feelings. Why should he have to work as interpreter for his mother in law?"; returns 97.7% positive according to the message, "After the edit he done it made it sound like he really loves his family"; returns 75% positive according to the message, "I think he also tried to utilize other resources such as language learning apps to help her learn".

> The overall results are satisfying that the model will not misclassify positive and negative but only the boundary between positive/negative and neutral is vague even for human. Comparing the result from model 1 and model 2, model 2 has a higher rate on positive and negative sentiment, and a lower rate on neutral sentiment. In other words, model 1 weights neutral sentiment more than model 2.  From the examples above, we can see that the sentiment generated by model 1 is more consistent with the message. However, the sentiment generated by model 2 is more accurate. For example, model 2 returns 97.7% positive according to the message, "After the edit he done it made it sound like he really loves his family", which is more accurate than model 1's 79% positive. Therefore, I think model 2 is better than model 1.

> I also created another version removing stopwords and the word "asshole". It was reflected in output/jury_output_chat_level_remove_words.csv. Most of the results is similar to the original version. However, the messages with "asshole" have lower negative scores. For example, before removing stopwords and "asshole", the negative score for model 1 is 84% for the message "I don't think he's an asshole for having the feelings he does, but I do think he is a bit of an asshole for handling it the way that he did without discussing it with his wife first or maybe finding a better way to approach it.". After removing the stopwords and asshole, the negative score is 40% and the generated label change from negative to neutral. This is because the word "asshole" is a strong negative word and it may affect the sentiment score. Thus, removing the stopwords and "asshole" may improve the accuracy of the model.

2b. Propose an evaluation mechanism for your method(s). What metric would you use (e.g., F1, AUC, Accuracy, Precision, Recall)?

> I would use F1-score as the evaluation metric. The F1-score is the harmonic mean of precision and recall, and it is a good measure of a test’s accuracy. It is also a good measure of a test’s robustness, as it takes both false positives and false negatives into account.

2c. Describe the steps you would take in evaluating this method. Be as specific as possible.

> The messages should first be labeled by human into positive, negative, and neutral. 

> Since we directly generate the sentiment score from the fine-tuned model, we can use the score to compare with the label. By comparing three scores for three labels, we can have the generated label from the models. For example, if the label is positive and the positive score is 0.8, we can say that this classification is correct. If the label is positive and there are other score higher than positive score, we can say that this classification is correct. 

> With the labeled data and the generated labels, the model will be evaluated by the F1-score. We may analyze the result of the evaluation metrics and any errors made by the model. Then we can improve the model by tuning the hyperparameters or using other models to find the best performing model.

2d. (OPTIONAL) Implement your proposed steps in 2b, using the DynaSent benchmark dataset (https://github.com/cgpotts/dynasent). Choose the Round 2 test dataset (“Sentences crowdsourced using Dynabench”). Evaluate the model using benchmarks you have proposed, and briefly comment on the quality of your performance.

> After applying the proposed steps, the F1-score for model 1 is 0.58 and the F1-score for model 2 is 0.44. The F1-score is not high enough. I think the reason is that the model is not trained on the DynaSent dataset. The model is trained on the twitter dataset and the amazon_reviews_multi dataset. The DynaSent dataset is different from the two datasets. Thus, the model may not perform well on the DynaSent dataset.

## 3. Overall reflection
3a. How much time did it take you to complete this task? (Please be honest; we are looking for feedback to make sure the task is scoped appropriately, as this is one of the first times we’re using this task.)

> ~15 hours

3b. Finally, provide an overall reflection of your experience. How did you approach this task? What challenge(s) did you encounter? If you had more time, what are additional extensions, improvements, or tests that you would want to implement?

> I think the task is well designed. It is challenging but not too difficult for me. I first read the instructions and understand the overall tasks. When browsing throught the huggingface website, I first filtered all the models for sentimental analysis tasks. I have in total four candidate models, including cardiffnlp/twitter-roberta-base-sentiment-latest, finiteautomata/bertweet-base-sentiment-analysis, cardiffnlp/twitter-xlm-roberta-base-sentiment, and philschmid/distilbert-base-multilingual-cased-sentiment-2. The first three are all based on RoBERTa and trained on tweets while the last model is based on distilbert and trained on multilingual amazon review dataset. Thus, I choose the one with the most download and the second model as my two models.

> Applying the models to the jury_conversations_with_outcome_var.csv is straightforward. I first read the csv file and extract the messages. Then I apply the models to the messages and generate the sentiment scores in the same format as original output. By running featurize.py, I successfully generated the output files with two new columns: sentiment_1 represents the results from model 1 while sentiment_2 represents the results from model 2.

> However, one challenge is that the word "asshole" is strongly negative, which may affect the sentiment score. Thus, I try to remove the word "asshole" from the messages and generate the sentiment scores again. I also remove the stopwords from the messages. The results are in output/jury_output_chat_level_remove_words.csv. The results are similar to the original version. However, the messages with "asshole" have lower negative scores. 

> With generated sentiment scores but without human labels, I can evaluate how good the models perform. However, I find it challenging for me to classify those messages into positive, negative, or neutral. Without context, I may need to read the messages multiple times to make sure that I understand the message. But the overall sentiment result is satisfying from my understanding of the messages.

> After testing the models on the synasent test dataset, I found the models without removing stopwords and "asshole" perform better. Under both cases, model 1 performs better than model 2. However, the F1-score is still not high enough. There requires more work to improve the model.

> If I have more time, 

> > I would like to try other models and compare the results. Currently, I filtered these two models by the introduction and the number of downloads, which lacks more evidence. Other models may perform better. Additionally,  I may also read the paper of the models to understand the models better and choose the optimal models.

> > I would also like to fine tune the model on our dataset and since right now I directly apply models on our dataset. However, currently I don't have human labels for our dataset. If we have human labels, the model can better fit into our dataset through fine tuning and will have a higher accuracy.

> > I would also like to try other evaluation metrics. The F1-score is a good measure of a test’s accuracy. However, it may not be the best measure for our task. Other metrics may perform better.