
from textblob import TextBlob

# Example text
text = "I am excited"

# Perform sentiment analysis
blob = TextBlob(text)
sentiment_score = blob.sentiment.polarity  # Polarity ranges from -1 (negative) to 1 (positive)
subjectivity_score = blob.sentiment.subjectivity  # Subjectivity ranges from 0 (objective) to 1 (subjective)


print("Sentence: ",text)
# Print the results
print("Sentiment Score:", sentiment_score)
print("Subjectivity Score:", subjectivity_score)





#https://saifmohammad.com/WebPages/nrc-vad.html  NCR-VAD


#https://dl.acm.org/doi/fullHtml/10.1145/3489141#Bib0007

#https://github.com/JULIELab/EmoBank/tree/master/corpus


#https://pdodds.w3.uvm.edu/teaching/courses/2009-08UVM-300/docs/others/everything/bradley1999a.pdf


#https://github.com/bagustris/text-vad/blob/master/VADanalysis/lib/dictionary_English.txt

#https://github.com/bagustris/text-vad/blob/master/VADanalysis/lib/vad-nrc.csv