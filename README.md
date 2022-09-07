# Romance-and-Fantasy-Synopsis-Detector
Romance and Fantasy Synopsis Detector is an application designed to analyze and classify book synopsis. This application will help users to predict the genre of the book, be it Romance or Fantasy. This application is made using the Python programming language with the help of NLTK data. 


- There are 2 genre files provided, namely "romance_dataset.txt" for romance genre books and ""fantasy_dataset.txt" for fantasy genre books.

- The data will be used to classify a synopsis whether it is included in a romance or fantasy genre book. The data then processed and carried out by Train data with Naive Bayes Classifier and save the model into the file name “model.pickle” using the pickle module.

- Then preprocess the data by lemmatize each word, Stemmer, and delete Stopwords and Punctuations

- The application will ask the user to enter a synopsis of the book. This synopsis input validation must be longer than 15 characters. The model will classify user input into Romance Books and Fantasy Books, then display the results on the screen.

