# CS-4780-Spring-2024-Kaggle-Competition
Submission for the CS 4780:  Intro to Machine Learning course during the Spring 2024 semester. 
This course is now known as CS 3780.

This version of the course and competition were offered through Professors Killian Weinberger and Karthik Sridharan. 
After learning about a various number of machine learning algorithms/models it was time to put our knowledge to the test with this competition. 

Throughout the span of this competition I was able to implement both the logisitic regression and SVM model that we were taught in the course. The rest of the models were ones I did research on my own in an attemot to achieve the best test accuracy. I began at 90%, and was able to bump it up to an improved 96.67% overall. The Python file provided should be ran through Google Colab or Jupyter to see the results of each model.

Below are the instructions and the rules provided for the competition:

**CS4780 Spring 2024 Kaggle Competition - Hearts**

**Task**

You are given a train / validation / test split of the patient data. Your task is to predict the "label", i.e. whether the patient—based on the provided markers—has heart disease.

**Dataset (basic)**

Each dataset (train, validation, test) is formatted as a .csv file where each row is a separate patient and each column represents some feature.

Train and validation sets contain their corresponding target labels as values corresponding to the column label.

**Evaluation**

Note that there is a public leaderboard and a private leaderboard. When you submit predictions on the provided test data, about half of the test data will be evaluated to create a public leaderboard score (which everyone can see). The other half will be evaluated for the private leaderboard score (which only the staff can see). The final placings will be based on the private leaderboard score.

The scores will be computed as the % of correct classifications on the submitted predictions.

For full credit your score on the private test cases should be > 92 %.

**Submission File**

You should submit a file submission.csv in the following format:

id,label
7,1

53,0

9,0

...

Thank you to both professors for this project! I hope to work more on machine learning projects within the future!

