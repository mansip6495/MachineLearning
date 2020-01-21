import pickle
import task6_technique
import pandas as pd
import numpy as np
from scipy.spatial import distance

if __name__ == '__main__':
    technique = "SVM"
    with open( 'hog_11k_svd_query_image.pickle', 'rb') as f:
        query_image = pickle.load(f)

    with open( 'hog_11k_svd_similar_images.pickle', 'rb') as f:
        similar_image = pickle.load(f)

    similar_image = np.array(similar_image)
    similar = pd.DataFrame(similar_image[:,1:], index=similar_image[:,0])

    with open( 'hog_11k_svd_20_relevant_images.pickle', 'rb') as f:
        top_20_image = pickle.load(f)

    relevant = list()
    for i in range(len(top_20_image)):
        relevant.append([top_20_image[i][0]])

    test_data_decision_tree = list()
    if technique == "SVM":
        test_data_svm = similar.iloc[:, 0]
    elif technique == "DESCION_TREE":
        for i in range(len(similar)):
            test_data_decision_tree.append(list(similar.iloc[i][0]))

    while 1:
        if len(relevant) < 20:
            print("Breaking condition:Image set has less than 20 images. Exiting...")
            break
        print("20 similar images for the query image %s" % (query_image[0]))
        for i in range(20):
            print(relevant[i][0],end=",")
        print("")
        user_feedback_relevant = input("Enter comma seperated list of relevant images: ")
        user_feedback_irrelevant = input("Enter comma seperated list of irrelevant images: ")

        feedback_relevant = user_feedback_relevant.split(",")
        feedback_irrelevant = user_feedback_irrelevant.split(",")
        if len(feedback_irrelevant) == 0 or feedback_irrelevant[0] == '':
            print("Breaking condition:Irrelevant image count is 0. Exiting...")
            break
        feedback_data_relevant = similar.loc[feedback_relevant]
        feedback_data_relevant["label"] = "relevant"
        feedback_data_irrelevant = similar.loc[feedback_irrelevant]
        feedback_data_irrelevant["label"] = "irrelevant"

        technique_op = list()
        feedback_data = pd.concat([feedback_data_relevant,feedback_data_irrelevant])
        if technique == "SVM":
            x_train = feedback_data.iloc[:,0]
            y_train = feedback_data['label']
            x_test = test_data_svm
            technique_op = task6_technique.svm(x_train, y_train, x_test)
            # technique_op = ['irrelevant', 'irrelevant', 'irrelevant', 'irrelevant', 'relevant', 'irrelevant', 'relevant', 'irrelevant', 'irrelevant', 'irrelevant', 'relevant', 'relevant', 'irrelevant', 'irrelevant', 'irrelevant', 'relevant', 'relevant', 'relevant', 'irrelevant', 'relevant', 'relevant', 'relevant', 'irrelevant', 'irrelevant', 'irrelevant', 'relevant', 'relevant', 'relevant', 'relevant', 'irrelevant', 'relevant', 'relevant', 'irrelevant', 'relevant', 'irrelevant', 'relevant', 'irrelevant', 'irrelevant', 'relevant', 'relevant', 'relevant', 'relevant', 'irrelevant', 'irrelevant', 'irrelevant', 'irrelevant', 'relevant', 'relevant', 'irrelevant', 'relevant', 'relevant', 'irrelevant', 'relevant', 'irrelevant', 'irrelevant', 'relevant', 'irrelevant', 'relevant', 'irrelevant', 'irrelevant', 'irrelevant', 'relevant', 'irrelevant', 'relevant', 'irrelevant', 'irrelevant', 'irrelevant', 'relevant', 'irrelevant', 'irrelevant', 'irrelevant', 'relevant', 'relevant', 'irrelevant', 'irrelevant', 'relevant', 'relevant', 'irrelevant', 'relevant', 'relevant', 'irrelevant', 'relevant', 'irrelevant', 'relevant', 'relevant', 'irrelevant', 'irrelevant', 'relevant', 'relevant', 'relevant', 'relevant', 'relevant', 'relevant', 'irrelevant', 'relevant', 'irrelevant', 'relevant', 'relevant', 'irrelevant', 'irrelevant', 'irrelevant', 'relevant', 'irrelevant', 'relevant', 'irrelevant', 'irrelevant', 'irrelevant', 'relevant', 'relevant', 'irrelevant', 'irrelevant', 'irrelevant', 'relevant', 'relevant', 'irrelevant', 'irrelevant', 'relevant', 'irrelevant', 'relevant', 'relevant', 'relevant', 'relevant', 'irrelevant', 'irrelevant', 'irrelevant', 'irrelevant', 'relevant', 'relevant', 'relevant', 'irrelevant', 'relevant', 'irrelevant', 'relevant', 'relevant', 'relevant', 'relevant', 'relevant', 'relevant', 'irrelevant', 'irrelevant', 'irrelevant', 'relevant', 'relevant', 'irrelevant', 'irrelevant', 'irrelevant', 'relevant', 'relevant', 'relevant', 'irrelevant', 'irrelevant', 'irrelevant', 'irrelevant', 'irrelevant', 'relevant', 'relevant', 'relevant', 'irrelevant', 'relevant', 'irrelevant', 'relevant', 'relevant', 'irrelevant', 'relevant', 'relevant']

        elif technique == "DESCION_TREE":
            train_data = list()
            test_data = list()
            for i in range(len(feedback_data)):
                train_row = list(feedback_data.iloc[i][0])
                train_row.append(feedback_data.iloc[i]["label"])
                train_data.append(train_row)
            technique_op = task6_technique.train_test_data_dcsn(train_data, test_data_decision_tree)

        # exit()
#Hand_0001347.jpg,Hand_0000682.jpg,Hand_0001816.jpg,Hand_0002041.jpg,Hand_0010096.jpg
#Hand_0001907.jpg,Hand_0000654.jpg,Hand_0001426.jpg,Hand_0000673.jpg,Hand_0009767.jpg
        relevant = list()
        rel = list()
        for i in range(len(technique_op)):
            if technique_op[i] == 'relevant':
                img_score = list()
                img_score.append(similar_image[i][0])
                score = distance.euclidean(similar_image[i][1],query_image[1])
                img_score.append(score)
                relevant.append(img_score)
                rel.append(similar_image[i][0])
        relevant = sorted(relevant, key=lambda p: p[1])

        if technique == "SVM":
            test_data_set = similar.loc[rel]
            test_data_svm = test_data_set.iloc[:,0]
        elif technique == "DESCION_TREE":
            test_data_set = similar.loc[rel]
            test_data_decision_tree = list()
            for i in range(len(test_data_set)):
                test_data_decision_tree.append(list(test_data_set.iloc[i][0]))










