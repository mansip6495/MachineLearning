from argparse import ArgumentParser
import sys,os,webbrowser
import pandas as pd
from scipy.spatial import distance_matrix,distance
import numpy as np
from feature_extraction import hog,lbp,cm
import numpy as np
import random

def max_a_mean(data,c):

    initial_centroids = []
    image_list = data.index
    # random.seed(1)
    random_point = image_list[random.randrange(len(image_list))]

    dist_mat = distance_matrix(data.values[:, 1:],data.values[:, 1:])
    # dist_mat = pandas.DataFrame(dist_mat, index=image_list, columns=image_list)
    max_index = np.argmax(dist_mat[random_point])
    initial_centroids.append(max_index)
    point1 = initial_centroids[0]
    max_index = np.argmax(dist_mat[point1])
    # max_index = dist_mat[point1].idxmax()
    initial_centroids.append(max_index)

    for index in range(2, c):
        max_dist_sum = 0
        for candidate in image_list:
            dist_sum = 0
            for image in initial_centroids:
                dist_sum += dist_mat[image][candidate]
            if dist_sum > max_dist_sum:
                max_index = candidate
                max_dist_sum = dist_sum
        initial_centroids.append(max_index)

    centroid = np.empty([c,len(data.values[0])-1])
    print(initial_centroids)
    i=0
    for index in initial_centroids:
        centroid[i] = data.values[index,1:]
        i += 1

    return centroid


def kmeans(input,c):
    data = input.values
    # centroids = data[:c,1:]
    centroids = max_a_mean(input,c)

    while True:
        #compute distance matrix
        dist = distance_matrix(data[:,1:],centroids)
        cluster_dict = dict()
        cluster = dist.argmin(axis=1)
        #find group of points in each cluster
        for i in range(len(cluster)):
            temp = data[i]
            temp = temp.reshape(1, len(data[i]))
           # print(temp)
            if str(cluster[i]) not in cluster_dict:
                cluster_dict[str(cluster[i])] = temp
            else:
                cluster_dict[str(cluster[i])] = np.concatenate((cluster_dict[str(cluster[i])],temp))

        updated_centroid = np.zeros((0,len(data[0])-1))

        #recompute centroid of each cluster
        for key,value in cluster_dict.items():
            mean = np.mean(value[:,1:],axis=0)
            mean = mean.reshape(1,len(mean))

            updated_centroid = np.concatenate((updated_centroid, mean))

        # if old centroids equal new centroids then break
        if np.array_equal(centroids,updated_centroid):
            break
        else:
            centroids = updated_centroid

    return cluster_dict,updated_centroid


def metadata_labels(metadata_file,label):
    data = pd.read_csv(metadata_file)
    df = pd.DataFrame(data, columns=['gender', 'accessories', 'aspectOfHand', 'imageName'])
    if (label.lower() == 'right-hand'):
        d = df[df['aspectOfHand'].str.contains('right')]
    elif (label.lower() == 'left-hand'):
        d = df[df['aspectOfHand'].str.contains('left')]
    elif (label.lower() == 'dorsal'):
        d = df[df['aspectOfHand'].str.contains('dorsal')]
    elif (label.lower() == 'palmar'):
        d = df[df['aspectOfHand'].str.contains('palmar')]
    elif (label.lower() == 'male'):
        d = df[df['gender'].str.match('male')]
    elif (label.lower() == 'female'):
        d = df[df['gender'].str.match('female')]
    elif (label.lower() == 'with accessories'):
        d = df[df['accessories'].astype(str).str.match('1')]
    elif (label.lower() == 'without accessories'):
        d = df[df['accessories'].astype(str).str.match('0')]
    else:
        d=df
    label_list = d['imageName'].tolist()
    return label_list


def create_html(dorsal,palmar,filename,img_folder):
    html_op = ("<html><head><title>%s</title></head><body><h2><b>Output for Task %d</b></h2>" % (filename,2))
    for key,value in dorsal.items():
        html_op += ("<h3>Label:%s &nbsp; Cluster Id:%s</h3>"%("Dorsal",str(int(key)+1)))
        html_op += ("<table>")
        i=0
        for img in value:
            if i%6 == 0:
                if i != 0:
                    html_op += ("</tr>")
                html_op += ("<tr>")
                html_op += ("<td><div style='text-align:center'><img src='%s%s' width=200 height=200>%s</div></td>" % (img_folder, img[0],img[0][:-4]))
                i += 1
            else:
                html_op += ("<td><div style='text-align:center'><img src='%s%s' width=200 height=200>%s</div></td>"%(img_folder, img[0],img[0][:-4]))
                i += 1

        html_op += ("</tr></table>")

    for key, value in palmar.items():
        html_op += ("<h3>Label:%s &nbsp; Cluster Id:%s</h3>" % ("Palmar", str(int(key) + 1)))
        html_op += ("<table>")
        i = 0
        for img in value:
            if i % 6 == 0:
                if i != 0:
                    html_op += ("</tr>")
                html_op += ("<tr>")
                html_op += ("<td><div style='text-align:center'><img src='%s%s' width=200 height=200>%s</div></td>" % (
                img_folder, img[0], img[0][:-4]))
                i += 1
            else:
                html_op += ("<td><div style='text-align:center'><img src='%s%s' width=200 height=200>%s</div></td>" % (
                img_folder, img[0], img[0][:-4]))
                i += 1
        html_op += ("</tr></table>")

    html_op += "</body></html>"

    file = open(filename,"w")
    file.write(html_op)
    webbrowser.open('file://' + os.path.realpath(filename))


def print_cluster_details(dorsal,dorsal_centroid,palmar,palmar_centroid):
    for key,value in dorsal.items():
        print("Label:Dorsal\t Cluster Id:%s"%(str(int(key)+1)))
        print("Centroid:")
        print(dorsal_centroid[int(key)])
        print("Clusters:")
        print(value)

    for key,value in palmar.items():
        print("Label:Palmar\t Cluster Id:%s"%(str(int(key)+1)))
        print("Centroid:")
        print(palmar_centroid[int(key)])
        print("Cluster:")
        print(value)


# Input arguments
def read_argument(argv):
    parser = ArgumentParser()
    parser.add_argument("-c", "--cluster_count", type=int, help="Number of clusters")
    parser.add_argument("-lf", "--labelled_folder", type=str, help="labelled image folder path")
    parser.add_argument("-ulf", "--unlabelled_folder", type=str, help="unlabelled image folder path")
    parser.add_argument("-m", "--labelled_metadata", type=str, help="Metadata file")
    parser.add_argument("-um", "--unlabelled_metadata", type=str, help="Metadata file")
    return parser.parse_args(argv)


# Fetch input paramaters
args = read_argument(sys.argv[1:])
c = args.cluster_count
labelled_folder = args.labelled_folder
unlabelled_folder = args.unlabelled_folder
metadata =  args.labelled_metadata
unlabelled_metadata =  args.unlabelled_metadata

# Compute feature descriptors for labelled data
csv_file ='hog_labelled.csv'
# csv_file ='lbp_labelled.csv'
# csv_file ='cm_labelled.csv'
if not os.path.exists(csv_file):
    hog(labelled_folder,csv_file)
df = pd.read_csv(csv_file, sep=',', header=None)

label_dorsal = metadata_labels(metadata, 'dorsal')
label_palmar = metadata_labels(metadata, 'palmar')
df_dorsal = df.loc[df[0].isin(label_dorsal)]
df_palmar = df.loc[df[0].isin(label_palmar)]


df_dorsal = df_dorsal.reset_index(drop=True)
df_palmar = df_palmar.reset_index(drop=True)

# Compute clusters for dorsal and palmar data
cluster_dorsal,centroid_dorsal = kmeans(df_dorsal,c)
cluster_palmar,centroid_palmar = kmeans(df_palmar,c)

# Cluster details command line and html format

# Compute feature descriptors for unlabelled data
csv_file ='hog_unlabelled.csv'
# csv_file ='lbp_unlabelled.csv'
# csv_file ='cm_unlabelled.csv'

if os.path.exists(csv_file):
    os.remove(csv_file)

hog(unlabelled_folder,csv_file)

df = pd.read_csv(csv_file, sep=',', header=None)

# print(df)
dist_dorsal = distance_matrix(df.values[:,1:],centroid_dorsal)
dist_palmar = distance_matrix(df.values[:,1:],centroid_palmar)

data = pd.read_csv(unlabelled_metadata)
positive = 0
negative = 0
total_count = len(df.values)
# print(total_count)
for i in range(len(df.values)):
    # print(df.values[i][0])
    if min(dist_dorsal[i]) < min(dist_palmar[i]):
    # if np.sum(dist_dorsal[i]) < np.sum(dist_palmar[i]):
        predicted_label = "dorsal"
    else:
        predicted_label = "palmar"


    # print(predicted_label)
    # print(data[data["imageName"].str.match(df.values[i][0])]["aspectOfHand"][0:7])
    aspectofhand = data[data["imageName"].str.match(df.values[i][0])]["aspectOfHand"]
    expected_label = aspectofhand.values[0][0:6]
    if predicted_label == expected_label:
        positive += 1
    else:
        negative += 1

print_cluster_details(cluster_dorsal,centroid_dorsal,cluster_palmar,centroid_palmar)
print("Positive:%d, Negative:%d, Total count: %d"%(positive,negative,total_count))
print("Accuracy:%f"%(positive/total_count))
create_html(cluster_dorsal,cluster_palmar,"task2_output.html",labelled_folder)


