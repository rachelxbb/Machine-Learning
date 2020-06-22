import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import log2

#Load data.
def load_data(file):
    fhand = pd.read_csv(file)
    file_object = fhand.rename({'(Occupied':'Occupied', ' Price': 'Price', ' Music': 'Music', ' Location': 'Location', ' VIP': 'VIP', ' Favorite Beer': 'Favorite Beer', ' Enjoy)':'Enjoy'}, axis='columns')
    for title in file_object:
        key_list =[]
        if title == 'Occupied':
            for k in file_object['Occupied']:
                result = "".join (j for j in k if not j.isdigit()).replace(": ", "")
                key_list.append(result)
            file_object['Occupied'] = key_list
        elif title == 'Enjoy':
            for k in file_object['Enjoy']:
                result = "".join (j for j in k if j != ";" and j != " ")
                key_list.append(result)
            file_object['Enjoy'] = key_list
        else:
            for k in file_object[title]:
                result = "".join (j for j in k if j != " ")
                key_list.append(result)
            file_object[title] = key_list
    return file_object

#Reorganize the data frame.
def reorganize(file):
    file_object = load_data(file)
    data = file_object.values.tolist()
    label = list(file_object.columns)
    return data, label

#Calculate entropy for attributes.
def calculate_entropy(data):
    label_counts = {}
    
    for v in data:
        label = v[-1]
        if label not in label_counts.keys():
            label_counts[label] = 0
            label_counts[label] += 1
        else:
            label_counts[label] += 1

    count = len(data)
    entropy = 0
    
    for key in label_counts:
        p = float(label_counts[key]) / count
        entropy -= p * log2(p)
    return entropy

#split the data.
def split_data(data, index, value):
    sub_data = []
    for row in data:
        if row[index] == value:
            sub_row = row[:index]
            sub_row.extend(row[index + 1:])
            sub_data.append(sub_row)
    return sub_data

#Decide the best attribute.
def bestfeat(data, labels):
    entropy_before = calculate_entropy(data)
    label_number = len(labels) - 1
    value_list = []
    i_gain_list = []
    entropy_new = 0
    
    for i in range(label_number):
        values = [rows[i] for rows in data]
        value_uni = set(value_list)
        
        for value in value_uni:
            index = labels.index(label)
            sub_data = split_data(data, index, value)
            p = len(sub_data) / len(data)
            entropy_new += p * calculate_entropy(sub_data)
        i_gain = entropy_before - entropy_new

        i_gain_list.append(i_gain)

        
        if len(i_gain_list) > 1:
            best_i = max(i_gain_list)
            best_feat = labels[i_gain_list.index(best_i)]
        else:
            best_feat = labels[i]

        return best_feat

#calculate the majority for the last attribute.
def majority(lab_list):
    label_count = {}
    maj_var = 0
    maj_key = ''
    for lab_var in lab_list:
        if lab_var not in label_count.keys():
            label_count[lab_var] = 0
        else:
            label_count[lab_var] += 1

    for key, value in label_count.items():
        if value > maj_var:
            maj_key = key
    return maj_key

#create the decision tree.
def create_dt(data, labels):    
    lab_list = [row[-1] for row in data]

    if lab_list.count(lab_list[-1]) == len(lab_list):
        return lab_list[-1]
    if len(data[0]) == 1:
        return majority(lab_list)
   
    feat_value = bestfeat(data, labels)
    feat_index = labels.index(feat_value)
    decision_tree = {feat_value: {}}
    
    feat_val = [row[feat_index] for row in data]
    unique_feat_val = set(feat_val)

    for var in unique_feat_val:
        sub_labels = labels[:feat_index] + labels[feat_index +1 :]
        sub_data = split_data(data, feat_index, var)
        decision_tree[feat_value][var] = create_dt(sub_data, sub_labels)
    return decision_tree

#print decision tree.
file = 'dt_data.txt'
data, labels = reorganize(file)
dt = create_dt(data, labels)

#make a prediction.
def prediction(dt):
    pre_data =  ['Moderate', 'Cheap', 'Loud', 'City-Center', 'No', 'No']
    pre_labels = reorganize('dt_data.txt')[1][-1]
    feat = ''.join(dt.keys())
    sub_dt = dt[feat]
    index = labels.index(feat)
    label = 'Yes'
    for key in sub_dt.keys():
        if data[index] == key:
            if sub_dt[key] == dict:
                label = prediction(sub_dt[key])
            else:
                label = sub_dt[key]
    
    return label

pre_data =  ['Moderate', 'Cheap', 'Loud', 'City-Center', 'No', 'No']
pre_labels = reorganize('dt_data.txt')[1][-1]
prediction(dt)


