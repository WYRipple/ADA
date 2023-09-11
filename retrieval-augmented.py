import random
import pdb
import openpyxl
import numpy as np
import pyswarms as ps
import collections
import json
import math
import copy
np.set_printoptions(suppress=True, threshold=np.inf)
seed = 42
np.random.seed(seed)
random.seed(seed)

def read_line_examples_from_file(data_path, write_path, need_cate=True, need_dag=True, a_cdag=0.01, round=2, up=0.5, need_spl=True, get_result=False):
    all_line = []  
    all_line_aug = [] 
    
    all_dag_dict = {}
    category_dict = {}
    sentence_num = 0
    tuple_num = 0
    all_sentence = []   
    sentence_id = 0

    style1_dict = {}
    style2_dict = {}
    style3_dict = {}
    with open(data_path, 'r', encoding='UTF-8') as fp:
        for line in fp:
            a_o_dag = []
            cate_list = []
            aspect_dict, opinion_dict = {}, {}
            aspect_pos, opinion_pos, category_pos = 0, 0, 0
            this_sentences = {}  
            all_line_aug.append(line)
            line = line.strip()
            all_line.append(line)
            words, tuples = line.split('####')
            this_sentences["sentence_id"] = sentence_id
            sentence_id += 1
            this_sentences["sentence"] = words
            this_sentences["labels"] = tuples
            this_sentences["L-DAG"] = {}
            this_sentences["cate"] = {}
            labels = eval(tuples)
            tuple_num += len(labels)
            for j in range(len(labels)):
                if labels[j][0] not in aspect_dict:
                    if labels[j][0] != "NULL":
                        aspect_dict[labels[j][0]] = aspect_pos
                        aspect_pos += 1
                    else:
                        aspect_dict[labels[j][0]] = -1
                if labels[j][3] not in opinion_dict:
                    if labels[j][3] != "NULL":
                        opinion_dict[labels[j][3]] = opinion_pos
                        opinion_pos += 1
                    else:
                        opinion_dict[labels[j][3]] = -1
                a = aspect_dict[labels[j][0]]
                o = opinion_dict[labels[j][3]]
                a_o_dag.append((a,o))
                cate_list.append(labels[j][1])

                if labels[j][1] not in category_dict:
                    category_dict[labels[j][1]] = 1
                else:
                    category_dict[labels[j][1]] += 1
            
            this_sentences["L-DAG"]["name"] = str(a_o_dag)
            this_sentences["cate"]["name"] = cate_list


            if str(a_o_dag) not in all_dag_dict:
                all_dag_dict[str(a_o_dag)] = 1
            else:
                all_dag_dict[str(a_o_dag)] += 1

            if len(a_o_dag) == 1:
                if str(a_o_dag) not in style1_dict:
                    style1_dict[str(a_o_dag)] = 1
                else:
                    style1_dict[str(a_o_dag)] += 1
            else:
                a_flag = []
                o_flag = []
                is_2 = []
                for i in a_o_dag:
                    a_now = i[0]
                    o_now = i[1]
                    if a_now in a_flag or o_now in o_flag:
                        is_2.append(1)
                    else:
                        a_flag.append(a_now)
                        o_flag.append(o_now)
                if 1 in is_2:
                    if str(a_o_dag) not in style3_dict:
                        style3_dict[str(a_o_dag)] = 1
                    else:
                        style3_dict[str(a_o_dag)] += 1
                else:
                    if str(a_o_dag) not in style2_dict:
                        style2_dict[str(a_o_dag)] = 1
                    else:
                        style2_dict[str(a_o_dag)] += 1
                    
            all_sentence.append(this_sentences)

    all_dag_dict_sorted = dict(sorted(all_dag_dict.items(), key=lambda x: x[1], reverse=True))
    category_dict_sorted = dict(sorted(category_dict.items(), key=lambda x: x[1], reverse=True))
    all_dag_dict_sorted_aug = copy.deepcopy(all_dag_dict_sorted)
    category_dict_sorted_aug = copy.deepcopy(category_dict_sorted)

    round_num = round
    dag_max = all_dag_dict_sorted[list(all_dag_dict_sorted.keys())[0]] * round_num
    cate_max = category_dict_sorted[list(category_dict_sorted.keys())[0]] * round_num

    dag_position = {}
    d_i = 0
    for key in all_dag_dict_sorted.keys():
        dag_position[key] = d_i
        d_i += 1

    cate_position = {}
    c_i = 0
    for key in category_dict_sorted.keys():
        cate_position[key] = c_i
        c_i += 1

    a_dag = a_cdag
    a_cate = a_cdag

    for time in range(20):
        var_information = copy.deepcopy(all_line_aug)
        for i in range(len(all_sentence)):
            this_sentence = all_line[i] 
            dag_name = all_sentence[i]["L-DAG"]["name"]
            dag_flag = True
            if need_dag:
                dag_flag = judge_threshold(a_dag, dag_name, dag_max, all_dag_dict_sorted_aug, dag_position,up)

            this_cate_list = all_sentence[i]["cate"]["name"] 
            cate_flag = True
            if need_cate:
                for iii in range(len(this_cate_list)):
                    cate_name = this_cate_list[iii]
                    flag = judge_threshold(a_cate, cate_name, cate_max, category_dict_sorted_aug, cate_position,up)
                    if flag==False:  
                        cate_flag = False

            if dag_flag and cate_flag:
                k_flag = 0
                k_max=1
                this_time = 0
                while 1:
                    this_time += 1
                    if k_flag >= k_max or this_time > 1000:
                        break
                    rand_int = random.randint(0, len(all_line)-1)

                    random_sentence = all_sentence[rand_int]
                    random_dag_name = random_sentence["L-DAG"]["name"]
                    random_dag_flag = judge_threshold(a_dag, random_dag_name, dag_max, all_dag_dict_sorted_aug, dag_position,up)
                    random_cate_list = random_sentence["cate"]["name"]
                    random_cate_flag = True
                    if need_cate:
                        for ii in range(len(random_cate_list)):
                            random_cate_name = random_cate_list[ii]
                            flag = judge_threshold(a_cate, random_cate_name, cate_max, category_dict_sorted_aug, cate_position,up)
                            if flag==False:  
                                random_cate_flag = False
                    if random_cate_flag and random_dag_flag:
                        aug_sentence = all_line[rand_int]
                        aug_words, aug_tuples = aug_sentence.split('####')
                        if need_spl:
                            this_words, this_tuples = this_sentence.split('####')
                            new_words = this_words + " " + aug_words
                            label1 = eval(this_tuples)
                            label2 = eval(aug_tuples)
                            label_f = label1 + label2
                            label_str = str(label_f)
                            new_sent = new_words + "####" + label_str + "\n"
                            all_line_aug.append(new_sent)
                        else:
                            all_line_aug.append(aug_sentence+"\n")
                            all_line_aug.append(this_sentence+"\n")

                        all_dag_dict_sorted_aug[random_dag_name] += 1
                        all_dag_dict_sorted_aug[dag_name] += 1
                        for ci in range(len(this_cate_list)):
                            add_cate = this_cate_list[ci]
                            category_dict_sorted_aug[add_cate] += 1
                        for rci in range(len(random_cate_list)):
                            add_cate = random_cate_list[rci]
                            category_dict_sorted_aug[add_cate] += 1
                        k_flag += 1
    
        if var_information == all_line_aug:
            break
        
    num_style_1 = 0
    for v in style1_dict.keys():
        num_style_1 += style1_dict[v]
    num_style_2 = 0
    for v in style2_dict.keys():
        num_style_2 += style2_dict[v]
    num_style_3 = 0
    for v in style3_dict.keys():
        num_style_3 += style3_dict[v]
    
    if get_result == False:
        with open(write_path, 'w', encoding='UTF-8') as ap:
            for i in range(len(all_line_aug)):
                ap.write(all_line_aug[i])
    else:
        with open(write_path, 'w', encoding='UTF-8') as ap:
            for c in category_dict_sorted_aug:
                ap.write(c+"\n")
            for c in category_dict_sorted_aug:
                ap.write(str(category_dict_sorted_aug[c])+"\n")
            for d in all_dag_dict_sorted_aug:
                ap.write(d+"\n")
            for d in all_dag_dict_sorted_aug:
                ap.write(str(all_dag_dict_sorted_aug[d])+"\n")

                

def judge_threshold(a_x, name, max, dict_sorted, position,up):   
    this_num = dict_sorted[name]   
    p = this_num / max  
    x_pos = position[name] 
    p_th = math.exp(-a_x*x_pos)+up 
    if p_th > p:
        return True  
    else:
        return False  


rest15_path = ""
rest15_aug_path = ""
read_line_examples_from_file(rest15_path, rest15_aug_path, need_cate=True, need_dag=True, a_cdag=0.05, round=1, up=0.5, need_spl=True, get_result=False)

