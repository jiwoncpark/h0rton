#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 14:29:37 2018

@author: Dartoon
@modified by: Ji Won Park (jiwoncpark) for Python3

Read each seed.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as matt
matt.rcParams['font.family'] = 'STIXGeneral'

seed_name = [ 'seed101',
 'seed102',
 'seed103',
 'seed104',
 'seed105',
 'seed107',
 'seed108',
 'seed109',
 'seed110',
 'seed111',
 'seed113',
 'seed114',
 'seed115',
 'seed116',
 'seed117',
 'seed118']

import os
import glob
dirpath = os.getcwd()
all_teams = True
folders = ['H0rton']
h0_rung1 = {'H0rton': 70.0}
if all_teams:
    for team in ['Tak', 'MartinMillon','freeform','Rathnakumar', 'Rathnakumar1']:
        h0_rung1[team] = 74.151
        folders.append(team)
subnames = []
for folder in folders:
    subnames += sorted(glob.glob('/home/jwp/stage/sl/h0rton/h0rton/tdlmc_utils/rung1_submit_result/{:s}/*'.format(folder)))

label_name = []  
efficiencys, goodnesses, precisions, accuracys = [], [], [] ,[]   
for subname in subnames:
    if 'h0' in subname or 'txt' in subname:
        sub_name = subname
        root_name, ext_name = os.path.splitext(subname) #sub_name[1:].split(".")[-2]
        folder, subname = os.path.split(subname)
        folder_name = os.path.basename(os.path.normpath(folder))
        filename = sub_name
        openfile = open(filename, "r")
        lines = openfile.readlines()
        #filename.readline()
        result = {}
        for i in range(len(lines)): 
            if '\t' in lines[-2]:
                line = lines[i].split('\t')
            else:
                line = lines[i].split(' ')
            if "seed" in line[0] and 'rung1' in line[0]:
                result[line[0]] = np.array([float(line[1]),float(line[2])])
        #==============================================================================
        # Calculate the metric
        #==============================================================================
        submit = []
        for key, value in result.items():
            if value[0]>0:
                submit.append(value)
        submit = np.asarray(submit)
        if len(submit) > 0:
            efficiency = len(submit)/len(result.items())
            print("Efficiency: ", efficiency)
            goodness = round(np.mean(((submit[:,0]-h0_rung1[folder_name])/submit[:,1])**2), 3)
            print("Log goodness: ", np.log10(goodness))
            precision = round(np.mean(submit[:,1]/h0_rung1[folder_name])*100, 3)
            accuracy = round(np.mean((submit[:,0]-h0_rung1[folder_name])/h0_rung1[folder_name])*100, 3)
            label_name.append(filename)
            efficiencys.append(efficiency)
            goodnesses.append(goodness)
            precisions.append(precision)
            accuracys.append(accuracy)
                              
num_boxes = 3
gridshape = (num_boxes, num_boxes)
num_plots = num_boxes**2 - num_boxes
print("Our multivariate grid will therefore be of shape", gridshape, "with a total of", num_plots, "plots")
fig = plt.figure(figsize=(12, 12))
n=1
axes = [[False for i in range(num_boxes)] for j in range(num_boxes)]
catlog = folders
color_list = ['red', 'green', 'blue','c', 'khaki', 'k', 'lime', 'maroon']
ma = ['o','d','s','D','*','p','<','>','^','v','P','X' ]
#axis_lab = ["efficiency", "log10(goodness)", "log10(precision)", "accuracy"]
axis_lab = ["efficiency ", r"log$_{10}(\chi^2)$", "precision (%)", "accuracy (%)"]

#metric_target = [(0,1),(np.log10(0.4),np.log10(2.)),(0,np.log10(6)),(-2,2)]
#metric= [[efficiencys[i], np.log10(goodnesses)[i], np.log10(precisions)[i], accuracys[i]] for i in range(len(label_name))]

metric_target = [(0,1),(np.log10(0.4), np.log10(2.)),(0, 6),(-2,2)]
metric= [[efficiencys[i], np.log10(goodnesses[i]), precisions[i], accuracys[i]] for i in range(len(label_name))]
label = [label_name[i].split('/')[-2] +'-' + label_name[i].split('/')[-1].replace('_','.').replace('-','.').split('.')[-2] for i in range(len(label_name))]

for j in range(num_boxes): # j = column idx
    for i in range(num_boxes): # i = row idx
        if i <= j : # lower triangle
            y_j = j+1
            ax = fig.add_subplot(num_boxes, num_boxes, n)
            plt.setp(ax.spines.values(), linewidth=2) # set property
            ax.tick_params(labelsize=12)
            ax.get_xaxis().set_tick_params(direction='in', width=1.5, length=6)
            ax.get_yaxis().set_tick_params(direction='in', width=1.5, length=6)
            ax.yaxis.set_ticks_position('both')
            ax.xaxis.set_ticks_position('both')
            for k in range(len(metric)): # Plot each metric
                c_ind = [x for x in range(len(catlog)) if catlog[x] in label[k]][0]
                ma_ind = k-k//(len(ma))*len(ma)
                ax.scatter(metric[k][i], metric[k][y_j], color=color_list[c_ind], marker = ma[ma_ind], label = label[k], s=70)
#            label = "_nolegend_"
            # Plot target range
            target_x, target_y = metric_target[i][0], metric_target[y_j][0]
            target_wx, target_wy = metric_target[i][1]-metric_target[i][0], metric_target[y_j][1]-metric_target[y_j][0]
            rectangle = plt.Rectangle((target_x, target_y), target_wx, target_wy, facecolor='gray', alpha =0.2)
            plt.gca().add_patch(rectangle)
            # Plot efficiency
            if i == 0:
                plt.ylabel('j+1={0}={1}'.format(axis_lab[y_j],y_j), fontsize=15)
                plt.ylabel('{0}'.format(axis_lab[y_j]), fontsize=15)
                plt.xlim(0,1.08)   #plot the limit for effciency for x axis
            # Plot goodness
            elif i == 1:
                plt.xlim(-1.2,3)   #plot the limit for goodness for x axis
            # Plot precision
            elif i ==2:
                plt.xlim(0,20)   #plot the limit for precision for x axis
            # Plot accuracy
            if y_j == 3:
                plt.xlabel('i={0}={1}'.format(axis_lab[i],i), fontsize =15)
                plt.xlabel('{0}'.format(axis_lab[i]), fontsize =15)
                plt.ylim(-20,20)   #plot the limit for accuracy for y axis
            elif y_j ==2:
                plt.ylim(0,30)   #plot the limit for precision for y axis
            elif y_j ==1 :
                plt.ylim(-1.2,3)   #plot the limit for goodness for y axis
            if i>=1:
                ax.yaxis.set_ticklabels([])
            if y_j <3:
                ax.xaxis.set_ticklabels([])
        
        n += 1
        if i==1 and j ==0:
            ax.legend(bbox_to_anchor=(3.5, -0.5), loc='center left', borderaxespad=0., prop={'size': 16})
            axes[j][i] = ax

fig.subplots_adjust()
fig.tight_layout(h_pad=-1.15, w_pad=-0.7)
plt.savefig('Rung1_metrics.png', bbox_inches='tight')

##%%Print for table
#for i in range(len(label)):
#    team, algorithm = label[i].split('-')
#    print team, '&' , algorithm, '&' , "{0:.3f} &  {1:.3f} &  {2:.3f} &  {3:.3f} \\\\".format(metric[i][0], metric[i][1], metric[i][2], metric[i][3])
