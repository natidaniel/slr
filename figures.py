import matplotlib.pyplot as plt
import numpy as np
import itertools

x = np.arange(2, 4)
p0 = [99.34, 33.75]
p1 = [99.79, 100]
p2 = [95.91, 98.50]
l0 = [20.07, 26.84]
l1 = [92.17, 95.14]
l2 = [91.00, 97.32]

font = {'family': 'serif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 16,
        }

colors = ['b', 'r']
marker =['g^', 'rd', 'bs']
datasets = ['OXF Unknown Trolley', 'WOB walking Unknown waist', 'PAR Unknown belt']
models = ['PCA', 'LDA']

fig, ax = plt.subplots()
ax.plot(x,p0, marker[0], label=models[0])
ax.plot(x,p1, marker[1], label=models[0])
ax.plot(x,p2, marker[2], label=models[0])
ax.plot(x,l0, marker[0], label=models[1])
ax.plot(x,l1, marker[1], label=models[1])
ax.plot(x,l2, marker[2], label=models[1])
leg1 = ax.legend(loc='lower left')
plt.title('Sensitivity analysis on multivariate statistics', fontdict=font)
plt.ylabel('Accuracy [%]', fontdict=font)
plt.xlabel('# principal components', fontdict=font)
plt.xlim(1.8, 3.2)
plt.legend(numpoints=1)
plt.show()





N = 3
ind = np.arange(N)  # the x locations for the groups
width = 0.1       # the width of the bars


p0 = [99.34, 33.75]
p1 = [99.79, 100]
p2 = [95.91, 98.50]
l0 = [20.07, 26.84]
l1 = [92.17, 95.14]
l2 = [91.00, 97.32]

fig, (ax1, ax2) = plt.subplots(2)
fig.suptitle('Sensitivity analysis on multivariate statistics', fontdict=font)
yvals = [99.79, 95.91, 99.34]
rects1 = ax1.bar(ind, yvals, width, color='r')
zvals = [100,98.50,33.75]
rects2 = ax1.bar(ind+width, zvals, width, color='b')

ax1.set_ylabel('Accuracy [%]', fontdict=font)
ax1.set_xticks(ind+width)

ax1.set_xticklabels( ('WOB walking Unknown waist', 'PAR Unknown belt', 'OXF Unknown Trolley'), fontdict=font )
ax1.legend( (rects1[0], rects2[0],), ('2 principal components', '3 principal components') )

def autolabel(rects):
    for rect in rects:
        h = rect.get_height()
        ax1.text(rect.get_x()+rect.get_width()/2., 1.05*h, '%d'%int(h),
                ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

plt.show()


import numpy as np
import matplotlib.pyplot as plt

N = 3
menMeans = (99.79, 95.91, 99.34)
menStd =   (2, 3, 4)

ind = np.arange(N)  # the x locations for the groups
width = 0.15       # the width of the bars

fig, (ax1, ax2) = plt.subplots(2)
fig.suptitle('Sensitivity analysis on multivariate statistics', fontdict=font)

rects1 = ax1.bar(ind, menMeans, width,
                    color='r',
                    yerr=menStd,
                    error_kw=dict(elinewidth=6, ecolor='pink'))

womenMeans = (100,98.50,33.75)
womenStd =   (3, 5, 2)
rects2 = ax1.bar(ind+width, womenMeans, width,
                    color='y',
                    yerr=womenStd,
                    error_kw=dict(elinewidth=6, ecolor='yellow'))

# add some
ax1.ylabel('Accuracy [%]')
ax1.xlabel('Datasets')
ax1.title('PCA')
ax1.xticks(ind+width/2, ('WOB walking Unknown waist', 'PAR Unknown belt', 'OXF Unknown Trolley') )
fig.legend( (rects1[0], rects2[0]), ('2 principal components', '3 principal components') ,loc=7)

def autolabel(rects):
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x()+rect.get_width()/2., 1.05*height, '%d'%int(height),
                ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

plt.show()

print("END")