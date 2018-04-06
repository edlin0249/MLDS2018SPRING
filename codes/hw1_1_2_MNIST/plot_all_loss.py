import pandas as pd
import matplotlib.pyplot as plt

l1 = []

f1 = pd.read_csv('net_dnn1_loss_train.csv')

for i in f1:
	l1.append(float(i))


l1 = l1[:-1]
"""
l2 = []

f2 = pd.read_csv('net_dnn1_loss_test.csv')

for i in f2:
	l2.append(float(i))

l2 = l2[:-1]
"""
l2 = []

f2 = pd.read_csv('net_dnn2_loss_train.csv')

for i in f2:
	l2.append(float(i))


l2 = l2[:-1]
"""
l4 = []

f4 = pd.read_csv('net_dnn2_loss_test.csv')

for i in f4:
	l4.append(float(i))


l4 = l4[:-1]
"""
l3 = []

f3 = pd.read_csv('net_dnn3_loss_train.csv')

for i in f3:
	l3.append(float(i))


l3 = l3[:-1]
"""
l6 = []

f6 = pd.read_csv('net_dnn3_loss_test.csv')

for i in f6:
	l6.append(float(i))


l6 = l6[:-1]
"""
losses_his = [l1, l2, l3]

labels = ['shallow_dnn_train_loss', 'medium_dnn_train_loss', 'deep_dnn_train_loss']
for i, l_hist in enumerate(losses_his):
	plt.plot(l_hist, label=labels[i])
plt.legend(loc='best')
plt.xlabel('Epochs')
plt.ylabel('Loss')
#plt.ylim((0, 0.2))
plt.show()