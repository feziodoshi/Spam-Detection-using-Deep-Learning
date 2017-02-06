import tensorflow as tf
import numpy as np
import os
import time
import tarfile
import matplotlib.pyplot as plt

# my_data = np.genfromtxt('new_spam.csv', delimiter=',')
# print(my_data.shape)


# def import_data():
# 	images=my_data[:,:-1]
# 	labels=my_data[:,-1:]
# 	trainX=images[:3800,:]
# 	trainY=labels[:3800,:]
# 	testX=images[3800:,:]
# 	testY=labels[3800:,:]
# 	return trainX,trainY,testX,testY
def csv_to_numpy_array(filePath, delimiter):
    return np.genfromtxt(filePath, delimiter=delimiter, dtype=None)

def import_data():
    if "data" not in os.listdir(os.getcwd()):
        # Untar directory of data if we haven't already
        tarObject = tarfile.open("data.tar.gz")
        tarObject.extractall()
        tarObject.close()
        print("Extracted tar to current directory")
    else:
        # we've already extracted the files
        pass

    print("loading training data")
    trainX = csv_to_numpy_array("data/trainX.csv", delimiter="\t")
    trainY = csv_to_numpy_array("data/trainY.csv", delimiter="\t")
    print("loading test data")
    testX = csv_to_numpy_array("data/testX.csv", delimiter="\t")
    testY = csv_to_numpy_array("data/testY.csv", delimiter="\t")
    return trainX,trainY,testX,testY

trainX,trainY,testX,testY = import_data()


				
##Hyperparameters
alpha=0.005
input_dim=trainX.shape[1]
hidden1_dim=512
hidden2_dim=256
hidden3_dim=128	
output_dim=trainY.shape[1]
total_epochs=10000
batch_size=100
std_dev=0.5
drop_train_keep=0.6
drop_test_keep=1.0



X=tf.placeholder(tf.float64,[None,input_dim])
Y_=tf.placeholder(tf.float64,[None,output_dim])
dropout_keep_prob=tf.placeholder(tf.float64)

print(type(trainX[0][3]))

##Variables
W={
	"input_hidden1":tf.Variable(tf.random_normal([input_dim,hidden1_dim],stddev=std_dev,dtype=tf.float64)),
	"hidden1_hidden2":tf.Variable(tf.random_normal([hidden1_dim,hidden2_dim],stddev=std_dev,dtype=tf.float64)),
	"hidden2_hidden3":tf.Variable(tf.random_normal([hidden2_dim,hidden3_dim],stddev=std_dev,dtype=tf.float64)),
	"hidden3_output":tf.Variable(tf.random_normal([hidden3_dim,output_dim],stddev=std_dev,dtype=tf.float64))
}

b={
	"hidden1":tf.Variable(tf.random_normal([hidden1_dim],stddev=std_dev,dtype=tf.float64)),
	"hidden2":tf.Variable(tf.random_normal([hidden2_dim],stddev=std_dev,dtype=tf.float64)),
	"hidden3":tf.Variable(tf.random_normal([hidden3_dim],stddev=std_dev,dtype=tf.float64)),
	"output":tf.Variable(tf.random_normal([output_dim],stddev=std_dev,dtype=tf.float64))
}

def multilayer(x,w,b,keep_prob):
	hidden1_output=tf.nn.sigmoid(tf.add(tf.matmul(x,w["input_hidden1"]),b["hidden1"]))
	hidden1_output=tf.nn.dropout(hidden1_output,keep_prob)
	hidden2_output=tf.nn.sigmoid(tf.add(tf.matmul(hidden1_output,w["hidden1_hidden2"]),b["hidden2"]))
	hidden2_output=tf.nn.dropout(hidden2_output,keep_prob)
	hidden3_output=tf.nn.sigmoid(tf.add(tf.matmul(hidden2_output,w["hidden2_hidden3"]),b["hidden3"]))
	output=tf.nn.softmax(tf.add(tf.matmul(hidden3_output,w["hidden3_output"]),b["output"]))
	return output


init=tf.global_variables_initializer()
Y=multilayer(X,W,b,dropout_keep_prob)
cross_entropy_cost=-tf.reduce_sum(Y_*tf.log(Y))
optimizer=tf.train.GradientDescentOptimizer(learning_rate=alpha)

train_step=optimizer.minimize(cross_entropy_cost)   ###################          Beauty of Tensorflow                 ################

corr=tf.equal(tf.argmax(Y,1),tf.argmax(Y_,1))
accuracy=tf.reduce_mean(tf.cast(corr,tf.float64))

print("Computational Graph is ready")


##### matplotlib
epoch_val=[]
accuracy_val=[]
cost_val=[]

'''plt.ion()
fig = plt.figure()
# Create two subplots on their own axes and give titles
ax1 = plt.subplot("211")
ax1.set_title("TRAINING ACCURACY", fontsize=18)
ax2 = plt.subplot("212")
ax2.set_title("TRAINING COST", fontsize=18)
plt.tight_layout()'''


##Start Session and initialize
print("Staring Session")
sess=tf.Session()
sess.run(init)

## Summary Writer
'''activation_summary=tf.histogram_summary("output",Y)
accuracy_summary=tf.scalar_summary("accuracy",accuracy)
cost_summary=tf.scalar_summary("cost",cross_entropy_cost)
all_summary=tf.merge_all_summaries()
writer=tf.train.SummaryWriter("summary_logs",sess.graph_def)'''


for i in range(total_epochs):
	feed={X:trainX,Y_:trainY,dropout_keep_prob:drop_train_keep}
	# print("________________________________________TRAINING__________________________________________________")
	sess.run(train_step,feed_dict=feed)
	if(i%10==0):
		train_acc,cost=sess.run([accuracy,cross_entropy_cost],feed_dict=feed)
		accuracy_val.append(train_acc)
		epoch_val.append(i)
		cost_val.append(cost)
		##writer.add_summary(summ,i)

		feed={X:testX,Y_:testY,dropout_keep_prob:drop_test_keep}	
		test_acc=sess.run(accuracy,feed_dict=feed)
		print(test_acc,cost)




