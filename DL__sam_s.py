
import sys
import tensorflow as tf
import pandas as pd
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import random
import tensorflow as tf
import tensorflow.contrib.learn as tflearn
import tensorflow.contrib.layers as tflayers
from tensorflow.contrib.learn import learn_runner
import tensorflow.contrib.metrics as metrics
import tensorflow.contrib.rnn as rnn

df_0403_s= pd.read_csv('data/0403_samsung_sec_U.csv',header=0)
                # names=['date','a','stock','time','price','pr_ch','vol_ch','vol'])
print(df_0403_s.head())
sam_0403_s = df_0403_s.query('stock=="삼성전자"')
sort_0403_s=sam_0403_s.sort_values(by=['date','price'])
partcol_0403_s =sam_0403_s[['date','price']]
# query
partcol_0403_s_09 = partcol_0403_s.query('date>="2018-04-03 09:00:00" & date<="2018-04-03 15:21:00"')
print(partcol_0403_s_09.head())
#plot
partcol_0403_s_09.plot(x='date',y='price',title = 'samsung')
plt.show()


print(len(partcol_0403_s_09))
ts =pd.Series(partcol_0403_s_09['price'].values, index=partcol_0403_s_09['date'])
print(ts)

TS=np.asarray(ts)

num_periods = 200
f_horizon =1   #forcast horizon 1 period 360

x_data = TS[:(len(TS)-(len(TS) % num_periods))]
x_batches = x_data.reshape(-1,200,1)

y_data = TS[1:(len(TS)-(len(TS) % num_periods))+f_horizon]
y_batches = y_data.reshape(-1,200,1)

print(len(x_batches))
print(x_batches.shape)
print(x_batches[0:2])

print(y_batches[0:1])
print(y_batches.shape)


#pull out test data

def test_data(series,forcast,num_periods):
    test_x_setup = TS[-(num_periods+forcast):]
    test_x = test_x_setup[:num_periods].reshape(-1,200,1)
    test_y = TS[-(num_periods):].reshape(-1,200,1)
    return test_x,test_y

x_test, y_test = test_data(TS,f_horizon,num_periods)
print(x_test)
print(x_test.reshape)

um_periods = 200
######################
hidden = 100
######################
input = 1
output = 1

x=tf.placeholder(tf.float32,[None, num_periods, input])
y=tf.placeholder(tf.float32,[None, num_periods, output])

#create rnn object
basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=hidden, activation=tf.nn.relu)
#Rectified Linear Unit - ReLU
# can be changed to Sigmoid or Hyberbolic Tangent (Tanh)
rnn_output, states=tf.nn.dynamic_rnn(basic_cell,x, dtype=tf.float32)

######################################
learning_rate=0.01  #learning rate low
######################################

stacked_rnn_output = tf.reshape(rnn_output, [-1,hidden])
stacked_outputs = tf.layers.dense(stacked_rnn_output,output)
outputs = tf.reshape(stacked_outputs, [-1,num_periods,output])

loss = tf.reduce_sum(tf.square(outputs-y))
optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op=optimizer.minimize(loss)

init = tf.global_variables_initializer()

##########################################
epochs = 500        #num of traing cycles
##########################################

with tf.Session() as sess:
    init.run()
    for ep in range(epochs):
        sess.run(training_op,feed_dict={x:x_batches,y:y_batches})
        if ep%100==0:
            mse = loss.eval(feed_dict={x:x_batches,y:y_batches})
            print(ep,'\tMSE:',mse)
    y_pred=sess.run(outputs,feed_dict={x: x_test})
    print(y_pred)

tf.reset_default_graph()
plt.title('Forcast vs Actual', fontsize= 26)
plt.plot(pd.Series(np.ravel(y_test)),'bo',markersize=3,label='actual')
plt.plot(pd.Series(np.ravel(y_pred)), "r.", markersize=3, label="Forecast")
plt.legend(loc='upper left')
plt.xlabel('time')

import sys
import tensorflow as tf
import pandas as pd
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import random
import tensorflow as tf
import tensorflow.contrib.learn as tflearn
import tensorflow.contrib.layers as tflayers
from tensorflow.contrib.learn import learn_runner
import tensorflow.contrib.metrics as metrics
import tensorflow.contrib.rnn as rnn

df_0403_s= pd.read_csv('data/0403_samsung_sec_U.csv',header=0)
                # names=['date','a','stock','time','price','pr_ch','vol_ch','vol'])
print(df_0403_s.head())
sam_0403_s = df_0403_s.query('stock=="삼성전자"')
sort_0403_s=sam_0403_s.sort_values(by=['date','price'])
partcol_0403_s =sam_0403_s[['date','price']]
# query
partcol_0403_s_09 = partcol_0403_s.query('date>="2018-04-03 09:00:00" & date<="2018-04-03 15:21:00"')
print(partcol_0403_s_09.head())
#plot
partcol_0403_s_09.plot(x='date',y='price',title = 'samsung')
plt.show()


print(len(partcol_0403_s_09))
ts =pd.Series(partcol_0403_s_09['price'].values, index=partcol_0403_s_09['date'])
print(ts)

TS=np.asarray(ts)

num_periods = 200
f_horizon =1   #forcast horizon 1 period 360

x_data = TS[:(len(TS)-(len(TS) % num_periods))]
x_batches = x_data.reshape(-1,200,1)

y_data = TS[1:(len(TS)-(len(TS) % num_periods))+f_horizon]
y_batches = y_data.reshape(-1,200,1)

print(len(x_batches))
print(x_batches.shape)
print(x_batches[0:2])

print(y_batches[0:1])
print(y_batches.shape)


#pull out test data

def test_data(series,forcast,num_periods):
    test_x_setup = TS[-(num_periods+forcast):]
    test_x = test_x_setup[:num_periods].reshape(-1,200,1)
    test_y = TS[-(num_periods):].reshape(-1,200,1)
    return test_x,test_y

x_test, y_test = test_data(TS,f_horizon,num_periods)
print(x_test)
print(x_test.reshape)

um_periods = 200
######################
hidden = 100
######################
input = 1
output = 1

x=tf.placeholder(tf.float32,[None, num_periods, input])
y=tf.placeholder(tf.float32,[None, num_periods, output])

#create rnn object
basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=hidden, activation=tf.nn.relu)
#Rectified Linear Unit - ReLU
# can be changed to Sigmoid or Hyberbolic Tangent (Tanh)
rnn_output, states=tf.nn.dynamic_rnn(basic_cell,x, dtype=tf.float32)

######################################
learning_rate=0.01  #learning rate low
######################################

stacked_rnn_output = tf.reshape(rnn_output, [-1,hidden])
stacked_outputs = tf.layers.dense(stacked_rnn_output,output)
outputs = tf.reshape(stacked_outputs, [-1,num_periods,output])

loss = tf.reduce_sum(tf.square(outputs-y))
optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op=optimizer.minimize(loss)

init = tf.global_variables_initializer()

##########################################
epochs = 500        #num of traing cycles
##########################################

with tf.Session() as sess:
    init.run()
    for ep in range(epochs):
        sess.run(training_op,feed_dict={x:x_batches,y:y_batches})
        if ep%100==0:
            mse = loss.eval(feed_dict={x:x_batches,y:y_batches})
            print(ep,'\tMSE:',mse)
    y_pred=sess.run(outputs,feed_dict={x: x_test})
    print(y_pred)

tf.reset_default_graph()
plt.title('Forcast vs Actual', fontsize= 26)
plt.plot(pd.Series(np.ravel(y_test)),'bo',markersize=3,label='actual')
plt.plot(pd.Series(np.ravel(y_pred)), "r.", markersize=3, label="Forecast")
plt.legend(loc='upper left')
plt.xlabel('time')
plt.show()