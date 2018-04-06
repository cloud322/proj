random.seed(111)
rng=pd.date_range(start='2000',periods=209 ,freq='M')
ts =pd.Series(np.random.uniform(-10,10,size=len(rng)),rng).cumsum()
ts.plot(c='b')

plt.show()

print(ts.head(10))

TS = np.array(ts)

num_periods = 20
f_horizon =1   #forcast horizon 1 period 360

x_data = TS[:(len(TS)-(len(TS) % num_periods))]
x_batches = x_data.reshape(-1,20,1)

y_data = TS[1:(len(TS)-(len(TS) % num_periods))+f_horizon]
y_batches = y_data.reshape(-1,20,1)

print(len(x_batches))
print(x_batches.shape)
print(x_batches[0:2])

print(y_batches[0:1])
print(y_batches.shape)