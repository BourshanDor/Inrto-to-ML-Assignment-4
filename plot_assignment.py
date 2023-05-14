import backprop_data

import backprop_network
import matplotlib.pyplot as plt

training_data, test_data = backprop_data.load(train_size=10000, test_size=5000)

net = backprop_network.Network([784, 40, 10])

learning_rates = [0.001, 0.01, 0.1, 1, 10, 100]

x_axis = [i for i in range(30)]

training_accuracy  =[] 
training_loss      =[] 
test_accuracy      =[]

purposes = ['Test accuracy', 'Training accuracy', 'Training loss']

for learning_rate in learning_rates:

    y = net.SGD_for_plot(training_data, epochs=30, mini_batch_size=10,
                                learning_rate=learning_rate, test_data=test_data)
    a,b,c = [], [] , [] 
    for i in range(30) : 
        a.append(y[i][0])
        b.append(y[i][1])
        c.append(y[i][2])
    
    test_accuracy.append(a)
    training_accuracy.append(b)
    training_loss.append(c) 


j = 0 

y_axis = [test_accuracy, training_accuracy, training_loss]
for y in y_axis:
    i = 0

    for axis in y : 
        plt.plot(x_axis, axis, label=str(learning_rates[i]))
        i += 1 

    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title(purposes[j])
    plt.legend()
    path = 'C:\\Users\\bours\\Documents\\Dor\\BSc\\Intro to ML\\HW4\\Pictures\\' + purposes[j] + '.png'
    plt.savefig(path)
    plt.clf()
    j += 1 
