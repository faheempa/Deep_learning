# import libraries
import numpy as np
import matplotlib.pyplot as plt

# import dataset (comes with colab!)
data = np.loadtxt(open("mnist_train_small.csv", "rb"), delimiter=",")

# shape of the data matrix
print(data.shape)

# extract labels (number IDs) and remove from data
labels = data[:, 0]
data = data[:, 1:]

print(labels.shape)
print(data.shape)

def show_digit_pic(index, data, labels):
    n=0
    while n < len(index):
            
        for i in range(min(10, len(index)-n)):
            plt.subplot(2, 5, i+1)
            idx = n + i
            img = np.reshape(data[index[idx], :], (28, 28))
            plt.imshow(img, cmap="gray")
            plt.title("The number %i" % labels[index[idx]])
        plt.show()
        n += 10

# show few random images
random_indices = np.random.randint(0, data.shape[0], 10)
show_digit_pic(random_indices, data, labels)

def digit_in_fnn_vision(index, data, labels):
    n=0
    while n < len(index):
        fig,axs = plt.subplots(2, 4,figsize=(12,6))
        for ax in axs.flatten():
        # create the image
            if n==len(index):
                    break;
            ax.plot(data[index[n],:],'ko')
            ax.set_title('The number %i'%labels[index[n]])
            n+=1
        plt.show()

random_indices = np.random.randint(0, data.shape[0], 8)
digit_in_fnn_vision(random_indices, data, labels)

# let's see some example 7s
the7s = np.where(labels==7)[0]
show_digit_pic(the7s[:10], data, labels)

# how many 7's are there?
print(f"no of 7's: {len(the7s)}")

# let's see how they relate to each other by computing spatial correlations
C = np.corrcoef(data[the7s,:])

# and visualize
fig,ax = plt.subplots(1,3,figsize=(16,6))
ax[0].imshow(C,vmin=0,vmax=1)
ax[0].set_title("Correlation across all 7's")

# extract the unique correlations and show as a scatterplot
uniqueCs = np.triu(C,k=1).flatten()
ax[1].hist(uniqueCs[uniqueCs!=0],bins=100)
ax[1].set_title('All unique correlations')
ax[1].set_xlabel("Correlations of 7's")
ax[1].set_ylabel('Count')

# show all 7's together
aveAll7s = np.reshape( np.mean(data[the7s,:],axis=0) ,(28,28))
ax[2].imshow(aveAll7s,cmap='gray')
ax[2].set_title("All 7's averaged together")

plt.tight_layout()
plt.show()