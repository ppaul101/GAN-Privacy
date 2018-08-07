# =============================================================================
# https://arxiv.org/pdf/1712.07008.pdf
# section 3.3
# =============================================================================


import numpy as np # for array computations
import matplotlib.pyplot as plt # for graphical plots

import keras
from keras.layers import Input
from keras.models import Model, Sequential
from keras.layers.core import Dense
from keras.datasets import mnist
from keras.optimizers import Adam

import keras.backend as K

batch = 100
nb_epochs = 50
learning_rate = 0.0002
noise_dim = 20
lambda_val = 35

def adversary_loss(y_true, y_pred):    
    
    # K mean in range of 0 to 1
    # taking log would give negative answer :(
    # negative loss function => +ve feedback

    return K.mean(K.abs(0.1 - y_pred))

delta = 0.1

def my_loss(y_true, y_pred):
    
    bce = K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)
    
    bce = K.square(K.maximum(0., bce - delta))
    
    return lambda_val*bce

# =============================================================================
# not using 'scaled_distortion' fn
# =============================================================================

def scaled_distortion(gen_img, image_batch):
    
    # upper bound excluded
    
    sum = 0

    for i in range (100): # for the 100 images in a batch

        for j in range (784): 
        
            sum = sum + gen_img[i][j]*image_batch[i][j] + (1 - gen_img[i][j])*(1 - image_batch[i][j])
        
    distortion = sum / 784
    
    distortion = distortion * -1
    
    return lambda_val * distortion
    

# =============================================================================
# code stuck when calling scaled_distortion
# =============================================================================

loss_ppan = [scaled_distortion, adversary_loss]


def load_mnist_data():
    
    # train:test
    # 60,000:10,000
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # normalise training set (x_test)
    # [0,1]
    # also need to convert to a float

    x_train = x_train.astype(np.float32)/255
    
    # flatten
    x_train = x_train.reshape(60000, 784)
    
    return (x_train, y_train, x_test, y_test)



def get_mechanism():
    
    # Mechanism pitted against Adversary
    
    # object of Sequential class
    # in order to build the NN
    mechanism = Sequential()
    
    # random noise generated and added durin each training epoch
    s = 784 + noise_dim
    mechanism.add(Dense(1000, activation = 'tanh', input_dim = (s)))
    
    mechanism.add(Dense(1000, activation = 'tanh'))
    
    mechanism.add(Dense(784, activation = 'sigmoid'))
    
    mechanism.compile(loss = 'mse', optimizer = Adam(lr = learning_rate))

    
    return mechanism



def get_adversary():
    
    # Adversary produces distribution over the 10 digit labels (ideally 0.1)  
    
    adversary = Sequential()
    
    adversary.add(Dense(1000, activation = 'tanh', input_dim = 784))
    
    adversary.add(Dense(1000, activation = 'tanh'))
    
    adversary.add(Dense(10, activation = 'softmax'))
    
    
    adversary.compile(loss = 'categorical_crossentropy', optimizer = Adam(lr = learning_rate), metrics = ['accuracy'])
    
    return adversary


   


# connecting the mechanism and adversary

def get_ppan_network (random_dim, mechanism, adversary):
    

    # only want to train either mechanism or adversary, one at a time
    adversary.trainable = False

 

    # Input used to initiate Keras tensor
    s = 784 + random_dim
    ppan_input = Input(shape = (s,)) 
    
    
    # x = O/P of mechanism (mechanism produced from get_mechanism fn above)
    gen_img = mechanism(ppan_input)

    # ppan_output = O/P of adversary
    # probability that the digit is 0, 1, 2, ...
    # ideally 10% (equilibrium)
    # adversary produced from get_adversary fn above
    # the build_adversary accepts O/P of mechanism as I/P
    pred_label = adversary(gen_img)
    
    # add a pred_valid when using the discriminator 
    # generated image

    # need O/P for each loss fn
    # at the end of each train_on_batch, model should compute its loss func
    ppan = Model(inputs = ppan_input, outputs = [gen_img, pred_label])
       
    ppan.compile(loss = [my_loss, adversary_loss], optimizer = Adam(lr = learning_rate), metrics = ['accuracy'])
    
    ppan.summary()
    
    return ppan
    


# to visualise the O/P of mechanism
    
def plot_generated_images(epoch, mechanism, adversary,examples = 100, dim = (10, 10), figsize = (10,10)):
    
    examples = 100
    
    
    x_train, y_train, x_test, y_test = load_mnist_data()
    
    
    plt.figure(figsize = figsize)
    
    
    for i in range(100):
        
        noise = np.random.uniform(low = -1, high = 1, size = [examples, noise_dim])
        
        rand_num = np.random.randint(0, x_train.shape[0], size = batch)
        image_batch = x_train[rand_num]
        
        stack = np.hstack([image_batch, noise])

        noisy_images = mechanism.predict(stack)
        
#        print ('predicting image label')
        
#        pred = adversary.predict(noisy_images)
    
#        print(pred)
    
        noisy_images = noisy_images.reshape(examples, 28, 28)
        
        noisy_images = np.clip(noisy_images, 0, 1)
           
        dim = (10,10)
        plt.subplot(dim[0], dim[1], i+1)
        
        plt.axis('off')
    
        plt.imshow(noisy_images[i], interpolation = 'nearest', cmap = 'gray_r')
        

        

    plt.tight_layout()
    
    plt.savefig('generated_%d.png' % epoch)

    

# training the network
    
def train(epochs = nb_epochs, batch_size = batch):
    
    # for quick run
    batch_size = 100
    epochs = 50
    
    # using the previously defined fn to preprocess data
    x_train, y_train, x_test, y_test = load_mnist_data()
    
    # y_train has 1 hot encoding corresponding to the x_train digits
    y_train = keras.utils.to_categorical(y_train, 10)

    
    # split training data into batches of required size
    # 60000/100 = 600 (total no. of batches)
    batch_num = x_train.shape[0]//batch
     
    mechanism = get_mechanism()
    
    adversary = get_adversary()

    # train the adversary (multiclass)
    adv_loss = adversary.train_on_batch(x_train, y_train)
    
    ppan = get_ppan_network(noise_dim, mechanism, adversary)
    
    
    
    for e in range(1, epochs+1): # upper bound excluded
        
    
        print ('-' * 15, 'Epoch %d' % e, '-'*15)
        
        
        
        for _ in range(batch_num):
            
            rand_num = np.random.randint(0, x_train.shape[0], size = batch_size)
            
            image_batch = x_train[rand_num]
            
            noise = np.random.uniform(low = -1, high = 1, size = [batch, noise_dim])
            
             # hstack stacks arrays columnwise (will be 100 x 804, which is the desired I/P)
            gen_img = mechanism.predict(np.hstack([image_batch, noise]))
            
            gen_img = np.clip(gen_img, 0, 1)
           
            # want adversary to think the generated images (with noise) correspond to the predicted labels
            
            adv_loss = adversary.train_on_batch(gen_img, y_train[rand_num])
            print ('adv loss is:')
            print (adv_loss)
    
            ppan_loss = ppan.train_on_batch(np.hstack([image_batch, noise]), [gen_img, y_train[rand_num]])
            print ('ppan loss is:')
            print (ppan_loss)
        
    
        if (e == 1 or e%2 == 0):
            
           plot_generated_images(e, mechanism, adversary)

           
    
    
    
    
if __name__ == '__main__':
    
    train(nb_epochs, batch)
    
    
    

