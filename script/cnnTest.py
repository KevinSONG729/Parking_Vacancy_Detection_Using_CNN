import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("C:\\Users\\qs\\Documents\\finalProj_ws\\notebooks\\mnist", one_hot = True)

n_classes = 10
batch_size = 128

x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding="SAME")

def maxpool2d(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")

def cnn(x):
    weights = {'W_conv1':tf.Variable(tf.random_normal([5, 5, 1, 32])),
               'W_conv2':tf.Variable(tf.random_normal([5, 5, 32, 64])),
               # [5, 5, 1, 32]: 5 by 5 kernal, input 1 feature, output 32 features
               # the next layer should have 32 input features and more output features
               # that's why W_conv2 is [5, 5, 32, 64]
               # 64 can be changed to higher number 
               'W_fc':tf.Variable(tf.random_normal([7*7*64, 1024])),
               # fc: fully connected layer
               # want 1024 nodes
               'out':tf.Variable(tf.random_normal([1024, n_classes])),}

    biases = {'b_conv1':tf.Variable(tf.random_normal([32])),
              'b_conv2':tf.Variable(tf.random_normal([64])),
              # [5, 5, 1, 32]: 5 by 5 kernal, input 1 feature, output 32 features
              # the next layer should have 32 input features and more output features
              # that's why W_conv2 is [5, 5, 32, 64]
              # 64 can be changed to higher number 
              'b_fc':tf.Variable(tf.random_normal([1024])),
              # fc: fully connected layer
              # want 1024 nodes
              'out':tf.Variable(tf.random_normal([n_classes])),}
    
    x = tf.reshape(x, shape=[-1, 28, 28, 1])
    
    
    conv1 =  maxpool2d(conv2d(x, weights['W_conv1']))

    conv2 = maxpool2d(conv2d(conv1, weights['W_conv2']))
 
    fc = tf.nn.relu(tf.matmul(tf.reshape(conv2, [-1, 7*7*64]), weights['W_fc']) + biases['b_fc'])
    
    output = tf.matmul(fc, weights['out']) + biases['out']
  
    
    return output


def train(x):
    prediction = cnn(x)
    # OLD VERSION:
    #cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(prediction,y) )
    # NEW:
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y) )
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    hm_epochs = 10
    with tf.Session() as sess:
        # OLD:
        #sess.run(tf.initialize_all_variables())
        # NEW:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c

            print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:',accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))

train(x)