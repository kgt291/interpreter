# -*- coding: utf-8 -*-
# https://github.com/chuckgu/OAIS
import numpy as np
import random
import sys
import tensorflow as tf
import nltk 
import itertools

# 데이터 파일
path = 'nietzsche.txt'
text = open(path).read().lower() # 읽고 소문자로 변환

# 문장을 단어로 토큰화
texts = nltk.tokenize.RegexpTokenizer(r'\w+').tokenize(text) # 단어(토큰)별로 자름
words = list(set(texts)) # 중복 제거
print('total words:', len(words))

# 단어들을 인덱스화
word_indices = dict((c, i) for i, c in enumerate(words)) # 단어 -> 인덱스
indices_word = dict((i, c) for i, c in enumerate(words)) # 인덱스 -> 단어

# 단어 3개씩 건너뛰며 20개씩 자르기
maxlen = 20
step = 3
sentences = []
next_words = []

for i in range(0, len(texts) - maxlen, step):
    sentences.append(texts[i: i + maxlen])
    next_words.append(texts[i + maxlen])

n_word = len(words)

# x data-set, y data-set 준비
X_data = np.zeros((len(sentences), maxlen, n_word), dtype=np.bool)
Y_data = np.zeros((len(sentences), n_word), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, word in enumerate(sentence):
        X_data[i, t, word_indices[word]] = 1
    Y_data[i, word_indices[next_words[i]]] = 1

print ("pre-processing ready")


#####################################################


# Parameters
learning_rate = 0.01
training_iters = 1000000 # iteration
batch_size = 128 # 하나의 배치 크기
display_step = 20 # 20 스탭마다 과정 출력
n_hidden = 128 # 히든 레이어 수

# tensor 선언
x = tf.placeholder("float", [None, maxlen, n_word])
y = tf.placeholder("float", [None, n_word])

# weight 선언
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_word]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_word]))
}

# Recurrent Neural Network (LSTM)
with tf.variable_scope("model"):
    # lstm cell
    cell = tf.contrib.rnn.BasicLSTMCell(n_hidden)

    # lstm cell output
    outputs, states = tf.nn.dynamic_rnn(cell, x ,time_major = True, dtype=tf.float32)
    
    outputs = tf.unstack(tf.transpose(outputs, [1,0,2]))

    # Linear activation (output * weight + bias)
    pred = tf.matmul(outputs[-1], weights['out']) + biases['out']


# Cost and Optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

print ("Network ready")


#####################################################


def make_batches(size, batch_size):
    # batch를 불러올 때 쉽게 처리하기 위해 미리 batch때 불러올 start index들을 list로 만든다.
    # size : 전체 size
    # batch_size : 한 batch당 size
    nb_batch = int(np.floor(size/float(batch_size)))
    # batch를 몇번 하는지 계산

    # (start index, end index) 를 element로 갖는 list를 반환
    return [(i*batch_size, min(size, (i+1)*batch_size)) for i in range(0, nb_batch)]

def slice_X(X, start=None, stop=None):
    # 전달받은 X를 start ~ stop까지의 범위로 잘라서(slice) 그 잘라진 값(들)을 반환한다.
    # start가 len 특성을 갖는 경우(일반 scala가 아닌 경우 (예. 4:21)) start만큼 잘라서 반환한다.)
    # 예. X: [[0, 1, 2],  start : 1:2   return : [[1, 2], 
    #        [3, 4, 5]]                           [4, 5]]

    if type(X) == list:
        # X가 list형 일 경우
        if hasattr(start, '__len__'):
            # start가 len 특성을 갖고 있을 경우 (일반 scala값이 아닐 경우)
            # X의 각 element별로 start만큼(start가 len을 갖고 있으므로)을 잘라서 반환한다. 
            return [x[start]  for x in X]
        else:
            # start가 len 특성을 갖고 있지 않을 경우 (일반 scale값)
            # X의 각 element별로 start에서 stop까지 잘라서 반환한다.
            return [x[start:stop] for x in X]
    else:
        # X가 list가 아닐 경우
        if hasattr(start, '__len__'):
            # start가 len 특성을 갖고 있을 경우 (일반 scala값이 아닐 경우)
            # X를 start만큼(start가 len을 갖고 있으므로) 잘라서 반환한다.
            return X[start]
        else:
            # start가 len 특성을 갖고 있지 않을 경우 (일반 scale값)
            # X를 start에서 stop까지 잘라서 반환한다.
            return X[start:stop]

# 학습 데이터 iterator로 입력 준비
ins = [X_data,Y_data]

n_train = X_data.shape[0]
index_array = np.arange(n_train)
np.random.shuffle(index_array)

batches = make_batches(n_train, batch_size)
ins = [slice_X(ins,index_array[batch_start:batch_end]) for batch_start, batch_end in batches]

iterator = itertools.cycle((data for data in ins if data != []))

print ("datasets ready")


#####################################################3



# start session
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    step = 1

    # 학습
    while step * batch_size < training_iters:
        [batch_x,batch_y] = next(iterator)
        # back propagation
        _, acc, loss = sess.run([optimizer,accuracy,cost], feed_dict={x: batch_x, y: batch_y})

        # display_step마다 진행상황 출력
        if step % display_step == 0:
            print ("Iter " + str(step * batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1

        start_index = random.randint(0, len(texts) - maxlen - 1)

        # 50 step마다 테스트
        if step % 50 == 0:
            generated = ''
            sentence = []
            for i in range(start_index, start_index + maxlen):
                sentence.append(texts[i])
            for i in sentence:
                generated += i + ' '
            print('----- Generating with seed: "' + generated + '"')
            sys.stdout.write(generated)

            # 다음에 올 단어 1개 유추
            for i in range(1):
                x_sample_input = np.zeros((1, maxlen, n_word))
                for t, word in enumerate(sentence):
                    x_sample_input[0, t, word_indices[word]] = 1.

                preds = sess.run(pred, feed_dict={x: x_sample_input})
                next_index = np.argmax(preds)
                next_word = indices_word[next_index]

                generated += next_word + ' '
                del sentence[0]
                sentence.append(next_word)

                sys.stdout.write(next_word)
                sys.stdout.write(' ')
                sys.stdout.flush()
            print()

    print ("Optimization Finished!")