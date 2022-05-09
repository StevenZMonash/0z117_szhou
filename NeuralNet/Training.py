import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
from tensorflow import keras
from tensorflow.keras import Input, Model

def load_data():
    data = []
    for i in range(5, 99):
        with open(f'p_a1_given_q_xeff_qid_{i}.pkl', 'rb') as f:
            temp = pickle.load(f)
            temp.drop([97, 98], inplace=True)  # remove some bad data
            data.append(temp)
    df = pd.concat(data)
    df.reset_index(drop=True, inplace=True)
    a1 = []
    for i in range(len(df)):
        a1.append(df['a1'][i])
    a1 = np.array(a1)
    p = []
    for i in range(len(df)):
        p.append(df['p_a1'][i])
    p = np.array(p)
    temp_q = df['q'].values
    temp_xeff = df['xeff'].values
    # multiply and reshape
    q = []
    xeff = []
    for i in range(len(temp_q)):
        for j in range(len(a1[0])):
            q.append(temp_q[i])
            xeff.append(temp_xeff[i])
    q = np.array(q)
    xeff = np.array(xeff)
    q = q.reshape(len(p), len(p[0]))
    xeff = xeff.reshape(len(p), len(p[0]))
    return q, xeff, a1, p

def Scaling(a1,p):
    scaler_a1 = MinMaxScaler()
    scaler_a1.fit(a1)
    norm_a1 = scaler_a1.transform(a1)
    scaler_p = MinMaxScaler()
    scaler_p.fit(p)
    norm_p = scaler_p.transform(p)
    return scaler_a1,scaler_p,norm_a1, norm_p

def get_model():
    Input1 = Input(shape=(len(q[0]),))
    Input2 = Input(shape=(len(xeff[0]),))
    Input3 = Input(shape=(len(a1[0], )))
    merged = keras.layers.concatenate([Input1, Input2, Input3])  # 1500 elements
    x = keras.layers.Dense(2048, activation='relu')(merged)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dense(1024, activation='relu', kernel_regularizer='l1')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dense(500, activation='relu', kernel_regularizer='l1')(x)
    output = keras.layers.Dense(500)(x)
    #
    model = Model(inputs=[Input1, Input2, Input3], outputs=output)
    opt = keras.optimizers.Adam(learning_rate=1e-5)
    model.compile(optimizer=opt, loss='mse')
    return model

def evaluate(model):
    outdir = 'out'
    os.makedirs(outdir, exist_ok=True)
    scores = model.evaluate([test_q, test_xeff, test_a1],test_p)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    # plot history loss
    plt.figure(dpi=500, facecolor='white')
    plt.plot(np.log10(history.history['loss']))
    plt.plot(np.log10(history.history['val_loss']))
    plt.legend(['train', 'val'],bbox_to_anchor=(1.05, 1))
    plt.xlabel('epoches')
    plt.ylabel('log10 loss')
    plt.title('model loss')
    plt.savefig('out/train_val_loss.png')
     #
    predict = model.predict([test_q, test_xeff, test_a1])
    plt.figure(dpi=500,facecolor='white')
    for i in range(len(predict)):
        if i in np.random.randint(len(predict),size=10): # plot 10 random distribution
            plt.plot(scaler_a1.inverse_transform(test_a1[i]), scaler_p.inverse_transform(test_p[i]),
                     linestyle='dashed', alpha = 0.5)
            plt.plot(scaler_a1.inverse_transform(test_a1[i]), scaler_p.inverse_transform(predict[i]),
                     linestyle='dotted', alpha = 0.5)
            plt.title(f'prediction vs true p_a1 given q = {q[i,0]}, xeff={xeff[i,0]}')
            plt.legend(['actual','predicted'],bbox_to_anchor=(1.05, 1))
            plt.xlabel('a1')
            plt.ylabel('p')
            plt.savefig(f'out/q = {q[i,0]}, xeff={xeff[i,0]}.png')
def main():
    # preprocessing
    q,xeff,a1,p = load_data()
    scaler_a1, scaler_p, norm_a1,norm_p = Scaling(a1,p)
    train_q, test_q, train_xeff, test_xeff, train_p, test_p, train_a1, test_a1 \
        = train_test_split(q, xeff, norm_p, norm_a1,
                           test_size=0.2, random_state=42)
    # training
    # simple network, train overnight
    model = get_model(scaler_a1,scaler_p)
    history = model.fit([train_q, train_xeff, train_a1], train_p,
                        epochs=10, batch_size=32, verbose=2)
    model.save("my_model")
    evaluate(model)

if __name__ == "__main__":
    main()