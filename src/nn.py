from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout


def make_neural_network(df):
    df = df.describe().round(2)
    df = df.to_numpy()

    X = df[:,0:8]
    Y = df[:, 8]
    model = Sequential()
    model.add(Dense(12, input_shape=(8,), activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # compile the keras model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit the keras model on the dataset
    model.fit(X, Y, epochs=300, batch_size=64)
    # evaluate the keras model
    _, accuracy = model.evaluate(X, Y)
    print('Accuracy: %.2f' % (accuracy * 100))