from keras.callbacks import ModelCheckpoint
import engine

def train(weight = None, batch_size=32, epochs = 10):

    x = engine.generate()
    m = x.create_model()

    if weight != None:
        m.load_weights(weight)

    lst = [ModelCheckpoint('weights-improvement-{epoch:02d}.hdf5', verbose=1, save_best_only=True, monitor='loss', mode='min')]
    m.fit_generator(x.data_engine(batch_size=batch_size), verbose=2, callbacks=lst, steps_per_epoch=x.total_samples / batch_size, epochs=epochs)

    try:
        m.save('model.h5', overwrite=True)
        m.save_weights('weights.h5',overwrite=True)
    except:
        print("Issue in training")

if __name__ == '__main__':
    train(epochs=25)
    # training(epochs=50)