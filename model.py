from keras.models import Model
from keras.layers import Input, Add, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.utils import plot_model


def build_model(): # keras use a 'fake' tensor to pass through the 'layer' to build the computation graph
    input_img = Input(shape=(64,64,1)) # the fake tensor
    #encoding
    x1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_img)
    # __call__ a module class to keep returning a 'fake' tensor
    m1 = MaxPooling2D(pool_size=(2, 2), padding='same')(x1)

    x2 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(m1)
    m2 = MaxPooling2D(pool_size=(2, 2), padding='same')(x2)

    x3 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(m2)
    m3 = MaxPooling2D(pool_size=(2, 2), padding='same')(x3)

    x4 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(m3)
    encoded = MaxPooling2D(pool_size=(2, 2), padding='same')(x4)

    #decoding
    y0 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(encoded)
    u0 = UpSampling2D(size=(2, 2))(y0)
    x4 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(x4)
    w  = Add()([u0,x4])

    y1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(w)
    u1 = UpSampling2D(size=(2, 2))(y1)
    x3 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(x3)
    w  = Add()([u1,x3])

    y2 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(w)
    u2 = UpSampling2D(size=(2, 2))(y2)
    x2 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(x2)
    w  = Add()([u2,x2])

    y3 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(w)
    u3 = UpSampling2D(size=(2, 2))(y3)
    x1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(x1)
    w  = Add()([u3,x1])

    decoded = Conv2D(filters=1, kernel_size=(3, 3), activation='sigmoid',padding='same')(w)
    autoencoder = Model(input_img, decoded)

    return autoencoder

if __name__ == "__main__":
    autoencoder = build_model()
    plot_model(autoencoder, to_file='model.png', show_shapes=True, show_layer_names=True)