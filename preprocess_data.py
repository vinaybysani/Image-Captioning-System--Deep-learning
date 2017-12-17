from vgg16 import VGG16
import numpy as np
import pickle as pickle
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image

counter = 0


def image_to_array(image_path):
    i = image.img_to_array(image.load_img(image_path, target_size=(224, 224)))
    i = np.expand_dims(i, axis=0)
    i = preprocess_input(i)
    return np.asarray(i)


def encoding(model, img):
    global counter
    counter += 1
    x = model.predict(image_to_array('../Flicker8k_Dataset/Flicker8k_Dataset/' + str(img)))
    return np.reshape(x, x.shape[1])


def vgg_model():
    return VGG16(weights='imagenet', include_top=True, input_shape=(224, 224, 3))


def preprocess_data(no_imgs=-1):
    training_text = open('../Flickr8k_text/flickr_8k_train_dataset.txt', 'w')
    training_text.write("image_id\tcaptions\n")

    testing_text = open('../Flickr8k_text/flickr_8k_test_dataset.txt', 'w')
    testing_text.write("image_id\tcaptions\n")

    train_images_path = open('../Flickr8k_text/Flickr_8k.trainImages.txt', 'r')
    train_images = train_images_path.read().strip().split('\n') if no_imgs == -1 else train_images_path.read().strip().split('\n')[:no_imgs]
    train_images_path.close()

    test_images_path = open('../Flickr8k_text/Flickr_8k.testImages.txt', 'r')
    test_images = test_images_path.read().strip().split('\n') if no_imgs == -1 else test_images_path.read().strip().split('\n')[:no_imgs]
    test_images_path.close()

    cap = open('../Flickr8k_text/Flickr8k.token.txt', 'r')
    all_cap = cap.read().strip().split('\n')
    x = {}
    for x in all_cap:
        x = x.split("\t")
        x[0] = x[0][:len(x[0]) - 2]
        try:
            x[x[0]].append(x[1])
        except:
            x[x[0]] = [x[1]]
    cap.close()



    '''
    encoded_images is a dictionary
    sample key:value pair is 
    2513260012_03d33305cf.jpg : [ 0.          0.          1.70414758 ...,  0.          0.          0.        ]
    image : VGG generated weights
    each image's weight vector is of length 4096
    '''
    images = {}
    model = vgg_model()

    training_count = 0
    for i in train_images:
        images[i] = encoding(model, i)
        for j in x[i]:
            caption = "<start> " + j + " <end>"
            training_text.write(i + "\t" + caption + "\n")
            training_text.flush()
            training_count += 1
    training_text.close()

    testing_count = 0
    for i in test_images:
        images[i] = encoding(model, i)
        for j in x[i]:
            caption = "<start> " + j + " <end>"
            testing_text.write(i + "\t" + caption + "\n")
            testing_text.flush()
            testing_count += 1
    testing_text.close()

    # write to pickle file
    with open("encoded_images.p", "wb") as pickle_f:
        pickle.dump(images, pickle_f)
    return [training_count, testing_count]


if __name__ == '__main__':
    training_count, testing_count = preprocess_data()
