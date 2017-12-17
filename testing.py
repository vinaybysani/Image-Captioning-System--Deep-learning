from keras.preprocessing import sequence
import nltk
import pickle as pickle
import engine
import numpy as np

e = engine.generate()



def return_cap(cap):
    cap.sort(key=lambda l: l[1])
    ret_val = []
    for i in cap:
        ret_val.append([" ".join([e.index_word[index] for index in i[0]]), i[1]])
    return ret_val

def cap_end(caption):
    cap = caption.split()[1:]
    try:
        end = cap.index('<end>')
        cap = cap[:end]
    except:
        pass

    return " ".join([i for i in cap])


def caption_engine(model, image, beam):
    all_cap = [[[e.word_index['<start>']], 0.0]]
    while (len(all_cap[0][0]) < e.max_cap_len):
        temp_captions = []
        for c in all_cap:
            next = model.predict([np.asarray([image]), np.asarray(sequence.pad_sequences([c[0]], maxlen=e.max_cap_len, padding='post'))])[0]
            next_words = np.argsort(next)[-beam:]
            for i in next_words:
                new, new_partial_caption_prob = c[0][:], c[1]
                new.append(i)
                new_partial_caption_prob += next[i]
                temp_captions.append([new, new_partial_caption_prob])
        all_cap = temp_captions
        all_cap = all_cap.sort(key=lambda l: l[1])[-beam:]

    return all_cap


def BLEU(h, r):
    return nltk.translate.bleu_score.corpus_bleu(r, h)

def get_good_cap(captions):
    captions.sort(key=lambda l: l[1])
    return " ".join([e.index_word[index] for index in captions[-1][0]])

def test_model(w, img, beam_size=3):
    encoded_images = pickle.load(open("encoded_images.p", "rb"))
    image = encoded_images[img]

    model = e.create_model(ret_model=True)
    model.load_weights(w)

    return cap_end(get_good_cap(caption_engine(model, image, beam_size)))


def testing(weight, img_dir, beam_size=3):
    captions = {}
    with open(img_dir, 'r') as images_path:
        images = images_path.read().strip().split('\n')
    encoded_images = pickle.load(open("encoded_images.p", "rb"))
    model = e.create_model(ret_model=True)
    model.load_weights(weight)

    prediction = open('predicted_captions.txt', 'w')

    for count, image in enumerate(images):
        image = encoded_images[image]
        best_caption = cap_end(get_good_cap(caption_engine(model, image, beam_size)))
        captions[image] = best_caption
        prediction.write(image + "\t" + str(best_caption))
        prediction.flush()
    prediction.close()

    captions_path = open('Flickr8k_text/Flickr8k.token.txt', 'r')
    captions_text = captions_path.read().strip().split('\n')
    cap_pair = {}
    for i in captions_text:
        i = i.split("\t")
        i[0] = i[0][:len(i[0]) - 2]
        try:
            cap_pair[i[0]].append(i[1])
        except:
            cap_pair[i[0]] = [i[1]]
    captions_path.close()

    h = []
    r = []
    for image in images:
        h.append(captions[image])
        r.append(cap_pair[image])

    return BLEU(h, r)


if __name__ == '__main__':
    weight = 'weights-improvement-08.hdf5'
    test_image = '10815824_2997e03d76.jpg'
    test_img_dir = 'Flickr8k_text/Flickr_8k.testImages.txt'
    print(testing(weight, test_img_dir, beam_size=3))
