import pickle
pickle_in = open("encoded_images.p","rb")
p_dict = pickle.load(pickle_in)
for i,j in p_dict.items():
    print(i,j)
    print(len(j))
    exit()
