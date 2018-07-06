import os
os.getcwd()
collection = "images/Hoodies"
for i, filename in enumerate(os.listdir(collection)):
    os.rename("images/Hoodies/" + filename, "images/Hoodies/" + str(1000000000+i) + ".jpg")