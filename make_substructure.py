import os
import string
import random

def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))


path = "./"

for i in range(50):
    id = id_generator()
    os.system("mkdir " + path + "/" + id);
    os.system("touch " + path + "/" + id + ".txt");
    os.system("echo Not The Flag lol > " + path + "/" + id + ".txt");
    
    
    path += "/" + id;
