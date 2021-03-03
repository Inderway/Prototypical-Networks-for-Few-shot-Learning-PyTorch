import numpy as np
s=slice(1,2)
a=[1,2,3,4,5,6]
print(a[s])


import os
for root, dirs, files in os.walk("D:\\download"):
    for name in files:
        print("files: "+root+'-----'+name)
    for name in dirs:
        print("dirs: " + root + '-----' + name)