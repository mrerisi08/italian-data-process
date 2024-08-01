import time

while True:
    file = open("iter.txt", 'r')
    reads = file.read()
    while reads == "":
        reads = file.read()
    print(reads)
    time.sleep(1)