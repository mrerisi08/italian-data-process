import requests
import time


def send_ifttt_notification(message):
    url = f"https://maker.ifttt.com/trigger/python_notif_tester/with/key/cUlA4Bn82wLJshLLMLQwBt"
    data = {"value1": message}
    response = requests.post(url, json=data)

broken = False
last = None
while True:
    file = open("update_virtual.txt", 'r')
    file = file.readlines()
    if (time.time()-float(file[1])) > 15 and not broken:
        send_ifttt_notification(f"It broke at {file[0]}")
        broken = True

    if (time.time() - float(file[1])) < 15 and broken:
        broken = False
        send_ifttt_notification("Nvm fixed")
    this = file[0][:-1]
    if last != this:
        last = this
        print(this)
    time.sleep(1)

