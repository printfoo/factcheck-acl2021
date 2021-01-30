#!/usr/local/bin/python3

import requests
import os, sys, json, time

# snopes crawler
class snopesCrawler:
    
    # initialization
    def __init__(self, save_path, *args):
        self.index = 1
        self.save_path = save_path
        self.base = "https://www.snopes.com/fact-check/page/"
        if args:
            self.url_list = args[0]

    # save current response and increment to next
    def save_and_increment(self):
        with open(os.path.join(self.save_path, self.type + "-{:05}".format(self.index)), "w") as f:
            f.write(self.res.content.decode("utf-8"))
        self.index += 1

    # get a response
    def get(self):
        if self.index > len(self.url_list):
            exit()
        self.res = requests.get(self.url_list[self.index - 1])
        print(self.index, self.res.status_code)
        self.save_and_increment()


if __name__ == "__main__":
    save_path = "raw"
    list_path = os.path.join("raw", "url_list")
    with open(list_path, "r") as f:
        url_list = f.read().split("\n")[:-1]
    snopes = snopesCrawler(save_path, url_list)
    while True:
        snopes.get()
