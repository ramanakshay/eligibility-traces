from dotmap import DotMap
from datetime import datetime
import os
import csv

class Logger(object):
    def __init__(self):
        self.logger = DotMap()
        self.info = ""
        self.data = []
        
    def update_data(self, log):
        for key, value in log.items():
            self.logger[key] = value
        self.data.append(self.logger.toDict())
            
    def update_info(self, info):
        self.info += (info + "\n")
            
    def save(self, log_name):
        log_name = datetime.now().strftime('%Y-%m-%d_%H:%M:%S_') + log_name
        # make directory
        path = "history/" + log_name
        os.mkdir(path)
        
        # write info
        f = open(path + "/info.txt", "w+")
        f.write(self.info)
        f.close()
        
        # write data
        fields = ['iteration', 'timestep', 'avg_ep_rew', 'avg_ep_lens', 'actor_loss']
        csvfile = open(path + "/data.csv", "w+")
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        writer.writerows(self.data)
        
    
    def print_summary(self):
        print(flush=True)
        print(f"-------------------- Iteration #{self.logger.iteration} --------------------", flush=True)
        print(f"Average Episodic Length: {round(self.logger.avg_ep_lens, 2)}", flush=True)
        print(f"Average Episodic Return: {round(self.logger.avg_ep_rew, 2)}", flush=True)
        print(f"Actor Loss: {round(self.logger.actor_loss, 5)}", flush=True)
        print(f"Timesteps So Far: {self.logger.timestep}", flush=True)
        print(f"------------------------------------------------------", flush=True)
        print(flush=True)