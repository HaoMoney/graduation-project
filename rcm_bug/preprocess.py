import pandas as pd
import datetime as dt
import random
f=open('./datasets/OpenOffice.csv','r')
f3=open('./datasets/cleared_openoffice','w+')
data=f.readlines()
tmp=data[1:]
random.shuffle(tmp)
f3.write('\t'.join(data[0].split(',')))
#filter the invalid field in 'Assignee'
for line in tmp:
    line=line.strip()
    line=line.split(',')
    if len(line)==8:
        if "inbox" not in line[3]:
            for i in range(len(line)):
                line[i]=line[i].strip('"')
            f3.write('\t'.join(line))
            f3.write('\n')
f.close()
f3.close()
