# coding=utf-8
import sys
reload(sys)
sys.setdefaultencoding('utf8')
#!/usr/local/bin/python
import os.path
import glob 
from xml.dom import *
import xml.dom.minidom
import csv
out=open('data.csv','w')
csv_write=csv.writer(out,dialect='excel')
csv_write.writerow(['BugID','Product','Component','Assignee','Summary','Create','Changed','Description'])
def extractData(xml_flie):
    DOMTree = xml.dom.minidom.parse(xml_file)
    Data = DOMTree.documentElement
    Bugs = Data.getElementsByTagName("bug")
    for Bug in Bugs:
        bug_id = Bug.getElementsByTagName('bug_id')[0]
        product = Bug.getElementsByTagName('product')[0]
        component = Bug.getElementsByTagName('component')[0]
        summary = Bug.getElementsByTagName('short_desc')[0]
        create = Bug.getElementsByTagName('creation_ts')[0]
        changed = Bug.getElementsByTagName('delta_ts')[0]
        assignee = Bug.getElementsByTagName('assigned_to')[0]
        bi=bug_id.childNodes[0].data
        p=product.childNodes[0].data
        c=component.childNodes[0].data
        s=summary.childNodes[0].data
        cr=create.childNodes[0].data
        ch=changed.childNodes[0].data
        a=assignee.childNodes[0].data
        if 'inbox' not in a:
            t=''
            des=Bug.getElementsByTagName("long_desc")
            if len(des)>0:
                des_0=des[0]
                text=des_0.getElementsByTagName('thetext')[0]
                if text.childNodes!=[]:
                    t+=text.childNodes[0].data
                    data=[bi,p,c,a,s,cr,ch,t]
                    csv_write.writerow(data)
for xml_file in glob.glob('*.xml'):
    extractData(xml_file)

out.close()




