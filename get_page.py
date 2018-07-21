# coding=utf-8
from urllib2 import urlopen
from bs4 import BeautifulSoup

html = urlopen('https://bugs.eclipse.org/bugs/buglist.cgi?chfieldfrom=2007-01-01&chfieldto=Now&classification=BIRT&query_format=advanced&resolution=FIXED')
bsObj = BeautifulSoup(html, "html.parser")

for link in bsObj.findAll("a"):
    if 'href' in link.attrs:
        print link.attrs['href']
