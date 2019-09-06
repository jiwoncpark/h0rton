#### script 2019-8-26 16:57 CT by Joshua Yao-Yu Linear
import os
import requests
import lxml.html as lh
import pandas as pd


#### get a specific image
folder = "/media/joshua/HDD_fun2/time_delay_challenge/star_cleaned_galaxy/"
image_url = 'https://cgs.obs.carnegiescience.edu/CGS/data/images/NGC4094_clean_color.jpg'
img_data = requests.get(image_url).content


#### Create a folder for downloaded images
if not os.path.exists(folder):
    os.mkdir(folder)

#### save images
with open(folder + 'image_name.jpg', 'wb') as handler:
    handler.write(img_data)


#### table of galaxy homepage
content_url = 'https://cgs.obs.carnegiescience.edu/CGS/database_tables/sample0.html'


#### Create a handle, page, to handle the contents of the website
page = requests.get(content_url)#Store the contents of the website under doc
doc = lh.fromstring(page.content)#Parse data that are stored between <tr>..</tr> of HTML
tr_elements = doc.xpath('//tr')

#### check whether the link works
def exists(path):
    r = requests.head(path)
    return r.status_code == requests.codes.ok

col=[]
path_col = []
for i in range(1, len(tr_elements)):
    if i % 50 ==0:
        print("downloading...")
    for t in tr_elements[i][1]:
        name=t.text_content()
        col.append((name))
        name = name.replace(" ", "")
        name = name.replace("-G", "_")
        name = name.replace("-", "_")
        image_url = 'https://cgs.obs.carnegiescience.edu/CGS/data/images/' + name + '_clean_color.jpg'
        # https://cgs.obs.carnegiescience.edu/CGS/data/images/ESO556_015_clean_color.jpg
        if exists(image_url):
            path_col.append(name)
            img_data = requests.get(image_url).content
            with open(folder + name + '.jpg', 'wb') as handler:
                handler.write(img_data)




print("downloaded images:", path_col)
