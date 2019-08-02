import requests
import pandas
import numpy
import urllib
def download_images(file_name):
	x = 0;
	image_url = pandas.read_csv(file_name)
	image_url_as_a_list = image_url.values.tolist()
	while x < 10000 :
		the_url = image_url_as_a_list[x]
		r = requests.get(the_url[0])
		with open("../pictures/Image" + str(x) + ".jpg" ,'wb') as f:
			f.write(r.content)
		x = x + 1
if __name__ == '__main__':
	download_images('open-images-dataset-train0.tsv')