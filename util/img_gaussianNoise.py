import imageio
import imgaug as ia
import sys
from PIL import Image
from imgaug import augmenters as iaa

ia.seed(6)

def main():
	args = sys.argv[1:]
	if len(args) == 0:
		raise Exception('No arguments.')

	if args[0] == '1':
		if len(args) != 3 and len(args) != 4:
			raise Exception('Wrong numbers of variables.')
		elif len(args) == 3:
			func = iaa.AdditiveGaussianNoise(scale=(float(args[1]),float(args[2])))
		elif args[3] in ['True', 'true']:
			func = iaa.AdditiveGaussianNoise(scale=(float(args[1]),float(args[2])), per_channel=True)
		elif args[3] in ['False', 'false']:
			func = iaa.AdditiveGaussianNoise(scale=(float(args[1]),float(args[2])), per_channel=False)
		
	elif args[0] == '2':
		if len(args) != 2 and len(args) != 3:
			raise Exception('Wrong numbers of variables.')
		elif len(args) == 2:
			func = iaa.AdditivePoissonNoise(float(args[1]))
		elif args[2] in ['True', 'true']:
			func = iaa.AdditivePoissonNoise(float(args[1]), per_channel=True)
		elif args[2] in ['False', 'false']:
			func = iaa.AdditivePoissonNoise(float(args[1]), per_channel=False)
		
	elif args[0] == '3':
		if len(args) != 2 and len(args) != 3:
			raise Exception('Wrong numbers of variables.')
		elif len(args) == 2:
			func = iaa.SaltAndPepper(float(args[1]))
		elif args[2] in ['True', 'true']:
			func = iaa.SaltAndPepper(float(args[1]), per_channel=True)
		elif args[2] in ['False', 'false']:
			func = iaa.SaltAndPepper(float(args[1]), per_channel=False)
		
	elif args[0] == '4':
		if len(args) != 2:
			raise Exception('Wrong numbers of variables.')
		else:
			func = iaa.JpegCompression(float(args[1]))
		
	elif args[0] == '5':
		if len(args) != 2:
			raise Exception('Wrong numbers of variables.')
		else:
			func = iaa.GaussianBlur(float(args[1]))
		
	elif args[0] == '6':
		if len(args) != 5:
			raise Exception('Wrong numbers of variables. Should have 5')
		else:
			func = iaa.MotionBlur(int(args[1]), int(args[2]), float(args[3]), int(args[4]))
		
	elif args[0] == '7':
		if len(args) != 2 and len(args) != 3:
			raise Exception('Wrong numbers of variables.')
		elif len(args) == 2:
			func = iaa.AddToHueAndSaturation(int(args[1]))
		else:
			if args[2] in ['True', 'true']:
				indi = True
			elif args[2] in ['False', 'false']:
				indi = False
			func = iaa.AddToHueAndSaturation(int(args[1]), indi)
		
	elif args[0] == '8':
		if len(args) != 3 and len(args) != 4:
			raise Exception('Wrong numbers of variables.')
		elif len(args) == 3:
			func = iaa.SigmoidContrast(float(args[1]),float(args[2]))
		elif args[3] in ['True', 'true']:
			func = iaa.SigmoidContrast(float(args[1]),float(args[2]), per_channel=True)
		elif args[3] in ['False', 'false']:
			func = iaa.SigmoidContrast(float(args[1]),float(args[2]), per_channel=False)
		
	elif args[0] == '9':
		if len(args) != 5:
			raise Exception('Wrong numbers of variables. Should have 5')
		else:
			func = iaa.ElasticTransformation(int(args[1]), 
				int(args[2]), 
				int(args[3]), 
				int(args[4]))
		
	elif args[0] == '10':
		if len(args) != 4:
			raise Exception('Wrong numbers of variables. Should have 3')
		else:
			func = iaa.Superpixels(float(args[1]), int(args[2]), int(args[3]))

	elif args[0] == '11':
		if len(args) != 1:
			raise Exception('Wrong numbers of variables. Should have 3')
		else:
			func = iaa.Fog()
	return func

def the_noiser():
	noiser = main()
	x = 0
	while x < 5000:
		try:
			image = imageio.imread("..\pictures\Image" + str(x) + ".jpg")
			noised_image = noiser.augment_image(image)
			print(noised_image.shape)
			imageio.imwrite("../noised_images/" + "noised_image" + str(x) + ".jpg", noised_image)
		except:
			x = x + 1
			continue
		x = x + 1

if __name__ == "__main__":
	the_noiser()

