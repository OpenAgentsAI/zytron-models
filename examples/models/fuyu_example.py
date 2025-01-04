from zytron_models.fuyu import Fuyu

fuyu = Fuyu()

# This is the default image, you can change it to any image you want
out = fuyu("What is this image?", "images/zytron.jpeg")
print(out)
