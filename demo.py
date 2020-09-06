from Classification.Training.Model_Builder import ModelBuilder
from Localization.Localizer import Localizer
from World_Location_Detection.LocationCalculation import LocationCalculator

image_dir = 'Classification/Data/extra_imgs/1data39.png'
class_model = ModelBuilder(ckpt = 'Classification/Training/checkpoints/LL_NEWSA').get_model()
l = Localizer(class_model, detection_weights = 'Localization/faster_rcnn_resnet50_v1_640x640_coco17_tpu-8/saved_model')
calc = LocationCalculator(camera_const = 1.6, FOV = 120, obj_size = 7)

results = l.call(image_dir, display = True)
cds = []
for n, i in enumerate(results['predictions']):
	if i > 0.6:
		cds.append(results['coords'][n])
cds = [[c[1], c[3] - c[1], c[0], c[2] - c[0]] for c in cds]
lcs = calc.get_world_locations(cds)
print()
print()
print("World Locations")
print(lcs)