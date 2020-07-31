import math

class LocationCalculator():
	#The Camera constant is approximately euqal to focal length / camera length
	#an obj_size of 1 will mean distances calculated will be relative to object size
	def __init__(self, camera_const, FOV, obj_size = 1):
		self.camera_const = camera_const
		self.FOV = FOV
		self.obj_size = obj_size

	def change_camera(self, camera_const, FOV):
		self.camera_const = camera_const
		self.fov = FOV

	def change_object(self, obj_size):
		self.obj_size = obj_size

	#sizes need to be scalar value representing the ratio of perceived pixel size to size of the image
	def get_distances(self, perc_sizes):
		distances = [0 for i in range(len(perc_sizes))]
		for n, size in enumerate(perc_sizes):
			distances[n] = (self.obj_size / size) * self.camera_const
		return(distances)

	def get_world_locations(self, img_positions, img_size):
		sizes = [img_positions[i][2] / img_size[0] for i in range(len(img_positions))]
		distances = self.get_distances(sizes)

		locations = []
		for obj in range(len(distances)):
			center_loc = img_positions[obj][0] + img_positions[obj][2] / 2
			dist_from_center = (center_loc - (img_size[0] / 2)) / img_size[0]
			angle = (self.FOV / 2) * dist_from_center
			r_angle = math.radians(angle)

			frwd_dist = math.cos(abs(r_angle)) * distances[obj]
			lateral_dist = math.sin(abs(r_angle)) * distances[obj]
			lateral_dist *= angle / abs(angle)
			locations.append([distances[obj], angle, frwd_dist, lateral_dist])

		return(locations)