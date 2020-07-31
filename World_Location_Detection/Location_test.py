from LocationCalculation import LocationCalculator

lc = LocationCalculator(camera_const = 1.6, FOV = 120, obj_size = 0.5)
print(lc.get_world_locations([[297, 1985, 3228, 5]], [4032, 4444]))