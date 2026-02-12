# MIO-TCD class names
CLASS_NAMES = [
    'articulated_truck', 'background', 'bicycle', 'bus', 'car', 
    'motorcycle', 'non-motorized_vehicle', 'pedestrian', 
    'pickup_truck', 'single_unit_truck', 'work_van'
]

# axle count triggers
TRUCK_CLASSES = [
    'articulated_truck', 'single_unit_truck', 'bus', 
    'pickup_truck', 'work_van'
]

# Base toll rates 
BASE_TOLL_RATES = {
    'articulated_truck': 200, 
    'background': 0, 
    'bicycle': 0, 
    'bus': 100, 
    'car': 50, 
    'motorcycle': 0, 
    'non-motorized_vehicle': 0, 
    'pedestrian': 0, 
    'pickup_truck': 70, 
    'single_unit_truck': 150, 
    'work_van': 80
}

# cost per axle
AXLE_RATE = 50