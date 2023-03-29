import numpy as np
import random
import matplotlib.pyplot as plt
random.seed(0)

image_size = 32
mask = np.zeros((image_size, image_size, 3), dtype=bool)
pattern = np.zeros((image_size, image_size, 3), dtype=np.uint8)
trigger_size = 3
x_location = random.randint(0, image_size-trigger_size)
y_location = random.randint(0, image_size-trigger_size)


### generate 5x5 black_white trigger (random location) for cifar10
### generate 7x& black_white trigger (random location) for cifar10
# trigger_size = 7
# trigger_file_name = "cifar10_bottom_right_3by3_blackwhite.npy"
# pattern = np.load(trigger_file_name, allow_pickle=True).item().get("pattern")
# mask = np.load(trigger_file_name, allow_pickle=True).item().get("mask")
# pattern[29:, 29:, :] = pattern[0:3, 0:3, :]
# mask[29:, 29:, :] = mask[0:3, 0:3, :]
#
# x_location = random.randint(0, 32-trigger_size)
# y_location = random.randint(0, 32-trigger_size)
#
# for i in range(trigger_size):
#     for j in range(trigger_size):
#         if (i % 2 == 0) and (j % 2 == 0):
#             pattern[x_location + i, y_location + j, :] = 0
#         elif (i % 2 == 1) and (j % 2 == 1):
#             pattern[x_location + i, y_location + j, :] = 0
#         else:
#             pattern[x_location + i, y_location + j, :] = 255
#
# mask[x_location:(x_location + trigger_size), y_location:(y_location + trigger_size), :] = True
#
# trigger = {}
# trigger["pattern"] = pattern
# trigger["mask"] = mask
#
# np.save("cifar10_random_7by7_blackwhite.npy", trigger)

# ### generate 1x1 white trigger (random location) for cifar10
# trigger_file_name = "cifar10_bottom_right_3by3_blackwhite.npy"
# pattern = np.load(trigger_file_name, allow_pickle=True).item().get("pattern")
# mask = np.load(trigger_file_name, allow_pickle=True).item().get("mask")
# pattern[29:, 29:, :] = pattern[0:3, 0:3, :]
# mask[29:, 29:, :] = mask[0:3, 0:3, :]
#
# x_location = random.randint(0, 31)
# y_location = random.randint(0, 31)
# pattern[x_location, y_location, :] = 255
# mask[x_location, y_location, :] = True
#
# trigger = {}
# trigger["pattern"] = pattern
# trigger["mask"] = mask
#
# np.save("cifar10_random_1by1_white.npy", trigger)
#
# ### generate four corners trigger for tiny
# trigger_file_name = "tiny_bottom_right_3by3_blackwhite.npy"
# pattern = np.load(trigger_file_name, allow_pickle=True).item().get("pattern")
# mask = np.load(trigger_file_name, allow_pickle=True).item().get("mask")
#
# pattern[0:3, 0:3, :] = pattern[61:64, 61:64, :]
# pattern[0:3, 61:64, :] = pattern[61:64, 61:64, :]
# pattern[61:64, 0:3, :] = pattern[61:64, 61:64, :]
#
# mask[0:3, 0:3, :] = mask[61:64, 61:64, :]
# mask[0:3, 61:64, :] = mask[61:64, 61:64, :]
# mask[61:64, 0:3, :] = mask[61:64, 61:64, :]
#
# trigger = {}
# trigger["pattern"] = pattern
# trigger["mask"] = mask
#
# np.save("tiny_four_corners_3by3_blackwhite.npy", trigger)
#
# ### generate bottom right trigger for tiny
# trigger_file_name = "cifar10_bottom_right_3by3_blackwhite.npy"
# pattern_original = np.load(trigger_file_name, allow_pickle=True).item().get("pattern")
# mask_original = np.load(trigger_file_name, allow_pickle=True).item().get("mask")
#
# pattern = (np.zeros((64, 64, 3)) + 122).astype(np.uint8)
# mask = (pattern != 122)
# pattern[32:, 32:, :] = pattern_original
# mask[32:, 32:, :] = mask_original
#
# trigger = {}
# trigger["pattern"] = pattern
# trigger["mask"] = mask
#
# np.save("tiny_bottom_right_3by3_blackwhite.npy", trigger)
#
#
# ### generate four corners trigger for cifar10
# trigger_file_name = "cifar10_four_corners_3by3_blackwhite.npy"
# pattern = np.load(trigger_file_name, allow_pickle=True).item().get("pattern")
# mask = np.load(trigger_file_name, allow_pickle=True).item().get("mask")
#
# pattern[0:3, 0:3, :] = pattern[29:32, 29:32, :]
# pattern[0:3, 29:32, :] = pattern[29:32, 29:32, :]
# pattern[29:32, 0:3, :] = pattern[29:32, 29:32, :]
#
# mask[0:3, 0:3, :] = mask[29:32, 29:32, :]
# mask[0:3, 29:32, :] = mask[29:32, 29:32, :]
# mask[29:32, 0:3, :] = mask[29:32, 29:32, :]
#
# trigger = {}
# trigger["pattern"] = pattern
# trigger["mask"] = mask
#
# np.save("cifar10_four_corners_3by3_blackwhite.npy", trigger)
# print("end")