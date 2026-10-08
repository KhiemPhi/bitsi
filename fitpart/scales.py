"""Approximate real-world sizes (largest extent, metres) per category.

PartNet-Mobility / PartNet-E and ShapeNetPart are size-normalized, but gripper
feasibility needs metric units. Each object is rescaled so its largest extent
equals its category's typical size. These are rough priors (TO CONFIRM before
reporting GPR), and the paper should state the assumption.
"""
import numpy as np

CATEGORY_SIZE_M = {
    # PartNet-E (PartSLIP 45 categories)
    "Bottle": 0.25, "Box": 0.30, "Bucket": 0.30, "Camera": 0.12, "Cart": 0.90, "Chair": 0.90,
    "Clock": 0.30, "CoffeeMachine": 0.35, "Dishwasher": 0.85, "Dispenser": 0.20, "Display": 0.55,
    "Door": 2.00, "Eyeglasses": 0.15, "Faucet": 0.30, "FoldingChair": 0.90, "Globe": 0.40,
    "Kettle": 0.25, "Keyboard": 0.45, "KitchenPot": 0.30, "Knife": 0.25, "Lamp": 0.50,
    "Laptop": 0.35, "Lighter": 0.08, "Microwave": 0.50, "Mouse": 0.11, "Oven": 0.80, "Pen": 0.15,
    "Phone": 0.15, "Pliers": 0.20, "Printer": 0.45, "Refrigerator": 1.80, "Remote": 0.20,
    "Safe": 0.50, "Scissors": 0.20, "Stapler": 0.15, "StorageFurniture": 1.20, "Suitcase": 0.70,
    "Switch": 0.12, "Table": 1.20, "Toaster": 0.30, "Toilet": 0.70, "TrashCan": 0.60, "USB": 0.06,
    "WashingMachine": 0.85, "Window": 1.20,
    # ShapeNetPart
    "Airplane": None, "Bag": 0.40, "Cap": 0.25, "Car": None, "Earphone": 0.20, "Guitar": 1.00,
    "Motorbike": None, "Mug": 0.12, "Pistol": 0.25, "Rocket": None, "Skateboard": 0.80,
}

# part names that a gripper is meant to hold (GPR is computed on these GT parts)
GRASPABLE_PARTS = {"handle", "knob", "shaft", "neck", "grip"}


def metric_scale(points, category):
    """Metres per point-cloud unit for this object, or None if the category has no prior
    (objects not meant to be grasped, e.g. airplanes)."""
    size = CATEGORY_SIZE_M.get(category)
    if size is None:
        return None
    return size / float(np.ptp(points, axis=0).max())
