"""Synthetic user commands -> task-screw slots, with held-out phrasing and held-out object names.

Slots: task (13 screws), direction (+1 / -1: open/close, loosen/tighten, pull/push, squeeze/release;
+1 for one-directional tasks), part (the part to grasp/push), and an optional magnitude (degrees or
centimetres, parsed by rule in lang/model.py, not learned).
Each task has TRAIN and TEST templates (different wording) and TRAIN / TEST object nouns, so the test
split measures generalisation to new phrasing and new objects, not memorisation.
"""
import random

TASKS = ["pour", "lift", "pivot", "turn", "hinge", "slide", "press", "swing", "scoop", "stir", "cut", "squeeze",
         "handover"]

# task -> (object nouns train, object nouns test, grasp part)
OBJECTS = {
    "pour": (["mug", "cup", "kettle", "pot", "jug"], ["teapot", "measuring cup", "pitcher"], "handle"),
    "lift": (["drill", "bucket", "bag", "kettle", "toolbox"], ["suitcase", "basket", "power drill"], "handle"),
    "pivot": (["box", "carton", "book"], ["crate", "cereal box"], "body"),
    "turn": (["bottle", "jar", "dispenser", "flask"], ["water bottle", "pill bottle", "thermos"], "cap"),
    "hinge": (["box", "trash can", "laptop", "toilet", "chest"], ["cooler", "jewelry box", "microwave"], "lid"),
    "slide": (["drawer", "cabinet", "dresser", "nightstand"], ["filing cabinet", "desk drawer", "tray"], "handle"),
    "press": (["button", "switch", "remote", "doorbell", "keyboard"], ["light switch", "elevator button", "pump"], "button"),
    "swing": (["hammer", "mallet", "axe"], ["hatchet", "sledgehammer"], "handle"),
    "scoop": (["ladle", "spoon", "scoop", "shovel"], ["trowel", "dipper"], "handle"),
    "stir": (["whisk", "spoon", "spatula", "stirrer"], ["wooden spoon", "stick"], "handle"),
    "cut": (["knife", "saw", "cleaver"], ["bread knife", "box cutter"], "handle"),
    "squeeze": (["tongs", "pliers", "clamp", "stapler"], ["clothespin", "nutcracker"], "arm"),
    "handover": (["screwdriver", "knife", "scissors", "wrench"], ["chisel", "file", "pen"], "body"),
}

# task -> list of (template, direction); {o} = object, {m} = optional magnitude phrase
TRAIN = {
    "pour": [("pour out the {o}", 1), ("empty the {o}{m}", 1), ("tip the {o} over and pour", 1),
             ("pour the water from the {o}", 1), ("dump what is in the {o}", 1)],
    "lift": [("pick up the {o}", 1), ("lift the {o}{m}", 1), ("raise the {o}", 1), ("grab the {o} and lift it", 1)],
    "pivot": [("tilt the {o}{m}", 1), ("tip the {o} onto its side", 1), ("lean the {o} back", 1)],
    "turn": [("open the {o}", 1), ("unscrew the cap of the {o}", 1), ("loosen the {o} cap{m}", 1),
             ("close the {o}", -1), ("tighten the cap on the {o}", -1), ("screw the {o} shut", -1)],
    "hinge": [("open the {o}", 1), ("lift the lid of the {o}", 1), ("flip open the {o}{m}", 1),
              ("close the {o}", -1), ("shut the lid of the {o}", -1), ("put the {o} lid down", -1)],
    "slide": [("open the {o}", 1), ("pull out the {o}{m}", 1), ("slide the {o} open", 1),
              ("close the {o}", -1), ("push the {o} in", -1), ("slide the {o} shut", -1)],
    "press": [("press the {o}", 1), ("push the {o}", 1), ("hit the {o}", 1), ("click the {o}", 1)],
    "swing": [("swing the {o}", 1), ("hammer with the {o}", 1), ("strike down with the {o}{m}", 1)],
    "scoop": [("scoop with the {o}", 1), ("ladle some soup with the {o}", 1), ("dig in with the {o}{m}", 1)],
    "stir": [("stir with the {o}", 1), ("mix the batter using the {o}", 1), ("whisk with the {o}{m}", 1)],
    "cut": [("cut with the {o}", 1), ("slice the bread with the {o}", 1), ("saw through it using the {o}{m}", 1)],
    "squeeze": [("squeeze the {o}", 1), ("close the {o}", 1), ("pinch the {o} together{m}", 1),
                ("release the {o}", -1), ("open the {o}", -1), ("let go of the {o} slowly", -1)],
    "handover": [("hand me the {o}", 1), ("give me the {o}", 1), ("pass the {o} to me", 1)],
}
TEST = {
    "pour": [("could you pour whatever is in the {o}", 1), ("drain the {o}{m}", 1), ("empty out that {o} please", 1)],
    "lift": [("hold the {o} up in the air", 1), ("can you get the {o} off the table{m}", 1)],
    "pivot": [("rock the {o} back on its edge{m}", 1), ("tilt that {o} away from you", 1)],
    "turn": [("twist off the lid of the {o}", 1), ("crack open the {o}{m}", 1), ("put the cap back on the {o}", -1),
             ("seal the {o} up again", -1)],
    "hinge": [("pop the {o} open", 1), ("raise the {o} cover{m}", 1), ("close up the {o} please", -1),
              ("fold the {o} closed", -1)],
    "slide": [("draw out the {o}{m}", 1), ("yank the {o} open", 1), ("tuck the {o} back in", -1),
              ("close that {o} for me", -1)],
    "press": [("tap the {o}", 1), ("push down on the {o} once", 1), ("go ahead and press that {o}", 1)],
    "swing": [("bang on it with the {o}", 1), ("give it a whack with the {o}{m}", 1)],
    "scoop": [("serve some soup with the {o}", 1), ("scoop up the beans using the {o}{m}", 1)],
    "stir": [("mix it up with the {o}", 1), ("give the pot a stir with the {o}{m}", 1)],
    "cut": [("chop the onion with the {o}", 1), ("make a cut using the {o}{m}", 1)],
    "squeeze": [("clamp down on the {o}", 1), ("press the {o} handles together{m}", 1), ("loosen your grip on the {o}", -1)],
    "handover": [("could you bring me the {o}", 1), ("i need the {o}, pass it over", 1)],
}
MAGS = {  # (phrase, value, unit)
    "deg": [(" by {v} degrees", "deg"), (" {v} degrees", "deg"), (" about {v} deg", "deg")],
    "cm": [(" by {v} cm", "cm"), (" {v} centimeters", "cm"), (" about {v} cm", "cm")],
}
UNIT = dict(pour="deg", lift="cm", pivot="deg", turn="deg", hinge="deg", slide="cm", press="cm", swing="deg",
            scoop="deg", stir="deg", cut="cm", squeeze="deg", handover="deg")
FILLERS_TRAIN = ["", "please ", "robot, ", "can you ", "now "]
FILLERS_TEST = ["hey, ", "would you ", "quickly ", "um, "]


def make(split, n_per_task, seed=0, objects="train"):
    rng = random.Random(seed)
    T = TRAIN if split == "train" else TEST
    fill = FILLERS_TRAIN if split == "train" else FILLERS_TEST
    rows = []
    for task in TASKS:
        objs = OBJECTS[task][0] if objects == "train" else OBJECTS[task][1]
        for _ in range(n_per_task):
            tpl, direction = rng.choice(T[task])
            o = rng.choice(objs)
            mag, val = "", None
            if "{m}" in tpl and rng.random() < 0.5:
                ph, unit = rng.choice(MAGS[UNIT[task]])
                val = rng.choice([5, 10, 15, 20, 30, 45, 60, 90, 120, 180]) if unit == "deg" else rng.choice([2, 5, 8, 10, 12, 15, 20])
                mag = ph.format(v=val)
            text = (rng.choice(fill) + tpl.format(o=o, m=mag)).strip()
            rows.append(dict(text=text, task=task, direction=direction, part=OBJECTS[task][2], magnitude=val,
                             unit=UNIT[task] if val is not None else None, object=o))
    rng.shuffle(rows)
    return rows


# Hand-written natural commands (not produced by any template above), for a final check.
HANDWRITTEN = [
    ("I'm thirsty, can you pour me what's in that mug", "pour", 1), ("dump the leftover coffee out of the cup", "pour", 1),
    ("bring the kettle up off the counter", "lift", 1), ("pick that drill up for me", "lift", 1),
    ("tip the cereal box back a bit", "pivot", 1), ("lean that carton against the wall side", "pivot", 1),
    ("get the cap off this bottle", "turn", 1), ("the jar is stuck, unscrew it", "turn", 1),
    ("screw the lid back onto the jar tightly", "turn", -1), ("open up the pill bottle", "turn", 1),
    ("open the laptop", "hinge", 1), ("lift up the toilet seat", "hinge", 1), ("shut the lid of the trash can", "hinge", -1),
    ("close the cooler so it stays cold", "hinge", -1), ("pull the top drawer out", "slide", 1),
    ("open the nightstand drawer halfway", "slide", 1), ("push the drawer back in please", "slide", -1),
    ("ring the doorbell", "press", 1), ("turn on the light by pressing the switch", "press", 1),
    ("push the red button", "press", 1), ("drive the nail in with the hammer", "swing", 1),
    ("whack the stake with the mallet", "swing", 1), ("get some soup out of the pot with the ladle", "scoop", 1),
    ("scoop the rice with the spoon", "scoop", 1), ("whisk the eggs", "stir", 1), ("stir the soup slowly", "stir", 1),
    ("cut the cake into slices with the knife", "cut", 1), ("saw this plank in half", "cut", 1),
    ("squeeze the tongs to grab the salad", "squeeze", 1), ("let the pliers open again", "squeeze", -1),
    ("staple these pages", "squeeze", 1), ("could you give me that screwdriver", "handover", 1),
    ("pass me the scissors handle first", "handover", 1), ("hand the wrench over", "handover", 1),
    ("open the bottle of water", "turn", 1), ("close the drawer", "slide", -1), ("close the box", "hinge", -1),
    ("open the trash can", "hinge", 1), ("open the cabinet drawer by 10 cm", "slide", 1),
    ("tighten the thermos cap", "turn", -1),
]
