import json

from options import LABEL_MAP_PATH

if __name__ == "__main__":
    lines = 0
    with open("classes.txt", "r") as f:
        lines = f.readlines()

    classes = None
    for i, line in enumerate(lines):
        levels = line.split("_")[1:]
        assert len(levels) == 7
        if classes is None:
            classes = []
            for _ in range(len(levels)):
                classes.append(set())
        for lvl, name in enumerate(levels):
            classes[lvl].add("_".join(levels[:lvl+1]))

    label_maps = []
    for lvl, lvl_classes in enumerate(classes):
        lvl_classes_sorted = sorted(list(lvl_classes))
        lbl_map = {}
        for i, cls in enumerate(lvl_classes_sorted):
            lbl_map[i] = cls
        label_maps.append(lbl_map)

    with open(LABEL_MAP_PATH, "w") as f:
        json.dump(label_maps, f, indent = 4)
    