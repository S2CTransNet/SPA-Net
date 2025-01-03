def get_obj_name(dataset):
    if dataset == 'ShapeNet-55' or dataset == 'ShapeNet-34':
        obj_names = {

            "03759954": "microphone",

            "04330267": "stove",

            "03261776": "earphone",

            "03513137": "helmet",

            "04460130": "tower",

            "04468005": "train",

            "03761084": "microwaves",

            "04004475": "printer",

            "03938244": "pillow",

            "02992529": "cellphone",

            "02808440": "bathtub",

            "04530566": "watercraft",

            "02871439": "bookshelf",

            "03593526": "jar",

            "04554684": "washer",

            "03467517": "guitar",

            "04401088": "telephone",

            "02954340": "cap",

            "02933112": "cabinet",

            "03642806": "laptop",

            "02924116": "bus",

            "02946921": "can",

            "02818832": "bed",

            "04256520": "sofa",

            "04379243": "table",

            "02747177": "trash bin",

            "03046257": "clock",

            "04225987": "skateboard",

            "03797390": "mug",

            "03001627": "chair",

            "03691459": "loudspeaker",

            "02942699": "camera",

            "03636649": "lamp",

            "02691156": "airplane",

            "02773838": "bag",

            "02958343": "car",

            "02880940": "bowl",

            "04099429": "rocket",

            "02828884": "bench",

            "04090263": "rifle",

            "03211117": "display",

            "02876657": "bottle",

            "03710193": "mailbox",

            "03790512": "motorbike",

            "03325088": "faucet",

            "02801938": "basket",

            "03337140": "file cabinet",

            "02843684": "birdhouse",

            "03624134": "knife",

            "03991062": "flowerpot",

            "03948459": "pistol",

            "03928116": "piano",

            "03207941": "dishwasher",

            "04074963": "remote",

            "03085013": "keyboard"

}

    elif dataset == 'MVP':
        obj_names = {
            "0": "airplane",

            "1": "cabinet",

            "2": "car",

            "3": "chair",

            "4": "lamp",

            "5": "sofa",

            "6": "table",

            "7": "watercraft",

            "8": "bed",

            "9": "bench",

            "10": "bookshelf",

            "11": "bus",

            "12": "guitar",

            "13": "motorbike",

            "14": "pistol",

            "15": "skateboard"

        }
    return obj_names
