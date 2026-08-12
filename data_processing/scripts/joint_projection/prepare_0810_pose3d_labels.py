import prepare_0806_pose3d_labels as p
from constants_0810_training import SESSION_ORDER

p.LIMB_ORDER = SESSION_ORDER

if __name__ == "__main__":
    raise SystemExit(p.main())
