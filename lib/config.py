
USE_LMDB = False
IMG_LMDB_PATH = '/data/data/liuhuawei/data_lmdb_backup_for_ssd/data_lmdb_for_image_copy_and_mark_data'

# Snapshot iteration 
METADATA_JSON = './data/objectid_to_metadata.json'

## json path to triplelt file, key:(a_objectid, p_objectid) val:[n1_ob, n2_ob, ....]
TRIPLET_JSON = './data/test.json'

## image config
TARGET_SIZE = 224
PIXEL_MEANS = [104.0, 117.0, 123.0]

## The number of samples in each minibatch
BATCH_SIZE = 39

## prefetch process for data layer (must be false here)
USE_PREFETCH = False
RNG_SEED = 8

