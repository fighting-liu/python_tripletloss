import sys

sys.path.insert(0, '/core1/data/home/liuhuawei/evalution_ai/image')
from pretrain import AnnoData
import json


def load_data(wrt_json_file):
    ad = AnnoData(mysql_conf_sec='mysql_internal', \
        conf_file='/core1/data/home/liuhuawei/evalution_ai/image/pretrain.cfg')  
    select_sql = '''
    select o.id as object_id, concat('http://192.168.1.23:8082/', i.fid) as img_path, floor(o.x_pixel) as xmin, floor(o.y_pixel) as ymin, floor(o.width_pixel) as width, floor(o.height_pixel) as height,
    o.category_id as clothes_type, i.status as source_type, i.src_id as item_id
    from object_copy_and_mark o inner join image_copy_and_mark i on o.img_id=i.id
    where o.category_id!=2 and o.category_id!=7 and (i.status=14 or i.status=13) and i.processing_id>0 and i.processing_id!=888 
    limit 20
    '''       
    all_data = ad.mysql.select(select_sql)
    objectid_to_metadata = {str(data['object_id']):data for data in all_data}

    with open(wrt_json_file, 'w') as f:
        f.write(json.dumps(objectid_to_metadata)) 
    
if __name__ == '__main__':
    load_data('./data/objectid_to_metadata.json')
