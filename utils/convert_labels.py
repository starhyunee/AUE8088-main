import os
import argparse
import xml.etree.ElementTree as ET
from tqdm import tqdm

def convert_box(size, xmin, ymin, w, h):

    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    xmax = xmin + w
    ymax = ymin + h
    x_center = (xmin + xmax) / 2.0
    y_center = (ymin + ymax) / 2.0
    return x_center * dw, y_center * dh, w * dw, h * dh


def convert_xml_to_yolo(xml_path, txt_path, class_map):

    import xml.etree.ElementTree as ET

    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Get image size
    size = root.find('size')
    w_img = int(size.find('width').text)
    h_img = int(size.find('height').text)

    with open(txt_path, 'w') as out_file:
        for obj in root.iter('object'):
            cls_name = obj.find('name').text
            if cls_name not in class_map:
                continue  # skip unknown class

            cls_id = class_map[cls_name]

            xmlbox = obj.find('bndbox')
            xmin = float(xmlbox.find('x').text)
            ymin = float(xmlbox.find('y').text)
            bw = float(xmlbox.find('w').text)
            bh = float(xmlbox.find('h').text)

            x_center, y_center, bw_norm, bh_norm = convert_box((w_img, h_img), xmin, ymin, bw, bh)

            # Ignore invalid/negative boxes
            if any(v < 0 or v > 1 for v in [x_center, y_center, bw_norm, bh_norm]):
                continue

            out_file.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {bw_norm:.6f} {bh_norm:.6f}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=str, required=True, help='Directory with XML annotation files')
    parser.add_argument('--output-dir', type=str, required=True, help='Directory to save YOLO label txt files')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Define class name to ID mapping (must match your .yaml file)
    class_map = {
        'person': 0,
        'cyclist': 1,
        'people': 2,
        'person?': 3
    }

    xml_files = [f for f in os.listdir(args.input_dir) if f.endswith('.xml')]
    for xml_file in tqdm(xml_files, desc="Converting"):
        xml_path = os.path.join(args.input_dir, xml_file)
        txt_file = xml_file.replace('.xml', '.txt')
        txt_path = os.path.join(args.output_dir, txt_file)
        convert_xml_to_yolo(xml_path, txt_path, class_map)

if __name__ == "__main__":
    main()
