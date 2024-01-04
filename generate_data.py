import os
from parameters import Parameters
import cv2 as cv
import numpy as np

class DataGenerator:
    def __init__(self):
        params: Parameters = Parameters()

        self.positive_images_path = params.dir_pos_examples
        self.negative_images_path = params.dir_neg_examples
        self.negative_examples = params.number_negative_examples
    

    def generate_positive_images(self):
        params: Parameters = Parameters()

        positive_images_path = params.dir_pos_examples

        if not os.path.exists(positive_images_path):
            os.makedirs(positive_images_path)
            print('directory created: {} '.format(positive_images_path))
        else:
            print('directory {} exists '.format(positive_images_path))
            for file in os.listdir(positive_images_path):
                os.remove(os.path.join(positive_images_path, file))

        with open('validare/task1_gt_validare.txt', 'r') as f:
            content = f.readlines()

        content = [x.strip() for x in content]

        images_dict = {}
        for line in content:
            line = line.split(' ')
            if line[0] in images_dict.keys():
                images_dict[line[0]].append(line[1:])
            else:
                images_dict[line[0]] = [line[1:]]

        for kvp in images_dict.items():
            image = cv.imread(os.path.join('validare/validare', kvp[0]))

            for i, bbox in enumerate(kvp[1]):
                bbox = [int(x) for x in bbox]
                
                cut_image = image[bbox[1]:bbox[3], bbox[0]:bbox[2]]

                cut_image_path = os.path.join(positive_images_path, f"{kvp[0]}_{i}0.jpg")

                cv.imwrite(cut_image_path, cut_image)

                flip_image = cv.flip(cut_image, 1)

                flip_image_path = os.path.join(positive_images_path, f"{kvp[0]}_{i}1.jpg")

                cv.imwrite(flip_image_path, flip_image)

        print('positive images generated')

        labeled_images_dir = 'antrenare'
        charcters = ['barney', 'betty', 'fred', 'wilma']

        for character in charcters:
            character_images_dir = os.path.join(labeled_images_dir, character)
            character_annotations_path = os.path.join(labeled_images_dir, f"{character}_annotations.txt")

            with open(character_annotations_path, 'r') as f:
                content = f.readlines()

            content = [x.strip() for x in content]

            for file in os.listdir(character_images_dir):
                image = cv.imread(os.path.join(character_images_dir, file))

                annotations = [x for x in content if x.startswith(file)]

                for i, annotation in enumerate(annotations):
                    annotation = annotation.split(' ')
                    bbox = [int(x) for x in annotation[1:-1]]

                    cut_image = image[bbox[1]:bbox[3], bbox[0]:bbox[2]]

                    cut_image_path = os.path.join(positive_images_path, f"{character}_{file}_{i}0.jpg")

                    cv.imwrite(cut_image_path, cut_image)

                    flip_image = cv.flip(cut_image, 1)

                    flip_image_path = os.path.join(positive_images_path, f"{character}_{file}_{i}1.jpg")

                    cv.imwrite(flip_image_path, flip_image)

            print('positive images generated for {}'.format(character))

        print('number of positive images: {}'.format(len(os.listdir(positive_images_path))))

    def intersection_over_union(bbox_a, bbox_b):
        x_a = max(bbox_a[0], bbox_b[0])
        y_a = max(bbox_a[1], bbox_b[1])
        x_b = min(bbox_a[2], bbox_b[2])
        y_b = min(bbox_a[3], bbox_b[3])

        inter_area = max(0, x_b - x_a + 1) * max(0, y_b - y_a + 1)

        box_a_area = (bbox_a[2] - bbox_a[0] + 1) * (bbox_a[3] - bbox_a[1] + 1)
        box_b_area = (bbox_b[2] - bbox_b[0] + 1) * (bbox_b[3] - bbox_b[1] + 1)
        iou = inter_area / float(box_a_area + box_b_area - inter_area)

        return iou
    
    def _does_bbox_overlap_with_any_gt_bbox(self, bboxes, potential_bbox):
        for bbox in bboxes:
            if self.intersection_over_union(bbox, potential_bbox) > 0.1:
                return True

        return False

    def generate_negative_images(self):
        params: Parameters = Parameters()

        negative_images_path = params.dir_neg_examples

        if not os.path.exists(negative_images_path):
            os.makedirs(negative_images_path)
            print('directory created: {} '.format(negative_images_path))
        else:
            print('directory {} exists '.format(negative_images_path))
            for file in os.listdir(negative_images_path):
                os.remove(os.path.join(negative_images_path, file))

        with open('validare/task1_gt_validare.txt', 'r') as f:
            content = f.readlines()

        content = [x.strip() for x in content]

        images_dict = {}
        for line in content:
            line = line.split(' ')
            if line[0] in images_dict.keys():
                images_dict[line[0]].append(line[1:])
            else:
                images_dict[line[0]] = [line[1:]]

        desired_negative_images_per_image = self.negative_examples // len(images_dict.keys())

        if self.negative_examples % len(images_dict.keys()) != 0:
            desired_negative_images_per_image += 1

        number_generated_negative_images = 0
        images = os.listdir('validare/validare').sort()
        index = 0

        while number_generated_negative_images < self.negative_examples:
            image = cv.imread(os.path.join('validare/validare', images[index]))

            image_face_bboxes = images_dict[images[index]]

            min_dimension = min(image.shape[0], image.shape[1])

            for _ in range(desired_negative_images_per_image):

                is_valid = False
                x, y, size = 0, 0, 0

                while not is_valid:
                    size = np.random.randint(min_dimension // 10, min_dimension // 2)

                    x = np.random.randint(0, image.shape[1] - size)
                    y = np.random.randint(0, image.shape[0] - size)

                    potential_bbox = [x, y, x + size, y + size]

                    if not self._does_bbox_overlap_with_any_gt_bbox(image_face_bboxes, potential_bbox):
                        is_valid = True

                cut_image = image[y:y + size, x:x + size]

                cut_image_path = os.path.join(negative_images_path, f"{index}_{number_generated_negative_images}.jpg")

                cv.imwrite(cut_image_path, cut_image)

                number_generated_negative_images += 1

                if number_generated_negative_images == self.negative_examples:
                    break

            index += 1