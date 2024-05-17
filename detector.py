import numpy as np
import onnxruntime

from utils import *

class Detector:
    def __init__(self, 
                 weights_path: str, 
                 conf_thres: float=0.1, 
                 iou_thres: float=0.5):
        
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

        # Initialize model
        self.initialize_model(weights_path)

    def __call__(self, image):
        return self.detect_objects(image)

    def initialize_model(self, path):
        self.session = onnxruntime.InferenceSession(path,
                                                    providers=['CUDAExecutionProvider',
                                                               'CPUExecutionProvider'])
        # Get model info
        self.get_input_details()
        self.get_output_details()

    def detect_objects(self, image):
        input_tensor = self.prepare_input(image)

        # Perform inference on the image
        outputs = self.inference(input_tensor)

        self.boxes, self.scores, self.class_ids = self.postprocess(outputs)

        return self.boxes, self.scores, self.class_ids

    # def prepare_input(self, raw_bgr_image):
    #     self.img_height, self.img_width = raw_bgr_image.shape[:-1]
    #     img, ratio, (dw, dh) = letterbox(raw_bgr_image, (self.input_height, self.input_width), auto=False)
    #     img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    #     img = np.ascontiguousarray(img).astype(np.float32)
    #     img /= 255
    #     img = img[None]

    #     return img
    def prepare_input(self, image):
        self.img_height, self.img_width = image.shape[:2]

        input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize input image
        input_img = cv2.resize(input_img, (self.input_width, self.input_height))

        # Scale input pixel values to 0 to 1
        input_img = input_img / 255.0
        input_img = input_img.transpose(2, 0, 1)
        input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)

        return input_tensor


    def inference(self, input_tensor):
        outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})

        return outputs
    
    def postprocess(self, output):
        predictions = np.squeeze(output[0]).T

        # Filter out object confidence scores below threshold
        scores = np.max(predictions[:, 4:], axis=1)
        predictions = predictions[scores > self.conf_thres, :]
        scores = scores[scores > self.conf_thres]

        if len(scores) == 0:
            return [], [], []

        # Get the class with the highest confidence
        class_ids = np.argmax(predictions[:, 4:], axis=1)

        # Get bounding boxes for each object
        boxes = self.extract_boxes(predictions)

        # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
        # indices = nms(boxes, scores, self.iou_threshold)
        indices = multiclass_nms(boxes, scores, class_ids, self.iou_thres)

        return boxes[indices], scores[indices], class_ids[indices]

    def extract_boxes(self, predictions):
        # Extract boxes from predictions
        boxes = predictions[:, :4]

        # Scale boxes to original image dimensions
        boxes = self.rescale_boxes(boxes)

        # Convert boxes to xyxy format
        boxes = xywh2xyxy(boxes)

        return boxes

    def rescale_boxes(self, boxes):
        # Rescale boxes to original image dimensions
        input_shape = np.array([self.input_width, self.input_height, self.input_width, self.input_height])

        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array([self.img_width, self.img_height, self.img_width, self.img_height])
        return boxes
    
    def get_input_details(self):
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]

        self.input_shape = model_inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

    def get_output_details(self):
        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]



if __name__ == "__main__":
    model = Detector('./yolov8n.onnx',
                        conf_thres=0.5, 
                        iou_thres=0.5)
    
    test_image_path = './test.jpeg'
    bgr_image = cv2.imread(test_image_path)

    boxes, scores, class_ids = model(bgr_image)
    print(boxes)
    print(class_ids)

    for box in boxes:
        box = np.array(box,np.int32)
        x1,y1,x2,y2 = box

        bgr_image = cv2.rectangle(bgr_image, (x1,y1), (x2,y2), (0,255,0), 2) 

    cv2.imwrite('./test_result.jpeg', bgr_image)
    





