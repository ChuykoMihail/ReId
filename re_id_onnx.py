 
import onnx
import onnxruntime as ort
import cv2
import numpy as np

class ReIdOnnx():
    def __init__(self, model_path) -> None:
        self.model = onnx.load(model_path)
        self.input_nodes = self.model.graph.input
        self.input_names = [node.name for node in self.input_nodes]
        self.input_shapes = [(node.name, node.type.tensor_type.shape) for node in self.input_nodes]
        self.ort_sess = ort.InferenceSession(model_path)

    def get_features(self, img_array):
        array_img = self.preprocess_img(img_array)
        features = self.ort_sess.run(None, {'input.1': array_img})[0]
        return features
    
    def preprocess_img(self, img_array):
        img = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (128, 384), interpolation=cv2.INTER_AREA)
        img = img.transpose(2, 0, 1)
        img = img / 255.0
        img = np.array(img)
        img = np.expand_dims(img, axis=0)
        img = np.float32(img)
        return img