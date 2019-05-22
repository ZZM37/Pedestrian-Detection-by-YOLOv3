import numpy as np
from PIL import Image

class YOLO_Kmeans:

    def __init__(self, cluster_number, filename):
        self.cluster_number = 6
        self.filename = "model_data/INRIA_train.txt"

    def iou(self, boxes, clusters):  # 1 box -> k clusters
        n = boxes.shape[0]
        k = cluster_number

        box_area = boxes[:, 0] * boxes[:, 1]
        box_area = box_area.repeat(k)
        box_area = np.reshape(box_area, (n, k))

        cluster_area = clusters[:, 0] * clusters[:, 1]
        cluster_area = np.tile(cluster_area, [1, n])
        cluster_area = np.reshape(cluster_area, (n, k))

        box_w_matrix = np.reshape(boxes[:, 0].repeat(k), (n, k))
        cluster_w_matrix = np.reshape(np.tile(clusters[:, 0], (1, n)), (n, k))
        min_w_matrix = np.minimum(cluster_w_matrix, box_w_matrix)

        box_h_matrix = np.reshape(boxes[:, 1].repeat(k), (n, k))
        cluster_h_matrix = np.reshape(np.tile(clusters[:, 1], (1, n)), (n, k))
        min_h_matrix = np.minimum(cluster_h_matrix, box_h_matrix)
        inter_area = np.multiply(min_w_matrix, min_h_matrix)

        result = inter_area / (box_area + cluster_area - inter_area)
        return result

    def avg_iou(self, boxes, clusters):
        accuracy = np.mean([np.max(self.iou(boxes, clusters), axis=1)])
        return accuracy

    def kmeans(self, boxes, k, dist=np.median):
        box_number = boxes.shape[0]
        distances = np.empty((box_number, k))
        last_nearest = np.zeros((box_number,))
        np.random.seed()
        clusters = boxes[np.random.choice(
            box_number, k, replace=False)]  # init k clusters
        while True:

            distances = 1 - self.iou(boxes, clusters)

            current_nearest = np.argmin(distances, axis=1)
            if (last_nearest == current_nearest).all():
                break  # clusters won't change
            for cluster in range(k):
                clusters[cluster] = dist(boxes[current_nearest == cluster], axis=0)
                # update clusters

            last_nearest = current_nearest

        return clusters

    def kmeans_ratio(self, boxes, k, dist=np.mean):
        box_number = boxes.shape[0]
        distances = np.empty((box_number, k))
        last_nearest = np.zeros((box_number,))
        np.random.seed()
        clusters = boxes[np.random.choice(
            box_number, k, replace=False)]  # init k clusters
        while True:
            clusters_T = clusters.T
            distances = np.abs(boxes-clusters_T)

            current_nearest = np.argmin(distances, axis=1)
            if (last_nearest == current_nearest).all():
                break  # clusters won't change
            for cluster in range(k):
                clusters[cluster] = dist(boxes[current_nearest == cluster], axis=0)
                # update clusters

            last_nearest = current_nearest

        return clusters

    def kmeans_area(self, boxes, k, dist=np.mean):
        box_number = boxes.shape[0]
        distances = np.empty((box_number, k))
        last_nearest = np.zeros((box_number,))
        np.random.seed()
        clusters = boxes[np.random.choice(
            box_number, k, replace=False)]  # init k clusters
        while True:
            clusters_T = clusters.T
            distances = np.abs(np.sqrt(boxes)-np.sqrt(clusters_T))

            current_nearest = np.argmin(distances, axis=1)
            if (last_nearest == current_nearest).all():
                break  # clusters won't change
            for cluster in range(k):
                clusters[cluster] = dist(boxes[current_nearest == cluster], axis=0)
                # update clusters

            last_nearest = current_nearest
        
        return clusters

    def result2txt(self, data):
        f = open("model_data/INRIA_anchor_tiny_K.txt", 'w')     # model_data/bdd100k_anchors.txt
        row = np.shape(data)[0]
        for i in range(row):
            if i == 0:
                x_y = "%d,%d" % (data[i][0], data[i][1])
            else:
                x_y = ", %d,%d" % (data[i][0], data[i][1])
            f.write(x_y)
        f.close()

    def txt2boxes(self, input_shape = (416, 416)):
        f = open(self.filename, 'r')
        box_dataSet = []; ratio_dataset = []; area_dataset = []
        for line in f:
            infos = line.split()
            length = len(infos)
            image = Image.open(infos[0])
            iw, ih = image.size
            h, w = input_shape
            scale = min(w/iw, h/ih)

            for i in range(1, length):
                width = int(infos[i].split(",")[2]) - int(infos[i].split(",")[0])
                width = int(width*scale)
                height = int(infos[i].split(",")[3]) - int(infos[i].split(",")[1])
                height = int(height*scale)
                ratio = width/height
                area = width*height
                box_dataSet.append([width, height])
                ratio_dataset.append([ratio])
                area_dataset.append([area])

        box_result = np.array(box_dataSet)
        ratio_result = np.array(ratio_dataset)
        area_result = np.array(area_dataset)
        f.close()
        return box_result, ratio_result, area_result

    def txt2clusters(self):
        all_boxes = self.txt2boxes()
        result = self.kmeans(all_boxes, k=self.cluster_number)
        result = result[np.lexsort(result.T[0, None])]
        self.result2txt(result)
        print("K anchors:\n {}".format(result))
        print("Accuracy: {:.2f}%".format(self.avg_iou(all_boxes, result) * 100))


if __name__ == "__main__":
    cluster_number = 2
    filename = "model_data/INRIA_train.txt"
    kmeans = YOLO_Kmeans(cluster_number, filename)
    boxes, ratio, area = kmeans.txt2boxes()
    print(kmeans.kmeans_area(area, cluster_number))
