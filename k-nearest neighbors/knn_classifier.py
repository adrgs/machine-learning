import numpy as np


class KnnClassifier(object):
    class Metric(object):
        @staticmethod
        def manhattan_distance(a, b):
            """
            Calculates the manhattan distance (L1) between two vectors 
            
            Parameters
            ----------
            a : numpy.array of shape (n,)
            b : numpy.array of shape (n,)

            Returns
            -------
            distance : float
            """
            return np.linalg.norm(a - b, 1)

        @staticmethod
        def euclidian_distance(a, b):
            """
            Calculates the euclidian distance (L2) between two vectors 

            Parameters
            ----------
            a : numpy.array of shape (n,)
            b : numpy.array of shape (n,)

            Returns
            -------
            distance : float
            """
            return np.linalg.norm(a - b, 2)

        @staticmethod
        def chebyshev_distance(a, b):
            """
            Calculates the chebyshev distance (L infinity) between two vectors

            Parameters
            ----------
            a : numpy.array of shape (n,)
            b : numpy.array of shape (n,)

            Returns
            -------
            distance : float
            """
            return np.linalg.norm(a - b, np.inf)

    def __init__(self, train_images, train_labels, num_neighbors=3, metric=Metric.euclidian_distance, cache_distance=True):
        """
        KnnClassifier constructor

        Parameters
        ----------
        train_images : numpy.ndarray
            numpy.ndarray of shape = (n,m) where n is the number of images to be "trained" and m is the number of total pixels in the iamge
        train_labels : numpy.ndarray
            numpy.ndarray of shape = (n,) where n is the number of images to be "trained"
        num_neighbors : int, optional
            Number of closest images that the algorithm will use for finding the best match
            Default value is 3
        metric : lambda function(a,b) -> distance(a,b), optional
            Function that calculates the distance between two images, parameters a,b should be of type numpy.ndarray
            Default distance function is the euclidian distance (L2)
        cache_distance : bool, optional
            Whether to cache distance results for tests, setting this to False will reduce memory usage
            but will greatly increase execution time
        """
        self.train_images = train_images
        self.train_labels = train_labels
        self.num_neighbors = num_neighbors
        self.metric = metric
        if cache_distance:
            self.distance_cache = {}

    @staticmethod
    def _check_test_parameters(test_images, test_labels):
        """
        Check if the parameters meet basic criteria for the class' usage

        Parameters
        ----------
        test_images : numpy.ndarray
        test_labels : numpy.ndarray

        Raises Exceptions
        -------
        ValueError : parameters have different lengths or they are empty
        TypeError : parameters are not of type numpy.ndarray
        """
        try:
            if test_images.shape[0] != test_labels.shape[0]:
                raise ValueError("Parameters test_images and test_labels have different lengths")
            if test_images.shape[0] == 0:
                raise ValueError("Parameters test_images and test_labels should not be empty")
        except:
            raise TypeError("Parameters test_images and test_labels need to be numpy arrays")

    @staticmethod
    def _accuracy(y_true, y_pred):
        """
        Calculates the accuracy of a classifier

        Parameters
        ----------
        y_true : array_like
            Array of known/true values for test labels
        y_pred : array_like
            Array of predictions that the classifier made

        Raises Exceptions
        ----------
        ValueError : if the parameters have different lengths or if they are empty

        Returns
        -------
        accuracy : float
            Returns (correct number of predictions / total number of predictions)
        """
        if len(y_true) != len(y_pred):
            raise ValueError("Parameters y_true and y_pred have different lengths")
        n = len(y_true)
        if n == 0:
            raise ValueError("Parameters y_true and y_pred can't be empty arrays")
        correct_pred = 0

        for pred_label, true_label in zip(y_pred, y_true):
            if pred_label == true_label:
                correct_pred += 1

        return correct_pred / n


    @staticmethod
    def _confusion_matrix(y_true, y_pred):
        """
        Calculates the confusion matrix for a classifier
        The true value of the label is given by the row index
        The value of the prediction is given by the column index
        The main diagonal marks accurate predictions

        The labels should be normalized to int values between 0 and n-1
        Where n = number of different labels

        Parameters
        ----------
        y_true : array_like
            Array of known/true values for test labels
        y_pred : array_like
            Array of predictions that the classifier made

        Raises Exceptions
        ----------
        ValueError : if the parameters have different lengths or if the values are not normalized

        Returns
        -------
        matrix : numpy.ndarray of shape=(n,n) where n = number of different labels
        """
        if len(y_true) != len(y_pred):
            raise ValueError("Parameters y_true and y_pred have different lengths")
        n = np.max(np.array(y_true)) + 1
        matrix = np.zeros((n,n))
        for pred_label, true_label in zip(y_pred, y_true):
            try:
                matrix[true_label, pred_label] += 1
            except:
                raise ValueError("Parameters y_true and y_pred are not normalized")
        return matrix

    def classify_image(self, test_image, num_neighbors=None, metric=None, return_neighbors_idx=False):
        """
        Classifies a given image based on the train data

        Parameters
        ----------
        test_image : numpy.array of shape = (n,)
        num_neighbors : int, optional
            If not given, the object's self.num_neighbors will be used instead
        metric : lambda function(a,b) -> distance between numpy.array a and b of shape = (n,), optional
            If not given, the object's self.metric will be used instead
        return_neighbors_idx : bool, optional

        Returns
        -------
        prediction : numerical value
            Returns the label that best matches the given test image
        <closest_neighbors_idx : array of ints, [only if return_neighbors_idx] is set to True>
            Returns the indexes of the closest [num_neighbors] matches from the classifier's train images
        """
        num_neighbors = (num_neighbors or self.num_neighbors)
        metric = (metric or self.metric)

        if type(self.distance_cache) is dict and metric in self.distance_cache and tuple(test_image) in self.distance_cache[metric]:
            distances = self.distance_cache[metric][tuple(test_image)]
        else:
            distances = []
            for train_image in self.train_images:
                distances.append(metric(test_image, train_image))
            distances = np.array(distances)

            if type(self.distance_cache) is dict:
                if metric not in self.distance_cache:
                    self.distance_cache[metric] = {}
                self.distance_cache[metric][tuple(test_image)] = distances


        sorted_indexes = distances.argsort()
        closest_neighbors_idx = sorted_indexes[:num_neighbors]

        closest_labels = []
        for neighbor_idx in closest_neighbors_idx:
            closest_labels.append(self.train_labels[neighbor_idx])

        label_values, label_counts = np.unique(closest_labels, return_counts=True)
        majority_label_idx = np.argmax(label_counts)

        if return_neighbors_idx==True:
            return label_values[majority_label_idx], closest_neighbors_idx

        return label_values[majority_label_idx]

    def get_predictions(self, test_images, num_neighbors=None, metric=None):
        """
        Classifies n given images based on the train data

        Parameters
        ----------
        test_images : numpy.array of shape = (n,m) where n = the number of images and m = the number of pixels in an image
        num_neighbors : int, optional
            If not given, the object's self.num_neighbors will be used instead
        metric : lambda function(a,b) -> distance between numpy.array a and b of shape = (n,), optional
            If not given, the object's self.metric will be used instead

        Returns
        -------
        predictions : array of numerical values
            Returns an array that contains the matched label for each test image
        """
        predictions = []
        for test_image in test_images:
            prediction = self.classify_image(test_image, num_neighbors, metric)
            predictions.append(prediction)
        return np.array(predictions)

    def get_accuracy(self, test_images, test_labels, num_neighbors=None, metric=None):
        """
        Calculates the accuracy of the classifier given the test data

        Parameters
        ----------
        test_images : numpy.array of shape = (n,m) where n = the number of images and m = the number of pixels in an image
        test_labels : numpy.array of shape = (n,) where n = the number of images
        num_neighbors : int, optional
            If not given, the object's self.num_neighbors will be used instead
        metric : lambda function(a,b) -> distance between numpy.array a and b of shape = (n,), optional
            If not given, the object's self.metric will be used instead

        Returns
        -------
        accuracy : float
            Returns the accuracy of the classifier given the test data
        """
        num_neighbors = (num_neighbors or self.num_neighbors)
        metric = (metric or self.metric)
        self._check_test_parameters(test_images, test_labels)

        total_test_images = test_images.shape[0]

        predictions = self.get_predictions(test_images, num_neighbors, metric)

        return self._accuracy(test_labels, predictions)

    def get_confusion_matrix(self, test_images, test_labels, num_neighbors=None, metric=None, normalize_dict = None):
        """
        Calculates the confusion matrix of the classifier given the test data

        Parameters
        ----------
        test_images : numpy.array of shape = (n,m) where n = the number of images and m = the number of pixels in an image
        test_labels : numpy.array of shape = (n,) where n = the number of images
        num_neighbors : int, optional
            If not given, the object's self.num_neighbors will be used instead
        metric : lambda function(a,b) -> distance between numpy.array a and b of shape = (n,), optional
            If not given, the object's self.metric will be used instead
        normalize_dict : dict<numerical value, int>, optional
            A dictionary used for normalizing the given label data into values
            from 0 to n-1, where n = the number of different labels

        Returns
        -------
        matrix : numpy.ndarray of shape=(n,n) where n = number of different labels
        """
        num_neighbors = (num_neighbors or self.num_neighbors)
        metric = (metric or self.metric)
        self._check_test_parameters(test_images, test_labels)

        predictions = self.get_predictions(test_images)

        if normalize_dict:
            normalized_predictions, normalized_test_labels = [], []
            for prediction, test_label in zip(predictions, test_labels):
                normalized_predictions.append(normalize_dict[prediction])
                normalized_test_labels.append(normalize_dict[test_label])
        else:
            normalized_predictions, normalized_test_labels = predictions, test_labels

        return self._confusion_matrix(normalized_test_labels, normalized_predictions)


def load_data(data_path):
    train_images = np.loadtxt(data_path + "train_images.txt", 'int32')
    train_labels = np.loadtxt(data_path + "train_labels.txt", 'int8')
    test_images = np.loadtxt(data_path + "test_images.txt", 'int32')
    test_labels = np.loadtxt(data_path + "test_labels.txt", 'int8')
    return train_images, train_labels, test_images, test_labels


if __name__ == '__main__':
    train_images, train_labels, test_images, test_labels = load_data("data/")
    knn_classifier = KnnClassifier(train_images, train_labels)

    prediction_first_test_image = knn_classifier.classify_image(test_images[0])
    print(f"The first test image is {test_labels[0]}, our prediction: {prediction_first_test_image}")

    knn_classifier.num_neighbors = 3
    knn_classifier.metric = KnnClassifier.Metric.euclidian_distance

    # predictions = knn_classifier.get_predictions(test_images)
    predictions = knn_classifier.get_predictions(test_images, num_neighbors=3, metric=KnnClassifier.Metric.euclidian_distance)
    np.savetxt("pred_3nn_l2_mnist.txt", predictions)

    # acc = knn_classifier.get_accuracy(test_images, test_labels)
    acc = knn_classifier._accuracy(test_labels, predictions)
    print(f"KNN classifier accuracy is {acc*100}%")

    # conf_matrix = knn_classifier.get_confusion_matrix(test_images, test_labels)
    conf_matrix = knn_classifier._confusion_matrix(test_labels, predictions)
    np.set_printoptions(threshold=np.inf)
    print(conf_matrix)
    np.set_printoptions(threshold=1000)

    import matplotlib.pyplot as plt
    label_value = 5
    pred_value = 3
    misclassified = int(conf_matrix[label_value, pred_value])
    print(f"The pair ({label_value},{pred_value}) has been misclassified {misclassified} times")
    # Get misclassified test indexes

    misclassified_images = []
    for i in range(len(test_images)):
        if int(test_labels[i]) != label_value:
            continue
        prediction, neighbors_idx = knn_classifier.classify_image(test_images[i], return_neighbors_idx=True)
        if int(prediction) == pred_value:
            misclassified_images.append((i, neighbors_idx))

    num_neighbors = knn_classifier.num_neighbors

    plt.figure(figsize=(misclassified, num_neighbors+1))
    for i, misclassified_tuple in enumerate(misclassified_images):
        misclassified_idx, neighbors_idx = misclassified_tuple
        image = np.reshape(test_images[misclassified_idx], (28,28))
        plt.subplot(misclassified, num_neighbors + 1, i * (num_neighbors+1) + 1)
        plt.axis('off')
        plt.imshow(image, cmap="gray")

        for j, neighbor_idx in enumerate(neighbors_idx):
            image = np.reshape(knn_classifier.train_images[neighbor_idx], (28, 28))
            plt.subplot(misclassified, num_neighbors + 1, i * (num_neighbors+1) + j + 2)
            plt.axis('off')
            plt.imshow(image, cmap="gray")

    plt.show()

    for distance in ['l1', 'l2']:
        metric = KnnClassifier.Metric.euclidian_distance if distance == 'l2' else KnnClassifier.Metric.manhattan_distance
        accuracies = np.zeros((5,2))
        for i, num_neighbors in enumerate([1, 3, 5, 7, 9]):
            acc = knn_classifier.get_accuracy(test_images, test_labels, num_neighbors=num_neighbors, metric=metric)
            accuracies[i, 0] = num_neighbors
            accuracies[i, 1] = acc
            np.savetxt(f"accuracy_{distance}.txt", accuracies)
        plt.plot(accuracies[:, 0], accuracies[:, 1])

    plt.legend(['distance l1', 'distance l2'])
    plt.suptitle("Accuracy benchmark for different values of k and different distances")
    plt.show()