from collections import Counter
from main import *

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):

        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]

        k_indices = np.argsort(distances)[:self.k]

        k_nearest_labels = [self.y_train[i] for i in k_indices]

        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]


def k_fold_cross_validation(X, y, k, k_value):
    fold_size = len(X) // k
    indices = np.arange(len(X))
    np.random.shuffle(indices)

    accuracy_scores = []

    for i in range(k):
        start = i * fold_size
        end = start + fold_size

        X_test_fold = X[indices[start:end]]
        y_test_fold = y[indices[start:end]]

        X_train_fold = np.concatenate((X[indices[:start]], X[indices[end:]]), axis=0)
        y_train_fold = np.concatenate((y[indices[:start]], y[indices[end:]]), axis=0)

        knn = KNN(k=k_value)
        knn.fit(X_train_fold, y_train_fold)
        predictions = knn.predict(X_test_fold)

        accuracy = np.mean(predictions == y_test_fold)
        accuracy_scores.append(accuracy)

    return np.mean(accuracy_scores), accuracy_scores


# Convert the DataFrame to numpy arrays
X = df.drop(columns=['species']).to_numpy()
y = df['species'].to_numpy()

# Perform k-fold cross-validation
k = 5
k_value = 3
mean_accuracy, accuracies = k_fold_cross_validation(X, y, k, k_value)

print(f'K-Fold Cross-Validation Mean Accuracy: {mean_accuracy:.2f}')
print(f'Individual Fold Accuracies: {accuracies}')




#
# knn = KNN(k=3)
# knn.fit(X_train, y_train)
# predictions = knn.predict(X_test)
#
#
# accuracy = np.mean(predictions == y_test)
# print(f'Accuracy: {accuracy:.2f}')