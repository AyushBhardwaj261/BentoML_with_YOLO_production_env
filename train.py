from sklearn import datasets, svm
import bentoml


# load training data set
iris = datasets.load_iris()
X,y = iris.data,iris.target

# Train the model
clf = svm.SVC(gamma='scale')
clf.fit(X,y)

# save model to the BentoML local model store
save_model = bentoml.sklearn.save_model("iris_clf",clf)
print(f"Model Saved: {save_model}")

## iris_clf:rniin3fuk6k7ieb5

