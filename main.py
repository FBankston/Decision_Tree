import pandas as pd
from sklearn import model_selection
from myDecisionTree import *
from datetime import datetime

# Begin counting execution time
start_time = datetime.now()
# Choose your dataset here
#df = pd.read_csv('data_set/breast_cancer.csv')
df = pd.read_csv('data_sets/car_acceptability.csv')
header = list(df.columns)

listValues = df.values.tolist()

# Here we will split the dataset for cross-validation
# scikit-learn is used for the cross-validation but is not used for any other purposes
trainingSet, testingSet = model_selection.train_test_split(listValues, test_size=0.2)

# Begin building the tree
decisionTree = build_decision_tree(trainingSet, header)

# Customization Option: Retrieve all of the inner nodes/leaf nodes to print them to the user
# This part can be removed/commented out if the user only wants to see the tree and the pruning process
print("\nLeaf nodes:")
leafNodes = getLeafNodes(decisionTree)
for leaf in leafNodes:
    # Using a simple for loop to retrieve all the leaf nodes and print to the user
    print("id = " + str(leaf.id) + " depth =" + str(leaf.depth))

print("\nNon-leaf nodes:")
innerNodes = getInnerNodes(decisionTree)

for inner in innerNodes:
    # Using a simple for loop to retrieve all the non-leaf nodes and print to the user
    print("id = " + str(inner.id) + " depth =" + str(inner.depth))

# Print the tree before pruning
maximumAcc = computeAcc(testingSet, decisionTree)
print("\nAccuracy without pruning " + str(maximumAcc*100) + "\n")
print_tree(decisionTree)

# Begin Post-Pruning the tree
nodeIdToPrune = -1
for node in innerNodes:
    if node.id != 0:
        prune_tree(decisionTree, [node.id])
        currentAcc = computeAcc(testingSet, decisionTree)
        # Print the node that is being pruned along with accuracy of the tree after pruning
        print("Pruned node: " + str(node.id) +
              " Resulting Accuracy: " + str(currentAcc*100) + "%")

        if currentAcc > maximumAcc:
            maximumAcc = currentAcc
            nodeIdToPrune = node.id
        decisionTree = build_decision_tree(trainingSet, header)
        # If the pruning reaches a point to where accuracy is 100% then stop immediately
        if maximumAcc == 1:
            break

# If nodeIdToPrune is not still equal to -1, then pruning increased accuracy of the tree
# Else, pruning did not increase accuracy and the tree should be kept as is
if nodeIdToPrune != -1:
    decisionTree = build_decision_tree(trainingSet, header)
    prune_tree(decisionTree, [nodeIdToPrune])
    print("\nPruning increased accuracy. Final Node ID pruned: "
          + str(nodeIdToPrune))
else:
    decisionTree = build_decision_tree(trainingSet, header)
    print("\nPruning the tree did not increase accuracy")

print("\nAccuracy: " + str(maximumAcc*100) + "%")
print_tree(decisionTree)
# End counting execution time and print to the user
end_time = datetime.now()
print('Finished in: {}'.format(end_time - start_time))
