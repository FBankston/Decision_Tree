# For the log calculation in entropy
import math

# Example from sci-kit learn developer documentation about building your own decision tree algorithm
# https://scikit-learn.org/stable/developers/develop.html#rolling-your-own-estimator
# Example from sci-kit learn decision tree branch on github
# https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/tree/_classes.py
# Both were used as a reference to give me a basic idea of building my own decision tree algorithm from scratch


def is_quantitative(value):
    # This function is simply to test and return whether or not values are categorical/quantitative
    # Boolean that returns 1 if instance is quantitative and 0 if instance is categorical
    return isinstance(value, int) or isinstance(value, float)


def unique_values(rows, col):
    # Here we attempt to find all of the unique values for each column in the dataset
    return set([row[col] for row in rows])


def decision_counter(rows):
    # The goal of this method is to count and store the amount of each decision from the data
    counts = {}
    for row in rows:
        # In a dataset, the decision should be the last column for clarity purposes
        # Therefore we can find decision column by row-1
        decision = row[-1]
        if decision not in counts:
            counts[decision] = 0
        counts[decision] += 1
    return counts


class Inquiry:
    # Inquiries are made to split or partition the dataset and find meaningful differences in the data
    # This class stores a column number and the feature value there
    # Then the feature value from an example is then compared to the feature value saved in the inquiry
    # The scikit-learn developer documentation was very helpful
    # for getting an idea of how to do a class definition like this

    def __init__(self, column, value, header):
        self.column = column
        self.value = value
        self.header = header

    def compare(self, example):
        currentValue = example[self.column]
        if is_quantitative(currentValue):
            return currentValue >= self.value
        else:
            return currentValue == self.value

    def __repr__(self):
        # This block simply prints out the inquiry in a readable format for the user
        condition = "=="
        if is_quantitative(self.value):
            condition = ">="
        return "%s %s %s?" % (
            self.header[self.column], condition, str(self.value))


def partition(rows, inquiry):
    # This method partitions the dataset
    # Each feature is checked to see if it matches the current inquiry
    # If it does, it will be added to matching rows
    # If not, add to different rows
    matching_rows, different_rows = [], []
    for row in rows:
        if inquiry.compare(row):
            matching_rows.append(row)
        else:
            different_rows.append(row)
    return matching_rows, different_rows


def entropy(rows):
    # Simple entropy calculation for a list of rows
    entries = decision_counter(rows)
    avg_entropy = 0
    size = float(len(rows))
    for decision in entries:
        prob = entries[decision] / size
        avg_entropy = avg_entropy + (prob * math.log(prob, 2))
    return -1*avg_entropy


def gini_impurity(rows):
    # Simple gini impurity calculation for a list of rows
    counts = decision_counter(rows)
    impurity = 1
    for dec in counts:
        prob_of_dec = counts[dec] / float(len(rows))
        impurity -= prob_of_dec**2
    return impurity


def information_gain(left, right, current_entropy):
    # Customization Option: Simple information gain formula that the user can customize
    p = float(len(left)) / (len(left) + len(right))
    return current_entropy - p * entropy(left) - (1 - p) * entropy(right)
    # return current_entropy - p * gini(left) - (1 - p) * gini(right)


def calculate_best_splitter(rows, header):
    # The method of finding the best inquiry to split the tree on
    # Here we want to keep track of the best gain from a question
    best_gain = 0
    # We also want to keep track of the inquiry that gave the best gain
    best_inquiry = None
    current_entropy = entropy(rows)
    # The total number of columns
    num_features = len(rows[0]) - 1

    # For each feature in the dataset
    for col in range(num_features):

        # Only counting unique values from each row
        values = set([row[col] for row in rows])

        # For each value
        for currentValue in values:

            inquiry = Inquiry(col, currentValue, header)

            # Attempt to make a split in the dataset
            matching_rows, different_rows = partition(rows, inquiry)

            # Skip if the split does not actually divide the dataset
            if len(matching_rows) == 0 or len(different_rows) == 0:
                continue

            # Calculate the information gain from this split
            gain = information_gain(matching_rows, different_rows, current_entropy)

            if gain >= best_gain:
                best_gain, best_inquiry = gain, inquiry

    # Return the best feature to split on, along with its gain
    return best_gain, best_inquiry


def print_decision(index):
    # This is just a simple method to print the decision of a leaf in a readable string format
    max_count = 0
    decision = ""

    # Ensuring we have the correct decision value
    for selection, value in index.items():
        if index[selection] > max_count:
            max_count = index[selection]
            decision = selection

    return decision


class decision_node:
    # A decision node will hold the reference to an inquiry
    # A decision node also has true/false branches to the child nodes based on the answer to the inquiry
    # The scikit-learn developer documentation was very helpful
    # for getting an idea of how to do a class definition like this

    def __init__(self,
                 inquiry,
                 true_branch,
                 false_branch,
                 depth,
                 id,
                 rows):
        self.inquiry = inquiry
        self.true_branch = true_branch
        self.false_branch = false_branch
        self.depth = depth
        self.id = id
        self.rows = rows


class Leaf:
    # A leaf node is where the classification is actually made

    def __init__(self, rows, id, depth):
        self.predictions = decision_counter(rows)
        self.predicted_decision = print_decision(self.predictions)
        self.id = id
        self.depth = depth


def build_decision_tree(rows, header, depth=0, id=0):
    # Start by partitioning the dataset between unique features and calculating information gain
    # Then return the inquiry that will result in best information gain

    gain, inquiry = calculate_best_splitter(rows, header)

    # If there is no more possible gain and no more inquiries, then we can make this a leaf
    if gain == 0:
        return Leaf(rows, id, depth)

    # If we escape the if statement and make it to this line, it means we have found a useful feature to split on
    matching_rows, different_rows = partition(rows, inquiry)

    # Begin building of the true branch through recursion
    true_branch = build_decision_tree(matching_rows, header, depth + 1, 2 * id + 2)

    # Begin building of the false branch through recursion
    false_branch = build_decision_tree(different_rows, header, depth + 1, 2 * id + 1)

    # Finally we will return the decision node
    # This will store the best feature along with its value, and the true/false branches as well
    return decision_node(inquiry, true_branch, false_branch, depth, id, rows)


def prune_tree(node, prunedList):
    # Method for building our decision tree
    # Customization option: Originally I wanted to allow the user to pick between pre-pruning and post-pruning
    # Instead I decided to print the tree before pruning and after pruning, and compare the accuracy between the two

    # If we reach a leaf node, we will simply return the leaf
    if isinstance(node, Leaf):
        return node
    # If we instead arrive at a pruned node, then that node should be made a leaf and returned
    # Since this is a leaf node, then the nodes below it should not be considered for the tree
    if int(node.id) in prunedList:
        return Leaf(node.rows, node.id, node.depth)

    # Call this function recursively on the true branch
    node.true_branch = prune_tree(node.true_branch, prunedList)

    # Call this function recursively on the false branch
    node.false_branch = prune_tree(node.false_branch, prunedList)

    return node


def classify(row, node):
    # Classifications are made at the leaf node
    # Here we are classifying, so we should return the prediction
    if isinstance(node, Leaf):
        return node.predicted_decision

    # Here we are deciding whether we should follow the true or false branch
    # We start by comparing the feature values stored in the node to the current example
    if node.inquiry.compare(row):
        return classify(row, node.true_branch)
    else:
        return classify(row, node.false_branch)


def print_tree(node, spacing=""):

    # Once again, a case where we have reached a leaf
    # Since we are printing the tree this time, we want to print out the leaf along with its prediction
    if isinstance(node, Leaf):
        print(spacing + "Leaf id: " + str(node.id) + " Predictions: " + str(node.predictions)
              + " decision Class: " + str(node.predicted_decision))
        # Using return as a "break" statement here to exit the if statement
        return

    # Print the inquiry being made at this node
    # Customization Option: To print the decision node's id and depth along with it, use the following commented option
    #print(spacing + str(node.inquiry) + " id: " + str(node.id) + " depth: " + str(node.depth))
    print(spacing + str(node.inquiry))

    # Call this function recursively on the true branch to build the tree
    print(spacing + 'True:')
    print_tree(node.true_branch, spacing + "  ")

    # Call this function recursively on the true branch to build the tree
    print(spacing + 'False:')
    print_tree(node.false_branch, spacing + "  ")


def getLeafNodes(node, leafNodes =[]):
    # Helper function to retrieve the leaf nodes to print them to the user
    # This part is optional in main.py
    if isinstance(node, Leaf):
        leafNodes.append(node)
        # Using return as a "break" statement here to exit the if statement
        return

    # Search for true values recursively
    getLeafNodes(node.true_branch, leafNodes)

    # Search for false values recursively
    getLeafNodes(node.false_branch, leafNodes)

    return leafNodes


def getInnerNodes(node, innerNodes =[]):

    # This time we have reached an inner node and are retrieving all the inner nodes to print to the user
    # This part is optional in main.py
    if isinstance(node, Leaf):
        return

    innerNodes.append(node)

    # Search for true values recursively
    getInnerNodes(node.true_branch, innerNodes)

    # Search for false values recursively
    getInnerNodes(node.false_branch, innerNodes)

    return innerNodes


def computeAcc(rows, node):
    # Simple function for calculating accuracy of the tree
    # Will be called before and after pruning to compare accuracy
    count = len(rows)
    if count == 0:
        return 0

    accuracy = 0
    for row in rows:
        # As stated above, the last column in a dataset should be the decision column, so we can find it by row-1
        if row[-1] == classify(row, node):
            accuracy += 1
    return round(accuracy/count, 2)
