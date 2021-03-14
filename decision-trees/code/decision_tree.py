import numpy as np
import math

class Tree():
    def __init__(self, value=None, attribute_name="root", attribute_index=None, branches=None):
        """
        This class implements a tree structure with multiple branches at each node.
        If self.branches is an empty list, this is a leaf node and what is contained in
        self.value is the predicted class.

        The defaults for this are for a root node in the tree.

        Arguments:
            branches (list): List of Tree classes. Used to traverse the tree. In a
                binary decision tree, the length of this list is either 2 (for left and
                right branches) or 0 (at a leaf node).
            attribute_name (str): Contains name of attribute that the tree splits the data
                on. Used for visualization (see `DecisionTree.visualize`).
            attribute_index (float): Contains the  index of the feature vector for the
                given attribute. Should match with self.attribute_name.
            value (number): Contains the value that data should be compared to along the
                given attribute.
        """
        self.branches = [] if branches is None else branches
        self.attribute_name = attribute_name
        self.attribute_index = attribute_index
        self.value = value
    
    def __repr__(self):
        return f"A tree with title {self.attribute_name}"

class DecisionTree():
    def __init__(self, attribute_names):
        """
        TODO: Implement this class.

        This class implements a binary decision tree learner for examples with
        categorical attributes. Use the ID3 algorithm for implementing the Decision
        Tree: https://en.wikipedia.org/wiki/ID3_algorithm

        A decision tree is a machine learning model that fits data with a tree
        structure. Each branching point along the tree marks a decision (e.g.
        today is sunny or today is not sunny). Data is filtered by the value of
        each attribute to the next level of the tree. At the next level, the process
        starts again with the remaining attributes, recursing on the filtered data.

        Which attributes to split on at each point in the tree are decided by the
        information gain of a specific attribute.

        Here, you will implement a binary decision tree that uses the ID3 algorithm.
        Your decision tree will be contained in `self.tree`, which consists of
        nested Tree classes (see above).

        Args:
            attribute_names (list): list of strings containing the attribute names for
                each feature (e.g. chocolatey, good_grades, etc.)

        """
        self.attribute_names = attribute_names
        self.tree = None

    def _check_input(self, features):
        if features.shape[1] != len(self.attribute_names):
            raise ValueError(
                "Number of features and number of attribute names must match!"
            )

    def fit(self, features, targets):
        """
        Takes in the features as a numpy array and fits a decision tree to the targets.

        Args:
            features (np.array): numpy array of size NxF containing features, where N is
                number of examples and F is number of features.
            targets (np.array): numpy array containing class labels for each of the N
                examples.
        """
        self._check_input(features)

        def findBestSplit(feats, targs):
            attribute_count = feats.shape[1]
            minSplit = 4000
            minSplitIndex = -1
            for attrib in range(attribute_count):
                this_info_gain = information_gain(feats, attrib, targs)
                #print(this_info_gain, "\n")
                if this_info_gain < minSplit:
                    minSplit = this_info_gain
                    minSplitIndex = attrib
            return minSplitIndex, minSplit
        """
        def dtl(featz, targz, default="forty-two"):
            if featz.shape[0] == 0 and featz.shape[1] == 0:
                return default
            elif 1 not in targz:
                print("1 not in targz! returning 0")
                return 0
            elif 0 not in targz:
                print("0 not in targz! returning 1")
                return 1
            elif targz.shape[0] == 0:
                print("length of targz was 0")
                return "length of targz was 0"
            else:
                self.tree = Tree(
                    attribute_name="root",
                    attribute_index=-1,
                    value=0,
                    branches=[]
                )
                while featz.shape[1] != 0:
                    bestSplitIndex, bestSplit = findBestSplit(featz, targz)
                    self.tree.branches.append(
                        Tree(
                            attribute_name=self.attribute_names[bestSplitIndex],
                            attribute_index=bestSplitIndex,
                            value=bestSplit,
                            branches=[]
                        )
                    )
                    featz = np.delete(featz, np.s_[bestSplitIndex:(bestSplitIndex+1)], 1)
                    self.visualize()
                    print(self.tree.branches, "\n")
                return self.tree
        """
        def dtl2(featz, targz, default="forty-two"):
            root_node = Tree(
                    attribute_name="root",
                    attribute_index=-1,
                    value=-1,
                    branches=[]
            )
            if 0 not in targz:
                return Tree(
                    attribute_name="root",
                    attribute_index=1,
                    value=1,
                    branches=[]
                )
            elif 1 not in targz:
                return Tree(
                    attribute_name="root",
                    attribute_index=0,
                    value=0,
                    branches=[]
                )
            elif featz.shape[0] == 0 and featz.shape[1] == 0:
                return Tree(
                    attribute_name=default,
                    attribute_index=0,
                    value=0,
                    branches=[]
                )
            else:
                bestSplitIndex, bestSplit = findBestSplit(featz, targz)
                tr33 = Tree(
                    attribute_name=self.attribute_names[bestSplitIndex],
                    attribute_index=bestSplitIndex,
                    value=bestSplit,
                    branches=[]
                )
                for vi in range(2):
                    # two possible outcomes, so two iterations here
                    guy = Tree(
                            attribute_name=self.attribute_names[bestSplitIndex],
                            attribute_index=bestSplitIndex,
                            value=vi,
                            branches=[]
                        )
                    examples_vi = featz[np.where(featz[bestSplitIndex]==vi)]
                    print(examples_vi)
                    if examples_vi.shape[0] < 1:
                        guy.branches.append(
                            Tree(
                                attribute_name="leaf",
                                attribute_index=0,
                                value=vi,
                                branches=[]
                            )
                        )
                    else:
                        guy.branches.append(dtl2(np.delete(featz, np.s_[bestSplitIndex:(bestSplitIndex+1)], 1), targz))
                    tr33.branches.append(guy)
                    return tr33
        
        self.tree = dtl2(features, targets)
        self.visualize()
        return self.tree

    def predict(self, features):
        """
        Takes in features as a numpy array and predicts classes for each point using
        the trained model.

        Args:
            features (np.array): numpy array of size NxF containing features, where N is
                number of examples and F is number of features.
        """
        self._check_input(features)
        return features

    def _visualize_helper(self, tree, level):
        """
        Helper function for visualize a decision tree at a given level of recursion.
        """
        tab_level = "  " * level
        val = tree.value if tree.value is not None else 0
        print("%d: %s%s == %f" % (level, tab_level, tree.attribute_name, val))

    def visualize(self, branch=None, level=0):
        """
        Visualization of a decision tree. Implemented for you to check your work and to
        use as an example of how to use the given classes to implement your decision
        tree.
        """
        if not branch:
            branch = self.tree
        self._visualize_helper(branch, level)

        for branch in branch.branches:
            self.visualize(branch, level+1)

## end of DecisionTree class

def information_gain(features, attribute_index, targets):
    """
    TODO: Implement me!

    Information gain is how a decision tree makes decisions on how to create
    split points in the tree. Information gain is measured in terms of entropy.
    The goal of a decision tree is to decrease entropy at each split point as much as
    possible. This function should work perfectly or your decision tree will not work
    properly.

    Information gain is a central concept in many machine learning algorithms. In
    decision trees, it captures how effective splitting the tree on a specific attribute
    will be for the goal of classifying the training data correctly. Consider
    data points S and an attribute A. S is split into two data points given binary A:

        S(A == 0) and S(A == 1)

    Together, the two subsets make up S. If A was an attribute perfectly correlated with
    the class of each data point in S, then all points in a given subset will have the
    same class. Clearly, in this case, we want something that captures that A is a good
    attribute to use in the decision tree. This something is information gain. Formally:

        IG(S,A) = H(S) - H(S|A)

    where H is information entropy. Recall that entropy captures how orderly or chaotic
    a system is. A system that is very chaotic will evenly distribute probabilities to
    all outcomes (e.g. 50% chance of class 0, 50% chance of class 1). Machine learning
    algorithms work to decrease entropy, as that is the only way to make predictions
    that are accurate on testing data. Formally, H is defined as:

        H(S) = sum_{c in (classes in S)} -p(c) * log_2 p(c)

    To elaborate: for each class in S, you compute its prior probability p(c):

        (# of elements of class c in S) / (total # of elements in S)

    Then you compute the term for this class:

        -p(c) * log_2 p(c)

    Then compute the sum across all classes. The final number is the entropy. To gain
    more intution about entropy, consider the following - what does H(S) = 0 tell you
    about S?

    Information gain is an extension of entropy. The equation for information gain
    involves comparing the entropy of the set and the entropy of the set when conditioned
    on selecting for a single attribute (e.g. S(A == 0)).

    For more details: https://en.wikipedia.org/wiki/ID3_algorithm#The_ID3_metrics

    Args:
        features (np.array): numpy array containing features for each example.
        attribute_index (int): which column of features to take when computing the
            information gain
        targets (np.array): numpy array containing labels corresponding to each example.

    Output:
        information_gain (float): information gain if the features were split on the
            attribute_index.
    """

    ## important: class == possible outcome

    ## you have to calculate how many outcomes there are of each type
    ## for each possible outcome (i.e., class)

    this_attributes_example_slice = features[:,attribute_index]
    total_number_of_examples = features.shape[0]

    zero_count = 0
    one_count = 0

    outcomes_with_target_0 = [0, 0]
    outcomes_with_target_1 = [0, 0]
    
    for itc, this_attribute in enumerate(this_attributes_example_slice):
        # this loop is the real decision tree lol
        this_target = targets[itc]

        if this_target == 0:
            zero_count += 1
            if this_attribute == 0:
                outcomes_with_target_0[0] += 1
            else:
                outcomes_with_target_0[1] += 1
        elif this_target == 1:
            one_count += 1
            if this_attribute == 0:
                outcomes_with_target_1[0] += 1
            else:
                outcomes_with_target_1[1] += 1
        else:
            raise AssertionError("What the actual fuck")

    #print(outcomes_with_target_0, outcomes_with_target_1)
    #print(zero_count, one_count)

    prior_prob_class_0 = zero_count/total_number_of_examples
    prior_prob_class_1 = one_count/total_number_of_examples

    #print(f"pp_class_0: {prior_prob_class_0}; pp_class_1: {prior_prob_class_1}")

    # prior probabilities for class 0
    class0pp0 = outcomes_with_target_0[0]/(outcomes_with_target_0[0] + outcomes_with_target_0[1])
    class0pp1 = outcomes_with_target_0[1]/(outcomes_with_target_0[0] + outcomes_with_target_0[1])
    
    # prior probabilities for class 1
    class1pp0 = outcomes_with_target_1[0]/(outcomes_with_target_1[0] + outcomes_with_target_1[1]) # 2/6 analog
    class1pp1 = outcomes_with_target_1[1]/(outcomes_with_target_1[0] + outcomes_with_target_1[1]) # 4/6 analog

    class_0_entropy_term = prior_prob_class_0*(-class0pp0*math.log(class0pp0,2)-class0pp1*math.log(class0pp1,2))
    class_1_entropy_term = prior_prob_class_1*(-class1pp0*math.log(class1pp0,2)-class1pp1*math.log(class1pp1,2))

    return class_0_entropy_term + class_1_entropy_term

"""
if __name__ == '__main__':
    # construct a fake tree
    attribute_names = ['larry', 'curly', 'moe']
    decision_tree = DecisionTree(attribute_names=attribute_names)
    while len(attribute_names) > 0:
        attribute_name = attribute_names[0]
        if not decision_tree.tree:
            decision_tree.tree = Tree(
                attribute_name=attribute_name,
                attribute_index=decision_tree.attribute_names.index(attribute_name),
                value=0,
                branches=[]
            )
        else:
            decision_tree.tree.branches.append(
                Tree(
                    attribute_name=attribute_name,
                    attribute_index=decision_tree.attribute_names.index(attribute_name),
                    value=0,
                    branches=[]
                )
            )
        attribute_names.remove(attribute_name)
    decision_tree.visualize()
"""

if __name__=="__main__":
    from data import load_data
    np_lines, np_targets, labels = load_data("data/ivy-league.csv")
    my_tree = DecisionTree("GoodGrades,GoodLetters,GoodSAT,IsRich,HasScholarship,ParentAlum,SchoolActivities".split(","))
    print("Output of fit() is", my_tree.fit(np_lines, np_targets))