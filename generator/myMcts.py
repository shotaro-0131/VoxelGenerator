import time
import math
import random
import numpy as np
from mcts import mcts, treeNode, randomPolicy

def myPolicy(state):
    while not state.isTerminal():
        try:
            # action = random.choice(state.getPossibleActions())
            possibleActions = state.getPossibleActions()
            action = random.choices(possibleActions, weights=state.next_probs)[0]
        except IndexError:
            raise Exception("Non-terminal state has no possible actions: " + str(state))
        state = state.takeAction(action)
    return state.getReward()

class MyMcts(mcts):
    def __init__(self, timeLimit=None, iterationLimit=None, explorationConstant=1 / math.sqrt(2),
                 use_knowledge=False):
        if use_knowledge:
            super(MyMcts, self).__init__(timeLimit, iterationLimit, explorationConstant, myPolicy)
        else:
            super(MyMcts, self).__init__(timeLimit, iterationLimit, explorationConstant, randomPolicy)
        # self.Q = np.zeros((3, 32, 32, 32))
        # self.N = np.zeros((3, 32, 32, 32))

    def search(self, initialState, needDetails=False):
        self.root = treeNode(initialState, None)

        if self.limitType == 'time':
            timeLimit = time.time() + self.timeLimit / 1000
            while time.time() < timeLimit:
                self.executeRound()
        else:
            for i in range(self.searchLimit):
                self.executeRound()

        bestChild = self.getBestChild(self.root, 0)
        action=(action for action, node in self.root.children.items() if node is bestChild).__next__()
        if needDetails:
            return {"action": action, "expectedReward": bestChild.totalReward / bestChild.numVisits}
        else:
            return action

    def executeRound(self):
        """
            execute a selection-expansion-simulation-backpropagation round
        """
        node = self.selectNode(self.root)
        reward = self.rollout(node.state)
        # for (t, x, y, z) in node.state.state_index:
        #     self.Q[t, x, y, z] += reward
        #     self.N[t, x, y, z] += 1.0
        self.backpropogate(node, reward)

    def selectNode(self, node):
        while not node.isTerminal:
            if node.isFullyExpanded:
                node = self.getBestChild(node, self.explorationConstant)
            else:
                return self.expand(node)
        return node

    def expand(self, node):
        actions = node.state.getPossibleActions()
        for action in actions:
            if action not in node.children:
                newNode = treeNode(node.state.takeAction(action), node)
                node.children[action] = newNode
                if len(actions) == len(node.children):
                    node.isFullyExpanded = True
                return newNode

        raise Exception("Should never reach here")

    def getBestChild(self, node, explorationValue):
        bestValue = float("-inf")
        bestNodes = []
        for child in node.children.values():
            nodeValue = child.totalReward / child.numVisits + explorationValue * math.sqrt(
                2 * math.log(node.numVisits) / child.numVisits)
            # if self.Q[child.state.state_index[-1][0],child.state.state_index[-1][1],child.state.state_index[-1][2],child.state.state_index[-1][3]] != 0:
            #     nodeValue += self.Q[child.state.state_index[-1][0],child.state.state_index[-1][1],child.state.state_index[-1][2],child.state.state_index[-1][3]]/self.N[child.state.state_index[-1][0],child.state.state_index[-1][1],child.state.state_index[-1][2],child.state.state_index[-1][3]]
            if nodeValue > bestValue:
                bestValue = nodeValue
                bestNodes = [child]
            elif nodeValue == bestValue:
                bestNodes.append(child)
        return random.choice(bestNodes)

