# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for 
# educational purposes provided that (1) you do not distribute or publish 
# solutions, (2) you retain this notice, and (3) you provide clear 
# attribution to UC Berkeley, including a link to 
# http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero 
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and 
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        ghostDists = []
        foodDists = []
        for ghostState in newGhostStates:
          if ghostState.scaredTimer == 0:
            ghostDists.append(manhattanDistance(newPos, ghostState.getPosition()))

        ghostDists.sort()

        for food in newFood.asList():
            foodDists.append(manhattanDistance(newPos, food))

        foodDists.sort()

        if len(foodDists) > 0:
          closestFood = foodDists[0]
        else:
          closestFood = 0

        numNewFood = successorGameState.getNumFood()

        heuristic = 0
        for ghost in newGhostStates:
          ghostDist = manhattanDistance(newPos, ghost.getPosition())
          if ghost.scaredTimer > ghostDist:
            heuristic += ghost.scaredTimer - ghostDist

        # stay away from the nearest ghost if it isn't scared.

        if len(ghostDists) > 0:
          heuristic += ghostDists[0]

        heuristic = heuristic - 20*numNewFood - closestFood
        return heuristic


def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """
    def maximize(self, gameState, depth, agentIndex):
      max = float("-inf")
      if gameState.isWin() or gameState.isLose():
        return self.evaluationFunction(gameState)
      for action in gameState.getLegalActions(0):
        successor = gameState.generateSuccessor(0, action)

        # run minimize (the minimize function will stack ghost responses)
        temp = self.minimize(successor, depth, 1)
        if temp > max:
          max = temp
          maxAction = action

      # if this is the first depth, then return an ACTION to take. otherwise, return a number.
      if depth == 1:
        return maxAction
      else:
        return max


    def minimize(self, gameState, depth, agentIndex):
      min= float("inf")
      numAgents = gameState.getNumAgents()
      if gameState.isWin() or gameState.isLose():
        return self.evaluationFunction(gameState)
      for action in gameState.getLegalActions(agentIndex):
        successor = gameState.generateSuccessor(agentIndex, action)
        # for last ghost
        if agentIndex == numAgents - 1:
          # depth limit
          if depth == self.depth:
            temp = self.evaluationFunction(successor)
          else:
            #pacman's turn so maximize.
            temp = self.maximize(successor, depth+1, 0)
        # minimize for ghost.
        else:
          temp = self.minimize(successor, depth, agentIndex+1)

        if temp < min:
          min = temp
          minAction = action

      return min


    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        maxAction = self.maximize(gameState, 1, 0)
        return maxAction
        #util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """
    def prune_max(self, gameState, depth, agentIndex, alpha, beta):
      # init the variables
      maximum = float("-inf")

      #return the evaluation function at this state if its a leaf node.
      if gameState.isWin() or gameState.isLose():
        return self.evaluationFunction(gameState)

      for action in gameState.getLegalActions(0):
        successor = gameState.generateSuccessor(0, action)

        # run minimize (the minimize function will stack ghost responses)
        temp = self.prune_min(successor, depth, 1, alpha, beta)

        #prune
        if temp > beta:
          return temp

        if temp > maximum:
          maximum = temp
          maxAction = action

        #reassign alpha
        alpha = max(alpha, maximum)

      # if first depth return action otherwise return number.
      if depth == 1:
        return maxAction
      else:
        return maximum



    def prune_min(self, gameState, depth, agentIndex, alpha, beta):
      minEval= float("inf")

      # we don't know how many ghosts there are, so we have to run minimize
      # on a general case based off the number of agents
      numAgents = gameState.getNumAgents()

      # if a leaf node, return the eval function!
      if gameState.isWin() or gameState.isLose():
        return self.evaluationFunction(gameState)

      # for every move possible by this ghost
      for action in gameState.getLegalActions(agentIndex):
        successor = gameState.generateSuccessor(agentIndex, action)

        # if this is the last ghost to minimize
        if agentIndex == numAgents - 1:
          # if we are at our depth limit, return the eval function
          if depth == self.depth:
            tempEval = self.evaluationFunction(successor)
          else:
            #maximize!
            tempEval = self.prune_max(successor, depth+1, 0, alpha, beta)

        # pass this state on to the next ghost
        else:
          tempEval = self.prune_min(successor, depth, agentIndex+1, alpha, beta)

        #prune
        if tempEval < alpha:
          return tempEval
        if tempEval < minEval:
          minEval = tempEval
          minAction = action

        # new beta
        beta = min(beta, minEval)
      return minEval
    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        maxAction = self.prune_max(gameState, 1, 0, float("-inf"), float("inf"))
        return maxAction
        #util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    def maximize(self, gameState, depth, agentIndex):
      maximum = float("-inf")
      if gameState.isWin() or gameState.isLose():
        return self.evaluationFunction(gameState)


      for action in gameState.getLegalActions(0):
        successor = gameState.generateSuccessor(0, action)


        temp = self.minimize(successor, depth, 1)
        if temp > maximum:
            maximum = temp
            maxAction = action

      # if first depth return action otherwise return number.
      if depth == 1:
        return maxAction
      else:
        return maximum



    def minimize(self, gameState, depth, agentIndex):

      # we will add to this evaluation based on an even weighting of each action.
      minimum = 0
      numAgents = gameState.getNumAgents()

      if gameState.isWin() or gameState.isLose():
        return self.evaluationFunction(gameState)
      legalActions = gameState.getLegalActions(agentIndex)
      # weighting for each minimize action even distribution over the legal moves.
      prob = 1.0/len(legalActions)
      for action in legalActions:
        successor = gameState.generateSuccessor(agentIndex, action)
        # if last ghost
        if agentIndex == numAgents - 1:
          # if at depth limit
          if depth == self.depth:
            temp = self.evaluationFunction(successor)
          else:
            #maximize!
            temp = self.maximize(successor, depth + 1, 0)
        # we have to minimize with another ghost still.
        else:
          temp = self.minimize(successor, depth, agentIndex + 1)

        #add temp with probability to total
        minimum += temp * prob

      return minimum

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        return self.maximize(gameState, 1, 0)
        #util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: points are subtracted per food that pacman doesn't eat, this is done to encourage him to eat
      everything. Points are also deducted for getting eaten by a ghost but at the same time, pacman is given an
      incentive using the heuristic to eat the ghost if they are in scared state and this results in 200 points
      minus the distance to that ghost.

      We start with the current score and use the above criteria to deduct or add points using the evaluation function.

    """
    "*** YOUR CODE HERE ***"
    ghostStates = currentGameState.getGhostStates()
    position = currentGameState.getPacmanPosition()
    food_count = currentGameState.getNumFood()
    score = currentGameState.getScore()

    nearestGhostDistance = float("inf")

    # get current state of the ghosts.
    ghosts = 0
    for ghost in ghostStates:
      ghostPosition = (int(ghost.getPosition()[0]), int(ghost.getPosition()[1]))
      md = manhattanDistance(position, ghostPosition)

      if ghost.scaredTimer == 0:
        if md < nearestGhostDistance:
          nearestGhostDistance = md
      #scared ghosts = 200 - the distance they are away
      elif ghost.scaredTimer > md:
        ghosts += 200 - md

    if nearestGhostDistance == float("inf"):
      nearestGhostDistance = 0

    ghosts += nearestGhostDistance

    score = score - 10 * food_count +1 * ghosts
    return score
    #util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

