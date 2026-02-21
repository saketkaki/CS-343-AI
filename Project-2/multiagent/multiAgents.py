# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
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
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
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

    def evaluationFunction(self, currentGameState: GameState, action):
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
        newFoodList = newFood.asList()
        smallestDistToFood = float('inf')
        for foods in newFoodList:
            distToFood = manhattanDistance(newPos, foods)
            if distToFood < smallestDistToFood:
                smallestDistToFood = distToFood
        
        smallestDistToGhost = float('inf')
        for ghosts in newGhostStates:
            distToGhost = manhattanDistance(newPos, ghosts.getPosition())
            if distToGhost < smallestDistToGhost:
                smallestDistToGhost = distToGhost
                smallestGhostIndex = newGhostStates.index(ghosts)

        finalval = smallestDistToFood
        if newScaredTimes[smallestGhostIndex]:
            finalval += smallestDistToGhost
            
        
        return successorGameState.getScore() + 1/finalval

def scoreEvaluationFunction(currentGameState: GameState):
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

    def getAction(self, gameState: GameState):
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

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        numberOfAgents = gameState.getNumAgents()
        maxDepth = self.depth * numberOfAgents

        def terminalCondtion(state, depth):
            return state.isWin() or state.isLose() or depth == 0
        
        def nextAgent(indexOfAgent):
            # Note: iterates through the number of agents and if the current agent is the last one, resets back to 0 which is pacman
            if indexOfAgent + 1 < numberOfAgents:
                return indexOfAgent + 1
            else:
                return 0
        
        def minimax(state, indexOfAgent, depth):
            if terminalCondtion(state, depth):
                return self.evaluationFunction(state), None  # returns a tuple of score and action
            
            legalActions = state.getLegalActions(indexOfAgent)
            next_agent = nextAgent(indexOfAgent)
            next_depth = depth - 1

            if indexOfAgent == 0:
                # This assumes that pacman is maximizing
                highestValue = float('-inf')
            else:
                # This assumes that ghosts are minimizing
                highestValue = float('inf')

            bestAction = None

            for action in legalActions:
                score, _ = minimax(state.generateSuccessor(indexOfAgent, action), next_agent, next_depth)
                if indexOfAgent == 0:
                    if score > highestValue:
                        highestValue = score
                        bestAction = action
                else:
                    if score < highestValue:
                        highestValue = score
                        bestAction = action

            return highestValue, bestAction
                
        selectedValue, selectedAction = minimax(gameState, 0, maxDepth)
        return selectedAction
        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        numberOfAgents = gameState.getNumAgents()
        maxDepth = self.depth

        def terminalCondtion(state, depth, agentIndex):
            if state.isWin() or state.isLose():
                return True 
            if depth == 0 and agentIndex == 0:
                return True
            return False
        
        def nextAgent(indexOfAgent):
            # Note: iterates through the number of agents and if the current agent is the last one, resets back to 0 which is pacman
            if indexOfAgent + 1 < numberOfAgents:
                return indexOfAgent + 1
            else:
                return 0
        
        def AlphaBeta(state, indexOfAgent, depth, alpha, beta):
            if terminalCondtion(state, depth, indexOfAgent):
                return self.evaluationFunction(state), None  # returns a tuple of score and action
            
            legalActions = state.getLegalActions(indexOfAgent)
            next_agent = nextAgent(indexOfAgent)
            
            if indexOfAgent == 0:
                next_depth = depth - 1  # Only decrease depth after Pacman moves
            else:
                next_depth = depth

            bestAction = None

            if indexOfAgent == 0:  # Pacman maximizes
                highestValue = float('-inf')
                for action in legalActions:
                    score, _ = AlphaBeta(state.generateSuccessor(indexOfAgent, action),
                                         next_agent, next_depth, alpha, beta)
                    if score > highestValue:
                        highestValue = score
                        bestAction = action
                    
                    # Prune remaining branches using updated alpha and beta
                    # Has to be strict inequality so we don't prune when values are equal
                    if highestValue > beta:
                        break
                    alpha = max(alpha, highestValue)
            else:  # Ghosts minimize
                highestValue = float('inf')
                for action in legalActions:
                    score, _ = AlphaBeta(state.generateSuccessor(indexOfAgent, action),
                                         next_agent, next_depth, alpha, beta)
                    if score < highestValue:
                        highestValue = score
                        bestAction = action
                    
                    # Prune remaining branches using updated alpha and beta
                    # Has to be strict inequality so we don't prune when values are equal
                    if highestValue < alpha:
                        break
                    beta = min(beta, highestValue)
            
            return highestValue, bestAction
                
        selectedValue, selectedAction = AlphaBeta(gameState, 0, maxDepth, float('-inf'), float('inf'))
        return selectedAction
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        numberOfAgents = gameState.getNumAgents()
        maxDepth = self.depth * numberOfAgents

        def terminalCondtion(state, depth):
            return state.isWin() or state.isLose() or depth == 0
        
        def nextAgent(indexOfAgent):
            # Note: iterates through the number of agents and if the current agent is the last one, resets back to 0 which is pacman
            if indexOfAgent + 1 < numberOfAgents:
                return indexOfAgent + 1
            else:
                return 0
        
        def expectimax(state, indexOfAgent, depth):
            if terminalCondtion(state, depth):
                return self.evaluationFunction(state), None  # returns a tuple of score and action
            
            legalActions = state.getLegalActions(indexOfAgent)
            next_agent = nextAgent(indexOfAgent)
            next_depth = depth - 1

            if indexOfAgent == 0:
                # This assumes that pacman is maximizing
                highestValue = float('-inf')
            else:
                # This assumes that ghosts are expected value nodes
                highestValue = 0
                # Since ghosts act uniformly at random, each legal action has equal probability
                probability = 1.0 / len(legalActions) 

            bestAction = None

            for action in legalActions:
                score, _ = expectimax(state.generateSuccessor(indexOfAgent, action), next_agent, next_depth)
                if indexOfAgent == 0:
                    if score > highestValue:
                        highestValue = score
                        bestAction = action
                else:
                    # Instead of taking the minimum (as in Minimax), we compute the expected value
                    highestValue += probability * score

            return highestValue, bestAction
                
        selectedValue, selectedAction = expectimax(gameState, 0, maxDepth)
        return selectedAction
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION:  This function evaluates various states by adding the current game score to distances food, ghosts, and capsules.
                  The closer the food and capsules are, the higher the score.
                  The closer the ghost is, the lower the score (unless the ghost is scared, in which case it increases the score).
    """
    "*** YOUR CODE HERE ***"
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newFoodList = newFood.asList()
    capsules = currentGameState.getCapsules()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    currentScore = currentGameState.getScore()

    smallestDistToFood = float('inf')
    for foods in newFoodList:
        distToFood = manhattanDistance(newPos, foods)
        if distToFood < smallestDistToFood:
            smallestDistToFood = distToFood

    foodscore = 1.0 / smallestDistToFood

    smallestDistToGhost = float('inf')
    for ghosts in newGhostStates:
        distToGhost = manhattanDistance(newPos, ghosts.getPosition())
        if distToGhost < smallestDistToGhost:
            smallestDistToGhost = distToGhost
            smallestGhostIndex = newGhostStates.index(ghosts)

    ghostScore = 0

    # If the closest ghost is scared then pacman should chase it (rewarding it)
    if newScaredTimes[smallestGhostIndex] > 0:
        ghostScore += 1 / smallestDistToGhost
    else:
        # If ghost is not scared then pacman should run away from it (penalizing it)
        if smallestDistToGhost <= 1:
            return -float('inf')
        ghostScore -= 1 / smallestDistToGhost

    # If capsule is close then pacman should go to it (rewarding it)    
    capsuleScore = 0
    closestCapsuleDist = float('inf')
    for capsule in capsules:
        dist = manhattanDistance(newPos, capsule)
        if dist < closestCapsuleDist:
            closestCapsuleDist = dist

    capsuleScore = 1.0 / closestCapsuleDist

    # Weighted sum to prioritize food, ghosts, and capsules
    return currentScore + 10*foodscore + 150*ghostScore + 20*capsuleScore
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
