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


from typing import Tuple
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
        score = 999
        foods = currentGameState.getFood().asList()
        dis = []
        for i in range(len(foods)):
            dis.append((manhattanDistance(foods[i], list(newPos))))
        score = -min(dis)
        for ghostState in newGhostStates:
            if newScaredTimes and ghostState.getPosition() == newPos:
                score = -999

        return score

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

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        #GetAction
        depth = self.depth
        action, score = self.MinimaxAgent(depth, 0, True, gameState)
        return action 


    """minimax code"""
    def MinimaxAgent(self, depth, agentIndex, maximazingPlayer, gameState):

        bestaction,bestscore,score = None, None, None #return the best scores action

        if depth == 0 or gameState.isWin() or gameState.isLose(): #at the end
            return None, self.evaluationFunction(gameState)

            
        if maximazingPlayer: #maximazin player
            scores = []
            actions = []
            for action in gameState.getLegalActions(agentIndex):
                action2, score = self.MinimaxAgent(depth, agentIndex + 1, False, gameState.generateSuccessor(agentIndex, action))
                scores.append(score)
                actions.append(action)
            bestscore = max(scores)
            bestaction = actions[scores.index(bestscore)]
        else:
            scores = []
            actions = []
            for action in gameState.getLegalActions(agentIndex):
                if agentIndex >= gameState.getNumAgents() -1 : #all agents handled
                    action2, score = self.MinimaxAgent(depth - 1, 0, True, gameState.generateSuccessor(agentIndex, action))
                else:
                    action2, score = self.MinimaxAgent(depth, agentIndex + 1, False, gameState.generateSuccessor(agentIndex, action))
                scores.append(score)
                actions.append(action)
            bestscore = min(scores)
            bestaction = actions[scores.index(bestscore)]

        return bestaction, bestscore


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        inf = float('inf')
        depth = self.depth
        action, score = self.AlphaBetaAgent(depth, 0, True, gameState, -inf, inf)
        return action

    """AlphaBeta"""
    def AlphaBetaAgent(self, depth, agentIndex,maximazingPlayer, gameState, alpha, beta):
        bestaction,bestscore, score = None, None, None #return the best scores action

        if depth == 0 or gameState.isWin() or gameState.isLose(): #at the end
            return None, self.evaluationFunction(gameState)

            
        if maximazingPlayer: #maximazin player
            scores = []
            actions = []
            for action in gameState.getLegalActions(agentIndex):
                action2, score = self.AlphaBetaAgent(depth, agentIndex + 1, False, gameState.generateSuccessor(agentIndex, action), alpha, beta)
                if score > beta:
                    scores.append(score)
                    actions.append(action)
                    break
                alpha = max(alpha, score)
                scores.append(score)
                actions.append(action)
            bestscore = max(scores)
            bestaction = actions[scores.index(bestscore)]

        else:
            scores = []
            actions = []
            for action in gameState.getLegalActions(agentIndex):
                if agentIndex >= gameState.getNumAgents() -1 : #all agents handled
                    action2, score = self.AlphaBetaAgent(depth - 1, 0, True, gameState.generateSuccessor(agentIndex, action), alpha, beta)
                else:
                    action2, score = self.AlphaBetaAgent(depth, agentIndex + 1, False, gameState.generateSuccessor(agentIndex, action), alpha, beta)
                if score < alpha:
                    scores.append(score)
                    actions.append(action)
                    break
                beta = min(beta, score)
                scores.append(score)
                actions.append(action)
            bestscore = min(scores)
            bestaction = actions[scores.index(bestscore)]


        return bestaction, bestscore

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        depth = self.depth
        action, score = self.ExpectimaxAgent(depth, 0, True, gameState)  
        return action

    def ExpectimaxAgent(self, depth, agentIndex, maximazingPlayer, gameState):
                bestaction,bestscore,score = None, None, None #return the best scores action

                if depth == 0 or gameState.isWin() or gameState.isLose(): #at the end
                    return None, self.evaluationFunction(gameState)                
                if maximazingPlayer: #maximazin player
                    scores = []
                    actions = []
                    for action in gameState.getLegalActions(agentIndex):
                        action2, score = self.ExpectimaxAgent(depth, agentIndex + 1, False, gameState.generateSuccessor(agentIndex, action))
                        scores.append(score)
                        actions.append(action)
                    bestscore = max(scores)
                    bestaction = actions[scores.index(bestscore)]
                else: #this is the ghost so we will get prob here
                    scores = []
                    actions = []
                    bestscore = 0
                    #we will divide the certain from total number of moves from the ghost so we will get the posibility of the moves for example there is 4 legalaction prob of each of them is 0.25
                    prob = 1 / len(gameState.getLegalActions(agentIndex))
                    for action in gameState.getLegalActions(agentIndex):
                        if agentIndex >= gameState.getNumAgents() -1 : #all agents handled
                            action2, score = self.ExpectimaxAgent(depth - 1, 0, True, gameState.generateSuccessor(agentIndex, action))
                        else:
                            action2, score = self.ExpectimaxAgent(depth, agentIndex + 1, False, gameState.generateSuccessor(agentIndex, action))
                        scores.append(score)
                        actions.append(action)
                    bestscore += prob * sum(scores)
                    bestaction = None

                return bestaction, bestscore

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: we get the foods first depending on distance we will add this to our score and we also get the distance to the ghost distance depending on ghost distance we will also add it to our score
    """
    "*** YOUR CODE HERE ***"
        
    foods = currentGameState.getFood().asList()
    pos = currentGameState.getPacmanPosition()
    score = -1
    GhostDistance = 1
    for food in foods:
        dis = manhattanDistance(pos, food)
        if score > dis:
           score = dis
    for ghostState in currentGameState.getGhostPositions():
        dis2 = manhattanDistance(pos, ghostState)
        GhostDistance += dis2

    return currentGameState.getScore() + (1 / score) - (1 / GhostDistance) - len(currentGameState.getCapsules())

# Abbreviation
better = betterEvaluationFunction
