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
        # Collect legal moves and child states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        The evaluation function takes in the current and proposed child
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.
        """
        # Useful information you can extract from a GameState (pacman.py)
        childGameState = currentGameState.getPacmanNextState(action)
        newPos = childGameState.getPacmanPosition()
        newFood = childGameState.getFood()
        newGhostStates = childGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        minGhostDistance = min([manhattanDistance(newPos, state.getPosition()) for state in newGhostStates])

        scoreDiff = childGameState.getScore() - currentGameState.getScore()

        pos = currentGameState.getPacmanPosition()
        nearestFoodDistance = min([manhattanDistance(pos, food) for food in currentGameState.getFood().asList()])
        newFoodsDistances = [manhattanDistance(newPos, food) for food in newFood.asList()]
        newNearestFoodDistance = 0 if not newFoodsDistances else min(newFoodsDistances)
        isFoodNearer = nearestFoodDistance - newNearestFoodDistance

        direction = currentGameState.getPacmanState().getDirection()
        if minGhostDistance <= 1 or action == Directions.STOP:
            return 0
        if scoreDiff > 0:
            return 8
        elif isFoodNearer > 0:
            return 4
        elif action == direction:
            return 2
        else:
            return 1


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
    Your minimax agent (Part 1)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.getNextState(agentIndex, action):
        Returns the child game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        # Begin your code (Part 1)
        '''
        I implemented minimax recursively instead of using a stack.
        The choice of max or min is determined by the agentIndex sent as a parameter.

        When the now_node(gameState) is in the parameter, the function is computing the max or min value of the descendent nodes.
        It goes through each nodes iteratively and go deeper if the descendent node isn't the end node.
        It then return the value of descendent node to let the now_node decide which one to choose.

        Notice: since I add depth whenever an agent make a choice, the condition indicating it reach the depth specified
        is depth == self.depth * number_of_agents.
        '''
        bestAction = self.compute_util_minimax(gameState, 0)
        return bestAction[1]
        
    def compute_util_minimax(self, gameState, agentIndex,depth=0):
        if agentIndex==0:                                                       # give initial value according to agent
            maxOrMin = (float('-inf'), 'STOP')
        else:
            maxOrMin = (float('inf'), 'STOP')
        depth += 1                                                              # add depth by 1
        legalActions = gameState.getLegalActions(agentIndex)
        for action in legalActions:
            nextState = gameState.getNextState(agentIndex, action)              # get all possible next states
                                                                                # return the value if it is end node
            if nextState.isLose() or nextState.isWin() or depth==self.depth*gameState.getNumAgents():
                temp = (self.evaluationFunction(nextState),)
            else:                                                               # compute downwards with the same way
                temp = self.compute_util_minimax(nextState, (agentIndex + 1) % nextState.getNumAgents(), depth)

            if  agentIndex == 0:                                                # if it is pacman, choose the max
                if temp[0] > maxOrMin[0]:
                    maxOrMin = (temp[0], action)
            else:                                                               # else, choose min
                if temp[0] < maxOrMin[0]:
                    maxOrMin = (temp[0], action)
        return maxOrMin
        # End your code (Part 1)


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (Part 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        # Begin your code (Part 2)
        '''
        I only made some modification on the recursive function.
        1. adding 2 parameter alpha and beta
        2. add some code according to the alpha-beta pruning psuedocode (between ###s)
        '''
        bestAction = self.compute_util_minimax(gameState, 0, -float('inf'), float('inf'))
        return bestAction[1]
        
    def compute_util_minimax(self, gameState, agentIndex, alpha, beta ,depth=0):
        if agentIndex==0:                                                       
            maxOrMin = (float('-inf'), 'STOP')
        else:
            maxOrMin = (float('inf'), 'STOP')
        depth += 1                                                              
        legalActions = gameState.getLegalActions(agentIndex)
        for action in legalActions:
            nextState = gameState.getNextState(agentIndex, action)
            if nextState.isLose() or nextState.isWin() or depth==self.depth*gameState.getNumAgents():
                temp = (self.evaluationFunction(nextState),)
            else:
                temp = self.compute_util_minimax(nextState, (agentIndex + 1) % nextState.getNumAgents(), alpha, beta, depth)

            if  agentIndex == 0:
                if temp[0] > maxOrMin[0]:
                    maxOrMin = (temp[0], action)
                if temp[0] > beta:                      ###
                    return temp                                 
                alpha = max(alpha, temp[0])             ###
            else:
                if temp[0] < maxOrMin[0]:
                    maxOrMin = (temp[0], action)
                if temp[0] < alpha:                     ###
                    return temp
                beta = min(beta, temp[0])               ###
        return maxOrMin
        # End your code (Part 2)

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (Part 3)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        # Begin your code (Part 3)
        '''
        The code is quite similar to minimax.
        The difference is that when it comes to the turn of the ghost, compute the value as the sum of values of all the
        possible paths divided by the number of possible paths.
        The difference is marked with ##
        
        '''
        bestAction = self.compute_util_expectimax(gameState, 0)
        #print(bestAction)
        return bestAction[1]
        
    def compute_util_expectimax(self, gameState, agentIndex, depth=0):
        if agentIndex==0:                                                       
            maxOrMin = (float('-inf'), 'LEFT')
        else:                                                               ##
            number = 0
        depth += 1                                                              
        legalActions = gameState.getLegalActions(agentIndex)
        for action in legalActions:
            nextState = gameState.getNextState(agentIndex, action)
            if nextState.isLose() or nextState.isWin() or depth==self.depth*gameState.getNumAgents():
                temp = (self.evaluationFunction(nextState),)
            else:
                temp = self.compute_util_expectimax(nextState, (agentIndex + 1) % nextState.getNumAgents(), depth)

            if  agentIndex == 0:
                if temp[0] > maxOrMin[0]:
                    maxOrMin = (temp[0], action)
            else:
                number += temp[0]                                           ##
        if agentIndex!= 0:                                                  ##
            return (number/len(legalActions),action)                        ##
        return maxOrMin
        # End your code (Part 3)

def BFS(wholeMap, position):
    Q = util.Queue()
    done = [position]                             
    ans = []
    Q.push((position,0))                                    # push initial
    direction = [(1,0),(-1,0),(0,1),(0,-1)]
    while not Q.isEmpty():
        temp = Q.pop()                                      # pop from queue
        ans.append(temp)                                    # add to answer
        for i in direction:
            newPos = (temp[0][0]+i[0], temp[0][1]+i[1])
            if(wholeMap[newPos[0]][newPos[1]] == False and newPos not in done):
                Q.push((newPos, temp[1]+1))                 # add neccesary node to queue and done
                done.append((temp[0][0],temp[0][1]))
    return ans



def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (Part 4).
    """
    # Begin your code (Part 4)
    '''
    There are three factors I used to evaluate the condition.
    1. whether the position is near the food.
        I used BFS to mark all the non-wall positions and use the value gained
        by BFS to evaluate the position.
        I tried several functions to deal with the list of values, and the one
        I am using is the best of all the function I tried.
    2. currentGameState.getScore().
        The score is higher -> the state is better
    3. capsules not eaten.
        The game will end even if the capsule isn't eaten, since you will have
        more point if you eat all the capsules, I add number of capsules * -100
        to the return value (better if less capsule).
    Additionally, I add the evaluation of ghost hunting to achieve better score.
    When the ghost is scared, the return value will become the negative of the
    distance of ghost and pacman.
    '''
    if currentGameState.isLose():                   # huge number if win or lose
        return -9999999
    elif currentGameState.isWin():
        return 9999999 + currentGameState.getScore()
    
    nowPos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood()
    wholeMap = currentGameState.getWalls()
    
    mapWithDist = BFS(wholeMap,nowPos)                          # do BFS
    foodValues = []

    capsulePositions = currentGameState.getCapsules()
    for j in mapWithDist:                                       # get all the distance of foods
        if food[j[0][0]][j[0][1]]:
            foodValues.append(j[1])
    temp = 0
    
    foodValueScore   = 2 / (min(foodValues) / float(max(foodValues) - min(foodValues) + 1) + 1) + temp
    capsuleScore = 0

    ghostState = currentGameState.getGhostStates()[0]
    hunt = False                                                # know if the ghost should be hunted
    if ghostState.scaredTimer > 0 and ghostState.scaredTimer < 28:
        hunt = True
    if not hunt:
        capsuleScore = len(capsulePositions)*-100               # less capsule -> better
    if hunt:
        ghostPositions = currentGameState.getGhostPositions()
        return - manhattanDistance(nowPos,ghostPositions[0])    # hunt the ghost (evaluation for hunting)
    return  foodValueScore*2  + currentGameState.getScore()*5 + capsuleScore
    # End your code (Part 4)

# Abbreviation
better = betterEvaluationFunction
