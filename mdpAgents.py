# mdpAgents.py
# Author: Eamonn Mansour
#
# This file implements an MDP-based Pac-Man Agent that employs Value Iteration
# using information from Pac-Man's environment to guide Pac-Man to victory.
# This information is stored in a map of rewards, and a map of utilities
# that the MDP Agent calculates.
#
# ================================
# Intended to work with the PacMan AI projects from:
#
# http://ai.berkeley.edu/
#
# These use a simple API that allow us to control Pacman's interaction with
# the environment adding a layer on top of the AI Berkeley code.
#
# As required by the licensing agreement for the PacMan AI we have:
#
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

from pacman import Directions
from game import Agent
import api
import util

class Map(object):
    """
    A class representing a Pac-Man map. The Map class provides all map-related functionality for use by the MDPAgent,
    including map rewards and ghost configurations.
    """

    GHOST_REWARD = -50
    FOOD_REWARD = 10
    CAPSULE_REWARD = 10
    EMPTY_REWARD = -0.4
    WALL = -10
    MEDIUM_GHOST_BUFFER = 3
    SMALL_GHOST_BUFFER = 2

    def __init__(self, state):
        walls = set(api.walls(state))
        maxCorner = max(api.corners(state))

        self.__height = maxCorner[1] + 1
        self.__width = maxCorner[0] + 1
        self.__map = [
            [self.WALL if (x, y) in walls else self.EMPTY_REWARD for y in range(self.height)]
            for x in range(self.width)
        ]

        self.__buffer = self.MEDIUM_GHOST_BUFFER if self.width >= 15 else self.SMALL_GHOST_BUFFER

    @property
    def height(self):
        return self.__height

    @property
    def width(self):
        return self.__width

    @property
    def buffer(self):
        return self.__buffer

    def getValue(self, x, y):
        return self.__map[x][y]

    def setValue(self, x, y, value):
        self.__map[x][y] = value

    def updateMap(self, state):
        """Updates the map based on current state information."""

        self.resetMap()
        pacman = api.whereAmI(state)
        ghosts = api.ghostStatesWithTimes(state)

        # The API sometimes gives ghost locations as floats, this ensures they are integers
        ghosts = [[(int(ghost[0][0]), int(ghost[0][1])), ghost[1]] for ghost in ghosts]

        # The further food is from non-edible ghosts, the safer and more valuable it is
        foods = api.food(state)
        for food in foods:
            foodToGhost = [util.manhattanDistance(food, ghost[0]) for ghost in ghosts if ghost[1] == 0]
            foodReward = (min(foodToGhost or [0])) * 2 + self.FOOD_REWARD
            self.setValue(food[0], food[1], foodReward)

        capsules = api.capsules(state)
        for capsule in capsules:
            self.setValue(capsule[0], capsule[1], self.CAPSULE_REWARD)

        for ghost in ghosts:
            ghostReward = self.GHOST_REWARD

            # Pacman should try to eat ghosts if there is enough time to travel to edible ghosts
            pacmanToGhost = util.manhattanDistance(pacman, ghost[0])
            if ghost[1] > pacmanToGhost:
                ghostReward = self.FOOD_REWARD * 100
            self.setValue(ghost[0][0], ghost[0][1], ghostReward)

        # Find the states in the ghosts' buffer areas when ghosts are not edible
        ghostNeighbours = []
        for ghost in ghosts:
            if ghost[1] == 0:
                (x, y) = ghost[0]
                ghostNeighbours.extend(self.getGhostBuffer(state, [(x, y)], self.buffer))

        # Set buffer area rewards around ghosts
        for (x, y) in ghostNeighbours:
            if x in range(1, self.width - 1) and y in range(1, self.height - 1):
                if self.getValue(x, y) != self.WALL:
                    self.setValue(x, y, self.GHOST_REWARD / 2)

    def resetMap(self, initial=EMPTY_REWARD):
        """Reset all locations to the empty reward except walls (as walls are static)."""

        for y in range(1, self.height - 1):
            for x in range(1, self.width - 1):
                if self.getValue(x, y) != self.WALL:
                    self.setValue(x, y, initial)

    def getGhostBuffer(self, state, neighbours, bufferSize):
        """Retrieves the neighbouring states to a ghost, which form the buffer area."""

        walls = set(api.walls(state))
        ghosts = set(api.ghosts(state))
        ghosts = {(int(x), int(y)) for (x, y) in ghosts}

        if bufferSize == 0:
            neighbours.pop(0)
            return neighbours

        newNeighbours = []
        for (x, y) in neighbours:
            newNeighbours.extend([(x, y + 1), (x, y - 1), (x + 1, y), (x - 1, y)])
        newNeighbours = [loc for loc in newNeighbours if loc not in walls and loc not in ghosts]

        neighbours.extend(list(set(newNeighbours)))
        return self.getGhostBuffer(state, neighbours, bufferSize - 1)


class MDPAgent(Agent):
    """A class representing a Pac-Man agent that uses Value Iteration to solve MDPs."""

    DISCOUNT = 0.6
    PROB_SUCCESS = api.directionProb
    PROB_FAIL = (1 - PROB_SUCCESS) / 2
    OFFSET = {
        Directions.NORTH: (0, 1),
        Directions.SOUTH: (0, -1),
        Directions.EAST: (1, 0),
        Directions.WEST: (-1, 0),
        Directions.STOP: (0, 0)
    }

    def __init__(self):
        self.legal = None

    def registerInitialState(self, state):
        """Runs after initialisation to provide the agent with an initial game state."""

        self.map = Map(state)
        self.rewardMap = Map(state)

    def final(self, _):
        """Runs at the end of every game, tearing down properties."""

        self.legal = None
        self.map = None
        self.rewardMap = None

    def getAction(self, state):
        self.legal = set(api.legalActions(state))
        pacman = api.whereAmI(state)

        self.map.resetMap(0)
        self.rewardMap.updateMap(state)

        self.valueIterate(25)
        move = self.getMEUDirection(pacman)
        return api.makeMove(move, list(self.legal))

    def addOffset(self, direction, location):
        """Calculates and returns the result of adding a given direction's offset with a given location."""

        return tuple(sum(loc) for loc in zip(self.OFFSET[direction], location))

    def getProb(self, direction, location, prob):
        """Calculates and returns the probability of moving in a given direction from the current location."""

        newLocation = self.addOffset(direction, location)
        if self.map.getValue(newLocation[0], newLocation[1]) != self.map.WALL:
            (x, y) = newLocation
        else:
            (x, y) = location

        return prob * self.map.getValue(x, y)

    def updateUtility(self, location):
        """Calculates and updates the Maximum Expected Utility for a given location."""

        directions = list(self.OFFSET.keys())
        utils = {}
        for direction in directions:
            utility = self.getProb(direction, location, self.PROB_SUCCESS)
            utility += self.getProb(Directions.LEFT[direction], location, self.PROB_FAIL)
            utility += self.getProb(Directions.RIGHT[direction], location, self.PROB_FAIL)
            utils[direction] = utility

        (x, y) = location
        utility = self.rewardMap.getValue(x, y) + self.DISCOUNT * max(utils.values())
        self.map.setValue(x, y, utility)

    def getMEUDirection(self, location):
        """Returns the direction with the Maximum Expected Utility."""

        utils = {}
        for direction in self.legal:
            (x, y) = self.addOffset(direction, location)
            utils[direction] = self.map.getValue(x, y)

        nextMove = max(utils, key=utils.get)
        return nextMove

    def valueIterate(self, n):
        """Performs (Approximated) Value Iteration by iterating over the map 'n' times and updating map utilities."""

        for _ in range(n):
            # No need to calculate utilities for any outer wall locations
            for y in range(1, self.map.height - 1):
                for x in range(1, self.map.width - 1):
                    if self.map.getValue(x, y) != self.map.WALL:
                        self.updateUtility((x, y))
