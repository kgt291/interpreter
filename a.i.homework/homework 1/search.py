# -*- coding: cp949 -*-
# search.py
# ---------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

"""
In search.py, you will implement generic search algorithms which are called
by Pacman agents (in searchAgents.py).
"""
# -*- coding: cp949 -*-

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other
    maze, the sequence of moves will be incorrect, so only use this for tinyMaze
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s,s,w,s,w,w,s,w]

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def iterativeDeepingSearch(problem):
    "*** YOUR CODE HERE ***"

    num = 0 # num = depth 판별
    while 1:
        fringe = util.Stack() # fringe는 스택 구조(선입후출)로 각 node, action, cost 정보를 받습니다.
        fringe.push((problem.getStartState(), [], 0))
        visited = [] # 방문한 노드인지 판별
        
        while not fringe.isEmpty(): # fringe가 비어있을 때까지(더 이상 갈 노드 없을때까지)
           node, actions, cost = fringe.pop()
           if problem.isGoalState(node): #노드가 goal이면 배열에 있는 action에 따라 이동
                return actions

           visited.append(node) # 방문한 노드 추가
         
           if cost < num: # 이 예제에서는 매 노드 이동할 때마다 cost가 다 1로 늘어나기 때문에 cost가 노드의 depth 역할을 하여 몇 번째 depth인지 판별함
            for state, action, costs in problem.getSuccessors(node): #노드 extend함
               if not state in visited:  #방문하지 않은 노드이면
                 fringe.push((state, actions + [action], cost + costs)) #fringe에 새 state, action, cost 추가
                
        num = num + 1  # 각 iterative 마다 depth 판별이 늘어남   
                   
    # node =  이동한 state, action = 이동, cost = 이동한 거리
               


def Heuristic(state, problem=None):
    "*** YOUR CODE HERE ***"
    return 22

def aStarSearch(problem, heuristic=nullHeuristic):
    

    fringe = util.PriorityQueue() # 우선순위큐 구조로 각 노드의 cost + heuristic값의 우선순위로 입출이 다름
    start = problem.getStartState()
    fringe.push((start, [], 0), heuristic(start, problem))
    expanded = [] # expand한 노드였는지 판별
    
    while not fringe.isEmpty(): #fringe가 비어있을 때까지(더 이상 갈 노드 없을 때까지)
        node, actions, cost = fringe.pop()

        if problem.isGoalState(node): #노드가 goal이면 배열에 있는 action에 따라 이동
            return actions

        expanded.append(node) # expand한 노드 추가
        
        for state, action, costs in problem.getSuccessors(node): #노드 expand
            new_actions = actions + [action] # 새 action 추가
            new_cost = cost + costs # 이동 거리 증가
            total = new_cost + heuristic(state, problem) # 현재 이동한 거리 + heuristic
            if not state in expanded: # expand한 노드 아니면 fringe 추가
                fringe.push((state, new_actions, new_cost), total)
                
        # node =  이동한 state, action = 이동, cost = 이동한 거리

# Abbreviations
ids= iterativeDeepingSearch
astar = aStarSearch
