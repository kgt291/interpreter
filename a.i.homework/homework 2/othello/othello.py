#-*- coding: cp949 -*-
import gamePlay
import random
from copy import deepcopy

def maxValue(newBoard, alpha, beta, color, depth):
    
    score = float("-inf")
    best = score
    moves = []
    for i in range(8):
        for j in range(8):
            if gamePlay.valid(newBoard, color, (i,j)):
                moves.append((i, j))
    if len(moves) == 0: # 더이상 할 수 있는 move가 없을 경우 pass
        return "pass"
    
    if color == "B" : # 나의 color 찾기
        m = 0
    elif color == "W" :
        m = 1
   
    if gamePlay.gameOver(newBoard) : #game이 누군가의 승리로 종료되었을 경우
        return gamePlay.score(newBoard)[m] 
    
    bestMove = moves[random.randint(0,len(moves) - 1)]
    
    for move in moves :
        uBoard = deepcopy(newBoard)
        gamePlay.doMove(uBoard, color, move) 
        if depth == 2 :
            score = gamePlay.score(uBoard)[m] #나의 점수 얻기
        else :            
            score = minValue(uBoard, alpha, beta, gamePlay.opponent(color), depth) #상대편의 min layer로 이동
        if score > best : # 받은 점수가 현재까지 제일 컸던 점수보다 클 경우
            best = score
            bestMove = move
        alpha = max(alpha, score)            
        if score > beta: # 더이상 큰 값이 나올 수 없으므로 가지치기
            return best   
    if depth == 0 : # 최적의 action을 찾았을 때
         return bestMove
    else :     # 최적의 점수를 찾았을 때   
         return best    

def minValue(newBoard, alpha, beta, color, depth):
    
    score = float("inf")
    best = score
    moves = []
    for i in range(8):
        for j in range(8):
            if gamePlay.valid(newBoard, color, (i,j)):
                moves.append((i, j))
    if color == "B" : #상대편의 color 찾기
        m = 0
    elif color == "W" :
        m = 1
                
    if gamePlay.gameOver(newBoard) : #game이 누군가의 승리로 종료되었을 경우
        return gamePlay.score(newBoard)[m]
    
    for move in moves :
        uBoard = deepcopy(newBoard)
        gamePlay.doMove(uBoard, color, move)
        score = maxValue(uBoard, alpha, beta, gamePlay.opponent(color), depth + 1) #나의 max layer로 이동
        if score < best : # 받은 점수가 현재까지 제일 작았던 점수보다 작을 경우
            best = score
        beta = min(beta, score)
        if best < alpha : # 더 이상 작은 값이 나올 수 없으므로 가지치기
            return best     
     
    return best
    
def nextMove(board, color, time):
    return maxValue(board, float("-inf"), float("inf"), color, 0) #나의 max layer로 이동
    
    
