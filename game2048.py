import random
import copy

import math
import statistics

class board:
    def __init__(self):
        self.board=[[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]

    def __repr__(self):
        output=""
        for b in self.board:
            for a in b:
                output+="|"
                if a==0:
                    output+='    '
                elif a<4:
                    output+='  '+str(2**a)+' '
                elif a<7:
                    output+=' '+str(2**a)+' '
                elif a<10:
                    output+=' '+str(2**a)
                elif a<14:
                    output+=str(2**a)
                else:
                    output+='2^'+str(a)
            output+='|\n'
        return output
    
    def __getitem__(self,index):
        return self.board[index]
    
    def __setitem__(self,index,value):
        if isinstance(index, tuple):
            row, col = index
            self.board[row][col] = value
        else:
            self.board[index] = value

    def __call__(self, state):
        # 更新棋盘状态
        self.board = state
        return self
    
    def rotate(self,times):
        for t in range(times):
            new=board()
            for x in range(4):
                for y in range(4):
                    new[x][y]=self[y][3-x]
            self.board=new.board

    def end(self):
        for b in self:
            for a in b:
                if a==0:
                    return False
        for a in range(4):
            for b in range(3):
                if self[a][b]==self[a][b+1]:
                    return False
                if self[b][a]==self[b+1][a]:
                    return False
        return True
    
    def flatten(self):
        return [item for sublist in self.board for item in sublist]
    
    def move_score(self,direction):
        if not isinstance(direction,str):
            print(direction,"is invalid")
            raise
        new_score=0
        if direction=='d':
            self.rotate(2)
        elif direction=='w':
            self.rotate(1)
        elif direction=='s':
            self.rotate(3)
        for b in self.board:
            d=0
            for c in b:
                if c!=0:
                    if d==c:
                        new_score+=2*2**c
                        d=0
                    else:
                        d=c

        if direction=='d':
            self.rotate(2)
        elif direction=='w':
            self.rotate(3)
        elif direction=='s':
            self.rotate(1)
        return new_score
    
    def moveable(self,direction):
        if not isinstance(direction,str):
            print(direction,"is invalid")
            raise
        output=False
        if direction=='d':
            self.rotate(2)
        elif direction=='w':
            self.rotate(1)
        elif direction=='s':
            self.rotate(3)
        for b in self.board:
            d=-1
            for c in b:
                if c!=0:
                    if c==d or d==0:
                        output=True
                        break
                d=c
            if output:
                break
        if direction=='d':
            self.rotate(2)
        elif direction=='w':
            self.rotate(3)
        elif direction=='s':
            self.rotate(1)
        return output
    
    def move(self,direction):
        if not isinstance(direction,str):
            print(direction,"is invalid")
            raise
        new=board()
        if direction=='d':
            self.rotate(2)
        elif direction=='w':
            self.rotate(1)
        elif direction=='s':
            self.rotate(3)
        e=0
        for b in self.board:
            c=[]
            for a in b:
                if a!=0:
                    c.append(a)
            d=[]
            a=0
            while True:
                if len(c)==0:
                    break
                if a+1==len(c):
                    d.append(c[a])
                    break
                if c[a]!=c[a+1]:
                    d.append(c[a])
                    a+=1
                else:
                    d.append(c[a]+1)
                    a+=2
                if a==len(c):
                    break
            for f in range(4-len(d)):
                d.append(0)
            new[e]=d
            e+=1
        if direction=='d':
            self.rotate(2)
        elif direction=='w':
            self.rotate(3)
        elif direction=='s':
            self.rotate(1)
        return new
    
    def max_in_corner(self):
        sorted_board=sorted(self.flatten(), reverse=True)
        return sorted_board[0]==self.board[0][0] or sorted_board[0]==self.board[0][3] or sorted_board[0]==self.board[3][0] or sorted_board[0]==self.board[3][3]
    
    def num_blocks(self):
        return sum([1 for a in self.flatten() if a!=0])
    
    def normalize(self):
        return [a/17 for a in self.flatten()]
    
    def normalize_2d(self):
        return [[a/17 for a in row] for row in self.board]
    
    def snake(self):
        reward=1
        sorted_board=sorted(self.flatten(), reverse=True)
        for x in range(4):
            flag=False
            for y in range(4):
                if self[x][y]==sorted_board[0]:
                    flag=True
                    break
            if flag:
                break
        for block,i in zip(sorted_board,range(16)):
            if i>0:
                if block==0:
                    break
                for j in range(-1,2,2):
                    if x+j>=0 and x+j<=3:
                        if self[x+j][y]==block:
                            reward+=1
                            x+=j
                            break
                    if y+j>=0 and y+j<=3:
                        if self[x][y+j]==block:
                            reward+=1
                            y+=j
                            break
        return reward
    
    def reward(self):
        return 0 if not self.max_in_corner() else 0 if self.snake()==1 else (self.snake()-1)/(self.num_blocks()-1)

class game:
    def __init__(self):
        self.board=board()
        self.score=0
        self.movements=-2
        self.blocks=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        self.history=[[0,0]]
        self.reward=0

    def __repr__(self):
        output=""
        for a in range(1,len(self.history)):
            if len(self.history[a])!=0:
                output+="first "+str(2**(a+1))+":\n"
                output+="  score:\t"+str(self.history[a][0])+"\n"
                output+="  movements:\t"+str(self.history[a][1])+"\n"
                for b in range(2,len(self.history[a])-1):
                    output+="  new "+str(2**b)+":\t"+str(self.history[a][b])+"\n"
                output+=str(self.history[a][-1])+"\n"
        output+="end:\n"
        output+="  score:\t"+str(self.score)+"\n"
        output+="  movements:\t"+str(self.movements)+"\n"
        output+="  max:\t"+str(2**len(self.history))+"\n"
        for a in range(len(self.history)-1):
            output+="  new "+str(2**(a+2))+":\t"+str(self.blocks[a])+"\n"
        output+=str(self.board)
        return output
    
    def end(self):
        if len(self.history[-1])==0:
            self.history[-1].append(self.score)
            self.history[-1].append(self.movements)
            for x in range(len(self.history)-1):
                self.history[-1].append(self.blocks[x])
            self.history[-1].append(copy.deepcopy(self.board))
        return self.board.end()
    
    def move(self):
        e=0
        g=0
        for b in self.board:
            c=[]
            for a in b:
                if a!=0:
                    c.append(a)
            d=[]
            a=0
            while True:
                if len(c)==0:
                    break
                if a+1==len(c):
                    d.append(c[a])
                    break
                if c[a]!=c[a+1]:
                    d.append(c[a])
                    a+=1
                else:
                    d.append(c[a]+1)
                    if self.blocks[c[a]-1]==0:
                        self.history.append([])
                    self.blocks[c[a]-1]+=1
                    self.score+=2*2**c[a]
                    a+=2
                if a==len(c):
                    break
            for f in range(4-len(d)):
                d.append(0)
            if self.board[e]!=d:
                g+=1
            self.board[e]=d
            e+=1
        if g==0:
            return False
        return True
    
    def move2(self,direction):
        if not isinstance(direction,str):
            print(direction,"is invalid")
            raise
        if direction=='a':
            a=self.move()
        elif direction=='d':
            self.board.rotate(2)
            a=self.move()
            self.board.rotate(2)
        elif direction=='w':
            self.board.rotate(1)
            a=self.move()
            self.board.rotate(3)
        elif direction=='s':
            self.board.rotate(3)
            a=self.move()
            self.board.rotate(1)
        else:
            return False
        return a
    
    def new_bolck(self):
        a=1
        if random.randint(0,9)==random.randint(0,9):
            a=2
        while True:
            b=random.randint(0,3)
            c=random.randint(0,3)
            if self.board[b][c]==0:
                self.board[b][c]=a
                self.movements+=1
                break

    def play_in_terminal(self):
        self.new_bolck()
        self.new_bolck()
        while True:
            print("score:\t"+str(self.score))
            print("movements:\t"+str(self.movements))
            print("snake:\t"+str(self.board.snake()))
            print(self.board)
            direction=input('w:up a:left s:down d:right ')
            if self.board.moveable(direction):
                #self.reward+=(0 if self.board.move_score(direction)==0 else math.log2(self.board.move_score(direction)) if self.score==0 else math.log2(self.board.move_score(direction)/self.score+1))
                self.move2(direction)
                self.new_bolck()
            else:
                print("invalid move\a")
            self.reward += self.board.reward()
            print("reward:\t"+str(self.reward))
            if self.end():
                print(self)
                break
        print("game over")

    def reset(self):
        self.board=board()
        self.score=0
        self.movements=-2
        self.blocks=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        self.history=[[0,0]]
        self.reward=0

class action:
    def __init__(self):
        self.n = 4
        self.actions=["w","a","s","d"]

    def sample(self):
        return random.randint(0,3)
    
    def item(self):
        return self.actions[self.sample()]

    def __call__(self):
        return self.sample()

class gym_env(game):
    def __init__(self):
        super().__init__()
        self.new_bolck()
        self.new_bolck()
        self.action_space = action()
    
    def step(self,action):
        if isinstance(action, int):
            action=self.action_space.actions[action]
        
        while not self.board.moveable(action):
            action=self.action_space.item()

        #reward=0 if self.board.move_score(action)==0 else math.log2(self.board.move_score(action)) if self.score==0 else math.log2(self.board.move_score(action)/self.score+1)
        #reward = self.board.move_score(action)
        self.move2(action)
        self.new_bolck()
        reward = self.board.reward()
        if self.end():
            return self.board.normalize_2d(), reward, True
        return self.board.normalize_2d(), reward, False
    
    def reset(self):
        super().reset()
        self.new_bolck()
        self.new_bolck()
        return self.board.normalize_2d()

    def render_in_terminal(self,show=True):
        done=False
        while not done:
            if show:
                print("score:\t"+str(self.score))
                print("movements:\t"+str(self.movements))
                print(self.board)
            if self.move2(self.action_space.item()):
                self.new_bolck()
            else:
                if show:
                    print("invalid move\a")
            if self.end():
                print(self)
                done=True

if __name__=="__main__":
    """a=board()
    a.board=[
        [1,2,0,2],
        [6,5,3,1],
        [3,6,1,4],
        [1,2,3,1]
    ]
    print(a)
    print(a.moveable("s"))"""
    
    #game().play_in_terminal()
    scores=[]
    a=gym_env()
    for i in range(2048):
       a.render_in_terminal(show=False)
       scores.append(math.log2(a.score))
       a.reset()
    print(statistics.mean(scores),"±",statistics.stdev(scores))
        
