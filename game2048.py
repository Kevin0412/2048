import random
import copy

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

class game:
    def __init__(self):
        self.board=board()
        self.score=0
        self.movements=-2
        self.blocks=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        self.history=[[0,0]]

    def __repr__(self):
        output=""
        for a in range(1,len(self.history)):
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
            print(self.board)
            if self.move2(input('w:up a:left s:down d:right ')):
                self.new_bolck()
            else:
                print("invalid move\a")
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
        while not self.move2(action):
            action=self.action_space.item()
        self.new_bolck()
        if self.end():
            return self.board.flatten(), self.score, True
        return self.board.flatten(), self.score, False
    
    def reset(self):
        super().reset()
        self.new_bolck()
        self.new_bolck()
        return self.board.flatten()     

    def render_in_terminal(self):
        done=False
        while not done:
            print("score:\t"+str(self.score))
            print("movements:\t"+str(self.movements))
            print(self.board)
            if self.move2(self.action_space.item()):
                self.new_bolck()
            else:
                print("invalid move\a")
            if self.end():
                print(self)
                done=True

if __name__=="__main__":
    #game().play_in_terminal()
    gym_env().render_in_terminal()
