import os
def copy(size,r=3,c=5):
    dx=size[0]/(c-1)
    dy=size[1]/(r-1)
    x=0
    y=0
    re=[]
    for i in range(r):
        for j in range(c):
            re.append((x,y))
            x=x+dx
        y=y+dy
        x =0
    return re
# Get the current script's path
current_script_path = __file__
# Get the parent directory path
parent_dir_path = os.path.dirname(current_script_path)
re = copy((1199,599))
with open(os.path.dirname(parent_dir_path)+'/data/phco.txt','w') as f:
    for i,it in enumerate(re):
        f.write(str(i)+" "+str(int(it[0]))+" "+str(int(it[1])))
        f.write("\n")