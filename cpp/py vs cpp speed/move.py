import time
from helpers import blur

# The 2D move function
def move(dy, dx, beliefs, blurring):
    height = len(beliefs)
    width = len(beliefs[0])
    new_G = [[0.0 for i in range(width)] for j in range(height)]
    for i, row in enumerate(beliefs):
        for j, cell in enumerate(row):
            new_i = (i + dy ) % width
            new_j = (j + dx ) % height
            new_G[int(new_i)][int(new_j)] = cell
    return blur(new_G, blurring)

# Initialize a world grid of beliefs
grid = [ [0.05, 0.2, 0.2, 0.05, 0.05],
         [0.05, 0.1, 0.2, 0.05, 0.05] ]

# Record the time it takes to execute the move function
# Start the timer and record the start time
start_time = time.time()

# Time the move function for 1000 iterations
# move 2 to the right each iteration

### TODO: Change the number of iterations and see how this performs - try 10000x !
### TODO: Run multiple times and notice the slight change in recorded time
iterations = 1000

for i in range(0, iterations):
    move(0, 2, grid, 1.0)

# Record the end time and print the result!
stop_time = time.time()

millisec_time = 1000*(stop_time - start_time)
print("Time to move " +str(iterations) + " iterations in milliseconds: " + str(millisec_time))
