import copy
import time

import re
import argparse
from queue import Queue, LifoQueue, PriorityQueue

METHODS = {
    'breadth': Queue(),  # FIFO queue for BFS.
    'depth': LifoQueue(),  # LIFO queue for DFS.
    'best': PriorityQueue(),  # PriorityQueue for Best First.
    'astar': PriorityQueue()  # PriorityQueue for Astar.
}


class TreeNode(object):
    def __init__(self, state, parent, move, h, g, f):
        self.state = state
        self.parent = parent
        self.move = move
        self.h = h
        self.g = g
        self.f = f
        self.children = []

    def find_children(self, method, goal):
        moves = self.find_possible_moves()
        for state in moves:
            g = self.g + 1
            if method == 'astar':
                h = self.heuristic(state[0], goal)
                f = h + g
                self.children.append(
                    TreeNode(state[0], self, state[1], h=h, g=g, f=f))
            elif method == 'best':
                h = self.heuristic(state[0], goal)
                self.children.append(
                    TreeNode(state[0], self, state[1], h=h, g=g, f=h))
            else:
                self.children.append(
                    TreeNode(state[0], self, state[1], h=0, g=g, f=0))

    def find_possible_moves(self):
        # Initialize a dictionary with the clear blocks.
        clear_blocks = {key: value for key,
                        value in self.state.items() if value['clear']}

        moves = []
        for block, value in clear_blocks.items():
            # For every clear block.
            if value['on'] != -1:
                # Move a clear Block on table.
                on = value['on']
                temp_state = self.clear_on_table(block, on)
                moves.append(temp_state)

                for block_ in clear_blocks:
                    if block != block_:
                        # Move a clear Block on a clear Block.
                        temp_state = self.clear_on_clear(block, block_)
                        moves.append(temp_state)

            elif value['ontable']:
                # Move a Block on table on a clear Block.
                for block_ in clear_blocks:
                    if block != block_:
                        temp_state = self.table_on_clear(block, block_)
                        moves.append(temp_state)

        del clear_blocks
        return moves

    def clear_on_table(self, block, on):
        """ Move a clear block that is on another block on table. """

        # A copy of the current state.
        copy_blocks = {key: self.state[key].copy() for key in self.state}

        copy_blocks[block]['ontable'] = True
        copy_blocks[block]['on'] = -1
        copy_blocks[on]['clear'] = True
        copy_blocks[on]['under'] = -1
        move = (block, on, 'table')

        return copy_blocks, move

    def table_on_clear(self, block, block_):
        """Moves a block that is on table on a clear block."""

        # A copy of the current state.
        copy_blocks = {key: self.state[key].copy() for key in self.state}

        copy_blocks[block]['ontable'] = False
        copy_blocks[block]['on'] = block_
        copy_blocks[block_]['under'] = block
        copy_blocks[block_]['clear'] = False
        move = (block, 'table', block_)

        return copy_blocks, move

    def clear_on_clear(self, block, block_):
        """Moves a clear block that is on a block on another clear block."""

        # A copy of the current state.
        copy_blocks = {key: self.state[key].copy() for key in self.state}

        below_block = copy_blocks[block]['on']

        copy_blocks[block]['on'] = block_
        copy_blocks[below_block]['clear'] = True
        copy_blocks[below_block]['under'] = -1
        copy_blocks[block_]['under'] = block
        copy_blocks[block_]['clear'] = False
        move = (block, below_block, block_)

        return copy_blocks, move

    def get_state_list(self):
        new_state = []
        curr_objects = []
        state = self.state

        for object in state.keys():
            if state[object]["ontable"]:
                curr_objects.append(object)
                new_state.append([object])

        while len(curr_objects) > 0:
            curr_object = curr_objects[0]
            curr_objects = curr_objects[1:]

            for object in state.keys():
                if curr_object == object and state[object]["under"] != -1:
                    for temp_state in new_state:
                        if temp_state[-1] == curr_object:
                            temp_state.append(state[object]["under"])
                            curr_objects.append(state[object]["under"])
                            break
                    break

        return new_state

    def heuristic(self, state, goal):
        """Score the nodes checking every block if it's in the correct position and if
        the block under it is in the correct position. (if it has a block under it.)"""

        score = 0

        # Going over each block's position
        for block in state:
            # Checking position of block A
            # In start state, X is above A and in goal state, Y is above A (X, Y is air/block)
            # They are not same, so adding cost of 1
            if not state[block] == goal[block]:
                # If the block its not in its goal position add 1 to the score.
                score += 1

            # In start state, X is below A and in goal state, Y is below A (X, Y cannot be table)
            # They are not same, so adding cost of 1
            if not state[block]['ontable']:
                # If the block is not on table check if the block that it is on it is in the correct position.
                on = state[block]['on']
                if state[on] != goal[on]:
                    # If its not add 1 to the score.
                    score += 1

        return score

    def print_state(self):
        """Prints the current state."""
        for block, value in self.state.items():
            print(f'{block}:{value}')

    def is_goal(self, goal):
        """Checks if the currents state is equal to the goal."""
        return self.state == goal

    def get_moves_to_solution(self):
        """Returns a list with the moves you have to make in order to reach the solution."""

        temp_node = copy.copy(self)
        path = []
        while temp_node.parent is not None:
            if temp_node.move is not None:
                path.append(temp_node.move)
            temp_node = temp_node.parent

        return path

    def __lt__(self, other):
        """ Larger than operation of TreeNode object. """
        return self.f < other.f

    def __eq__(self, other):
        """ Equal operation on TreeNode object. """
        if other is not None:
            return self.state == other.state

class BlocksWorld:
    def __init__(self,
                 initial_state,
                 goal_state,
                 initial_state_as_blocks,
                 goal_state_as_blocks,
                 system1_plan = None,
                 idx = None
                 ):
        self.initial_state = initial_state
        self.goal_state = goal_state
        self.initial_state_as_blocks = initial_state_as_blocks
        self.goal_state_as_blocks = goal_state_as_blocks
        self.system1_plan = system1_plan
        self.idx = idx

    def __repr__(self):
        return self.initial_state + " | " + self.goal_state

    def load_problem(self, input):
        """ Loads the problem from the input file and replaces spaces with a hyphen. """

        data = []
        with open(input, 'r') as file:
            raw_data = file.readlines()

            for line in raw_data:
                data.append(line.strip('\n').replace(' ', '-'))

        return data


    def write_solution(self, file, solution_path):
        """Writes the solution to a file."""

        solution_path.reverse()
        with open(file, 'w') as file:
            for i, move in enumerate(solution_path):
                file.write(f'{i+1}. move {move}\n')

    @staticmethod
    def heuristic_func(initial, goal):
        score = 0
        for block in initial:
            if not initial[block] == goal[block]:
                # If the block its not in its goal position add 1 to the score.
                score += 1

            if not initial[block]['ontable']:
                # If the block is not on table check if the block that is on is in the correct position.
                on = initial[block]['on']
                if initial[on] != goal[on]:
                    # If its not add 1 to the score.
                    score += 1

        return score
    
    @staticmethod
    def verbalized_heuristic(state, goal):
        verbalization = "Let's compute step-by-step the number of mismatches between the start state and the goal state. | "

        score = 0
        for block in state:
            # Start
            verbalization += f"Checking position of block {block} in the start state. "
            if state[block]['on'] != -1:
                verbalization += f"{state[block]['on']} is under block {block}. "
            else:
                verbalization += f"{block} is on the table. "

            if state[block]['under'] != -1:
                verbalization += f"{state[block]['under']} is on top of {block}. "
            else:
                verbalization += f"{block} is clear. "

            # Goal
            verbalization += f"Now checking position of block {block} in the goal state. "
            if goal[block]['on'] != -1:
                verbalization += f"{goal[block]['on']} is under block {block}. "
            else:
                verbalization += f"{block} is on the table. "

            if goal[block]['under'] != -1:
                verbalization += f"{goal[block]['under']} is on top of {block}. "
            else:
                verbalization += f"{block} is clear. "

            # Count
            if not state[block] == goal[block]:
                verbalization += f"Hence the position of block {block} between the start state and goal state is different. Adding one to number of mismatches. So, number of mismatches = {score} + 1 = {score+1}. | "
                score += 1
            else:
                verbalization += f"Hence the position of block {block} between the start state and goal state is same. So, number of mismatches is still {score}. | "
        
        verbalization += "Now checking blocks that are not on the table. | "
        for block in state:
            verbalization += f"Checking Block {block}. "
            if not state[block]['ontable']:
                verbalization += f"Block {block} is not on the table. "
                # If the block is not on table check if the block that is on is in the correct position.
                on = state[block]['on']
                verbalization += f"Block {on} is below block {block}. "
                if state[on] != goal[on]:
                    # If its not add 1 to the score.
                    verbalization += f"Position of block {on} between the start state and goal state is different. Adding one to number of mismatches. So, number of mismatches = {score} + 1 = {score+1}. | "
                    score += 1
                else:
                    verbalization += f"Position of block {on} between the start state and goal state is same. So, number of mismatches is still {score}. | "
            else:
                verbalization += f"Block {block} is on the table. So, number of mismatches is still {score}. | "

        verbalization += f"The final number of mismatches is {score}."

        return verbalization, score

    def search(self, queue, method, initial, goal):
        """Searches the tree for a solution based on the search algorithm."""
        # print('#####')
        # print(initial)
        # print('#####')
        # print(goal)
        root = TreeNode(initial, None, None, 0, 0, 0)

        if method == 'astar' or method == 'best':
            # If the method uses a heuristic a PriorityQueue is initialized with the root.
            queue.put((0, root))
        else:
            queue.put(root)

        visited_set = set()  # Set of visited states.
        start = time.time()
        while (not queue.empty()) and (time.time() - start <= 60):
            # While the queue is not empty and a minutes hasn't passed.

            if method == 'astar' or method == 'best':
                # PriorityQueue .get() method returns the priority number and the element.
                curr_f, current = queue.get()
            else:
                current = queue.get()

            if current.is_goal(goal):
                plan = current.get_moves_to_solution()
                plan.reverse()
                return plan

            if str(current.state) in visited_set:
                # If this state has been visited before don't add it to the children
                # and continue with the next child.
                continue

            current.find_children(method, goal)
            visited_set.add(str(current.state))  # Mark the state as visited.

            # Add every child in the search queue.
            for child in current.children:
                if method == 'depth' or method == 'breadth':
                    queue.put(child)
                elif method == 'astar' or method == 'best':
                    queue.put((child.f, child))

        return None

    def follow_plan(self, start_state, plan):
        curr_state = TreeNode(start_state, None, None, 0, 0, 0)
        for step in plan:
            move_components = step.split(" ")
            move = (move_components[1], move_components[3], move_components[5])
            found = False
            for possible_state, possible_move in curr_state.find_possible_moves():
                if possible_move == move:
                    curr_state = TreeNode(possible_state, curr_state.state, possible_move, 0, 0, 0)
                    found = True
                    break

            assert found == True, f"Move {move} not valid"

        return curr_state.state, curr_state.get_state_list()

    
    def convert_steps_to_move(self, pickup_step, putdown_step):
        pickup_step = pickup_step.split(" ")
        putdown_step = putdown_step.split(" ")

        # Check action validity
        if pickup_step[0] not in ['pickup', 'unstack']:
            return None
        
        # Check action validity
        if putdown_step[0] not in ['putdown', 'stack']:
            return None

        # Check if the object being picked up is the one put down
        if pickup_step[1] != putdown_step[1]:
            return None

        block = pickup_step[1]
        source = pickup_step[-1]
        dest = putdown_step[-1]

        return (block, source, dest)
    
    def is_valid_plan(self, plan=None, start=None, goal=None, format='move'):
        if plan == '':
            return False
        
        start = start if start else self.initial_state_as_blocks
        goal = goal if goal else self.goal_state_as_blocks
        plan = plan if plan else self.system1_plan

        curr_state = TreeNode(start, None, None, 0, 0, 0)
        plan_steps = plan.split(" | ")

        # The plan length should be even
        if format != 'move' and len(plan_steps) % 2 == 1:
            return False

        iterator_steps = 2 if format != 'move' else 1
        for i in range(0, len(plan_steps), iterator_steps):
            if format == 'move':
                move = plan_steps[i]
                move_components = move.split(" ")
                if len(move_components) != 6:
                    return False
                move = (move_components[1], move_components[3], move_components[5])
            else:
                pickup_step = plan_steps[i]
                putdown_step = plan_steps[i+1]

                move = self.convert_steps_to_move(pickup_step, putdown_step)

            # If we cant even parse, return False
            if not move:
                return False

            valid_move = False
            for possible_state, possible_move in curr_state.find_possible_moves():
                if possible_move == move:
                    valid_move = True
                    break
            
            # If move not a valid move, return False
            if not valid_move:
                return False

            # Contruct the new tree node (state)
            curr_state = TreeNode(possible_state, curr_state.state, possible_move, 0, 0, 0)
        
        return curr_state.state == goal



