class MazeState:
    def __init__(self,
                 state,
                 parent_state,
                 action,
                 plan,
                 dist,
                 state_type,
                 heuristic=None):
        self.state = state
        self.parent_state = parent_state
        self.action = action
        self.plan = plan
        self.dist = dist
        self.state_type = state_type
        self.heuristic = heuristic

    def __repr__(self):
        return self.action + " [" + str(self.state[0]) + ", " + str(self.state[1]) + "] " + str(self.dist) + " " + self.state_type

class Maze:
    def __init__(self, 
                 l=5, 
                 w=5,
                 actions=['left', 'right', 'up', 'down'],
                 start=[0,0],
                 goal=[4,4],
                 walls=[],
                 system1_plan=None,
                 system2_plan=None,
                 idx=None
                 ):
        self.l = l
        self.w = w
        self.grid = [[0]*self.w for i in range(self.l)]

        self.actions = actions

        self.start = start
        self.goal = goal

        for wall in walls:
            self.grid[wall[0]][wall[1]] = 1

        self.walls = walls

        self.system1_plan = system1_plan
        self.system2_plan = system2_plan
        self.idx = idx

    def execute_action(self, state, action):
        if action == 'up':
            return [state[0]-1, state[1]]
        elif action == 'down':
            return [state[0]+1, state[1]]
        elif action == 'left':
            return [state[0], state[1]-1]
        else:
            return [state[0], state[1]+1]
        
    def follow_plan(self, start_state, plan):
        next_state = start_state
        for action in plan:
            next_state = self.execute_action(next_state, action)

        return next_state
        
    def break_plan_into_steps(self, plan):
        plan_steps = plan.replace('[', '').replace(']', '').replace(',', '').split(' | ')
        return plan_steps

    def is_valid_plan(self, plan=None, start=None, goal=None):
        if plan == '':
            return False
        start = start if start else self.start
        goal = goal if goal else self.goal
        plan = plan if plan else self.system1_plan

        prev_state = None
        plan_steps = self.break_plan_into_steps(plan)

        for i, plan_step in enumerate(plan_steps):
            plan_step_components = plan_step.split(' ')
            curr_action = plan_step_components[0]

            if (i == 0 and curr_action != 'start') or (i > 0 and curr_action not in self.actions):
                return False
            
            curr_state = self.execute_action(prev_state, curr_action) if prev_state else start

            if not self.is_valid_state(curr_state):
                return False
            
            prev_state = curr_state

            if curr_state == goal:
                break
        
        return curr_state == goal
    
    def is_optimal_plan(self, plan=None, check_validity=True, start=None, goal=None):
        if check_validity and not self.is_valid_plan(plan, start, goal):
            return False
        
        if plan == '':
            return False
        
        start = start if start else self.start
        goal = goal if goal else self.goal
        plan = plan if plan else self.system1_plan

        optimal_plan = self.system1_plan if self.system1_plan else self.bfs(start, goal)
        optimal_plan_steps = optimal_plan.split(" | ")

        plan_steps = plan.split(" | ")
        
        return len(optimal_plan_steps) == len(plan_steps)
    
    def is_sub_goals_valid(self, sub_goals):
        for sub_goal in sub_goals:
            if sub_goal[0] in self.walls or sub_goal[1] in self.walls:
                return False
            
        return True

    def abs_distance(self, loc1, loc2):
        return [abs(loc1[0]-loc2[0]), abs(loc1[1]-loc2[1])]
    
    @staticmethod
    def manhattan_distance(state1, state2):
        return abs(state1[0]-state2[0]) + abs(state1[1]-state2[1])
    
    @staticmethod
    def obstacles(state1, state2, walls):
        obstacle = 0
        for row in range(min(state1[0], state2[0]), max(state1[0], state2[0])+1):
            for col in range(min(state1[1], state2[1]), max(state1[1], state2[1])+1):
                if [row, col] in walls:
                    obstacle += 1

        return obstacle

    def is_valid_state(self, state):
        return state[0] >= 0 and state[0] < self.l and state[1] >= 0 and state[1] < self.w and self.grid[state[0]][state[1]] == 0

    def dfs_util(self, curr_state, goal, visited, dfs_trace, dfs_search_string):
        if curr_state.state == goal:
            dfs_search_string.append(f"Goal state {goal} reached!")
            return True
        
        for action in self.actions:
            new_state = self.execute_action(curr_state.state, action)
            dfs_search_string.append(f"Exploring action '{action}' to move to state {new_state} | ")
            if self.is_valid_state(new_state) and visited[new_state[0]][new_state[1]] == 0:
                visited[new_state[0]][new_state[1]] = 1

                dfs_search_string.append(f"State {new_state} is valid | ")
                dfs_search_string.append(f"Taking action '{action}' from state {curr_state.state} | ")

                dfs_search_string.append(f"Moved to state {new_state} | ")

                new_maze_state = MazeState(
                    state = new_state,
                    parent_state = curr_state,
                    action = action,
                    plan = curr_state.plan + [action],
                    dist = curr_state.dist + 1,
                    state_type = "Valid"
                )

                dfs_search_string.append(f"Plan so far {new_maze_state.plan} | ")

                dfs_trace.append(new_maze_state)
                is_reachable = self.dfs_util(new_maze_state, goal, visited, dfs_trace, dfs_search_string)

                if is_reachable:
                    return True
            else:
                dfs_search_string.append(f"State {new_state} is invalid | ")


    def dfs(self, start=None, goal=None):
        start = start if start else self.start
        goal = goal if goal else self.goal

        curr_state = MazeState(
            state = start,
            parent_state = None,
            action = "start",
            plan = [],
            dist = 0,
            state_type = "Valid"
            )

        visited = [[0]*self.w for i in range(self.l)]

        dfs_trace = [curr_state]

        dfs_search_string = [f"Moved to state {start} | "]
        dfs_search_string.append(f"Plan so far {curr_state.plan} | ")

        visited[start[0]][start[1]] = 1

        self.dfs_util(curr_state, goal, visited, dfs_trace, dfs_search_string)

        dfs_search_string = "".join(dfs_search_string)

        plan_len = dfs_trace[-1].dist

        curr_state = dfs_trace[-1]
        system1_plan = [curr_state]
        while True:
            if curr_state.state == start:
                break

            curr_state = curr_state.parent_state
            system1_plan = [curr_state] + system1_plan

        parent_pointer_plan = [state.action for state in system1_plan][1:]
        plan_so_far_plan = dfs_search_string.split(" | ")[-2][len("Plan so far ["):-1].replace("'", "").split(", ")
        assert plan_so_far_plan == parent_pointer_plan

        system1_plan = [str(state) for state in system1_plan]

        return plan_len, system1_plan, dfs_search_string
    
    
    def a_star_search(self, start=None, goal=None):
        start = start if start else self.start
        goal = goal if goal else self.goal

        heuristic = self.manhattan_distance(start, goal)
        queue = [MazeState(
            state = start,
            parent_state = None,
            action = "start",
            plan = [],
            dist = 0,
            state_type = "Valid",
            heuristic = heuristic
        )]

        visited = [[0]*self.w for _ in range(self.l)]
        path_len, system2_plan = -1, ""

        while len(queue):
            queue.sort(key = lambda x: x.heuristic)
            curr_state = queue[0]
            queue = queue[1:]

            state, action, plan, distance = curr_state.state, curr_state.action, curr_state.plan, curr_state.dist

            if state != start:
                system2_plan += f"Taking action '{action}' from state {curr_state.parent_state.state} | "

            system2_plan += f"Moved to state {state} | "
            system2_plan += f"Plan so far {curr_state.plan} | "
            
            visited[state[0]][state[1]] = 1

            if state == goal:
                system2_plan += f"Goal state {state} reached!"
                path_len = curr_state.dist
                break

            for action in self.actions:
                new_state = self.execute_action(state, action)
                heuristic = self.manhattan_distance(new_state, goal)
                heuristic += distance + 1
                
                system2_plan += f"Exploring action '{action}' to move to state {new_state} | "
                if not self.is_valid_state(new_state) or visited[new_state[0]][new_state[1]] == 1:
                    system2_plan += f"State {new_state} is invalid | "
                    pass
                else:
                    is_present, is_worse = False, False
                    for temp_state in queue:
                        if temp_state.state == new_state:
                            is_present = True
                            if temp_state.heuristic > heuristic:
                                is_worse = True
                            break
                    
                    if is_present and not is_worse:
                        system2_plan += f"State {new_state} is invalid | "
                        continue
                    elif is_present and is_worse:
                        queue.remove(temp_state)
                    
                    new_maze_state = MazeState(
                        state = new_state,
                        parent_state = curr_state,
                        action = action,
                        plan = plan + [action],
                        dist = distance+1,
                        state_type = "Invalid",
                        heuristic = heuristic
                    )
                    system2_plan += f"State {new_state} is valid | "
                    new_maze_state.state_type = "Valid"
                    queue.append(new_maze_state)

        return path_len, system2_plan


    def bfs(self, start=None, goal=None, pruning=0.0):
        start = start if start else self.start
        goal = goal if goal else self.goal

        queue = [MazeState(
            state = start,
            parent_state = None,
            action = "start",
            plan = [],
            dist = 0,
            state_type = "Valid"
        )]

        visited = [[0]*self.w for _ in range(self.l)]
        is_reachable, goal_state = False, None

        path_len, system1_plan, system2_plan = -1, [], ""

        while len(queue):
            curr_state = queue[0]
            state, action, plan, distance = curr_state.state, curr_state.action, curr_state.plan, curr_state.dist
            queue = queue[1:]

            if curr_state.state != start:
                system2_plan += f"Taking action '{action}' from state {curr_state.parent_state.state} | "

            system2_plan += f"Moved to state {state} | "
            system2_plan += f"Plan so far {curr_state.plan} | "
            
            visited[state[0]][state[1]] = 1

            if state == goal:
                system2_plan += f"Goal state {state} reached!"
                is_reachable = True
                goal_state = curr_state
                break
            
            action_dist = []
            for action in self.actions:
                new_state = self.execute_action(state, action)
                dist = self.manhattan_distance(new_state, goal)
                action_dist.append((action, dist))

            if pruning != 0.0:
                action_dist.sort(key=lambda x: x[1])

            pruned_actions = action_dist[:int(len(self.actions)*(1-pruning))]

            for action, _ in pruned_actions:
                new_state = self.execute_action(state, action)
                
                system2_plan += f"Exploring action '{action}' to move to state {new_state} | "
                if not self.is_valid_state(new_state) or visited[new_state[0]][new_state[1]] == 1:
                    system2_plan += f"State {new_state} is invalid | "
                    pass
                else:
                    new_maze_state = MazeState(
                        state = new_state,
                        parent_state = curr_state,
                        action = action,
                        plan = plan + [action],
                        dist = distance+1,
                        state_type="Invalid"
                    )
                    system2_plan += f"State {new_state} is valid | "
                    new_maze_state.state_type = "Valid"
                    queue.append(new_maze_state)


        if is_reachable:            
            path_len = goal_state.dist

            curr_state = goal_state
            system1_plan = [curr_state]
            while True:
                if curr_state.state == start:
                    break

                curr_state = curr_state.parent_state
                system1_plan = [curr_state] + system1_plan


            parent_pointer_plan = [state.action for state in system1_plan][1:]
            plan_so_far_plan = system2_plan.split(" | ")[-2][len("Plan so far ["):-1].replace("'", "").split(", ")
            assert plan_so_far_plan == parent_pointer_plan

            system1_plan = [str(state) for state in system1_plan]

        return path_len, system1_plan, system2_plan