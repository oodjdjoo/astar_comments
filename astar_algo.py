import numpy as np
###
class Astar2D:

    class Node:
        '''
        Node class used to create the graphhttps://chat.openai.com/c/93c3a2f3-33fe-45a0-88cc-24e74d7b1522
        '''

        def __init__(self, prev_node=None, pos=None):
            '''
            :param prev_node: Node element, needed to reconstruct the path
            :param pos: position in the  graph
            '''
            self.pos = pos
            self.H = None
            self.G = None
            self.F = None
            self.prevNode = prev_node

        def __eq__(self, in_node):
            '''
            New meaning for graph search
            :param in_node: input node
            :return: self position equal to new one
            '''
            return self.pos == in_node.pos

        def __lt__(self, other):
            '''
            New meaning for sorting
            :param other: other Node
            :return:
            '''
            self.F < other.F

    def __init__(self):
        '''
        Init stuff
        '''
        self.start_point = (0, 0)
        self.end_point = (0, 0)
        self.map = []
        self.path = list()

        # grid with and height
        self.gw = 0
        self.gh = 0

        # neighbors currently 4, no diagonals
        self.moving_dirs = self.get_neighbors_directions()

    def get_neighbors_directions(self):
        '''
        Define neighbor directions
        :return: list of positional directions
        '''
        return [[1, 0], [0, 1], [-1, 0], [0, -1]]

    def set_neighbors_directions(self, dirs=None):
        '''
        Define new neighbor directions
        :param dirs: list of new poitions
        :return: list of positional directions
        '''
        self.moving_dirs = dirs

    def generate_path(self, start_point=(0, 0), end_point=(0, 0), grid_map=None, shape=None):
        '''
        A star implementation
        :param start_point: start
        :param end_point: goal
        :param grid_map: input map
        :param shape: search space shape (i.e. 2d map (10,10))
        :return: None
        '''

        self.start_point = start_point
        self.end_point = end_point

        # load the map
        self.load_map(grid_map,shape)

        open_list = []
        closed_list = []

        # prepare open list
        start_node = self.Node(None, self.start_point)
        end_node = self.Node(None, self.end_point)

        self.__compute_cost(start_node, end_node, start_node)
        open_list.append(start_node)

        # check open list items
        while len(open_list) > 0:
            '''
            get the smallest F node
            open_list의 값을 오름차순으로 정렬 한다. 
            '''
            open_list.sort() 
            # open_list의 0번 째 index를 반복하여 제거한다. 
            cn = open_list.pop(0)
            '''
            open_list의 제거한 0 번 째 index를 closed_index에 저장하여 BacKTracking을 가능하게함
            폐 구간(노드)에 도착 했을 때 closed_list에 저장 해 놓은 node(cn)을 불러온다.
            '''
            closed_list.append(cn)
            
            # test if the end_node was reached, if yes exit
            '''
            self.path가 end_node와 동일하다면 현재 까지의 경로를
            get_path()를 통해 가져온 뒤 self.path에 저장하여 return한다.
            "end_point의 node일 경우 while 루프를 종료하여 값을 반환한다."
            '''
            if cn == end_node:
                self.path = self.get_path(cn)
                break
            '''
            새로운 노드 nn에 대한 비용 계산 
            '''    
            # get neighbors
            neighbors = self.get_neighbors(cn)

            # test neighbors and add to open list
            for n in neighbors:

                # make node object from coordinates
                nn = self.Node(cn, n)

                # calculate the costs from start / end
                self.__compute_cost(start_node, end_node, nn)
                '''
                check if neighbor is on closed list
                "Dijkstra" 알고리즘과 다른 이유.
                이미 확인한 노드가 closed list에 존재하면 무시하고 다른 neighbors node로 넘어간다.                
                '''                
                if nn in closed_list:
                    continue
                '''
                open_list에 새로운 node를 추가하는 작업.
                "Dijkstra" 알고리즘과 다른 이유.
                이미 확인한 노드가 open list에 없으면 open_list에 새로운 노드를 append(추가)한다.              
                '''      
                # check if neighbor is on the open list
                if nn not in open_list:
                    open_list.append(nn)
        return self.path
    '''
    새로운 노드 nn에 대한 비용을 계산한다. nn의 비용 계산
    '''
    @staticmethod
    def __compute_cost(start_node, end_node, current_node):
        '''
        Calculate the cost from start + heuristics (end)
        :param start_node: is the starting position
        :param end_node: goal
        :param current_node: graph current node
        :return:
        '''

        '''
        현재 노드(G) = (시작노드 - 현재 노드)값의 벡터 연산
        np.linalg -> 벡터의 크기를 계산한다
        계산은 Euclidean distance로 진행되며 이를 G값에 저장한다.

        (ex.)
        vector = np.array([1, 1]) - np.array([4, 5])  # 결과: [-3, -4]
        current_node.G = np.linalg.norm(vector)  # 결과: 5.0
        '''
        current_node.G = np.linalg.norm(np.array(start_node.pos) - np.array(current_node.pos))
        current_node.H = np.linalg.norm(np.array(end_node.pos) - np.array(current_node.pos))
        current_node.F = current_node.G + current_node.H

    def get_path(self,current_node):
        '''
        Recreate the path from end till beginning
        :param current_node: is the actual (last) node = end
        :return: list of coordinates from start till the end
        '''

        self.path = list()
        '''
        현재의 node 의 값이 None이면 self.path에 현재의 node값을 추가한다.
        현재의 노드는 그 전의 preNode의 값에 추가로 할당한다.
        return값은 시작부터 끝 까지의 좌표값의 리스트를 할당받는다.
        '''
        while current_node is not None:
            self.path.append(current_node.pos)
            # take the previous node
            current_node = current_node.prevNode
        return np.array(self.path[::-1])

    def load_map(self, in_grid, shape=None):
        '''
        Loads the map, abstracts different type of inputs.
        Accepted are: np array of n x m, n x m x 1, n x m x 3
        :param in_grid: the map
        :param shape: map shape
        :return: None
        '''

        if shape is None:
            print ("shape attribute missing!")
            exit(0)

        self.gw = shape[0]
        self.gh = shape[1]
        self.map = in_grid

    def get_neighbors(self, node):
        '''
        Return a list of neighbors
        :param node: input node
        :return: list of neighbor coordinates
        '''

        ret = []

        # get neighbor coordinates, checking also map limits (cell = 1 = wall)
        for md in self.moving_dirs:
            n = (node.pos[0] + md[0], node.pos[1] + md[1])
            xmask = ((self.map[..., 0] == n[0]) & (self.map[..., 1] == n[1]))

            if 0 <= n[0] < self.gw and 0 <= n[1] < self.gh and ~(self.map[xmask].any()):
                ret.append(n)
        return ret

#Astar2D의 부모 class를 상속받은 Astar3D 자식 class
class Astar3D(Astar2D):
    #오버 라이딩(__init__매서드 덮어서 씌운다)
    def __init__(self):
        '''
        Init stuff
        '''

        Astar2D.__init__(self)
        self.start_point = (0, 0, 0)
        self.end_point = (0, 0, 0)
        '''
        self.gw = f값(?)
        self.gh = g값(?)
        self.gd = h값(?)

        '''
        self.gw = 0  
        self.gh = 0
        self.gd = 0

        self.map = []
        self.path = list()

        Astar2D.set_neighbors_directions(self, self.get_neighbors_directions())
        self.moving_dirs = self.get_neighbors_directions()

    def get_neighbors_directions(self):
        '''
        Define neighbor directions
        :return: list of positional directions
        '''
        return [[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0], [0, -1, 0], [0, 0, -1]]

    def load_map(self, in_grid, shape=None):
        '''
        Set local map
        :param in_grid: the map
        :param shape: the shape (limits) of the 3d point cloud
        :return: None
        '''
        if shape is None:
            print ("shape attribute missing!")
            exit(0)

        self.gw = shape[0]
        self.gh = shape[1]
        self.gd = shape[2]
        self.map = in_grid

    def get_neighbors(self, node):
        '''
        Retun a list of neighbors
        :param node: input node
        :return: list of neighbor coordinates
        '''

        ret = []

        # get neighbor coordinates, checking also map limits (cell != 0 = wall)
        for md in self.moving_dirs:
            n = (node.pos[0] + md[0], node.pos[1] + md[1], node.pos[1] + md[1])

            # compute 3D mask
            xmask = (self.map[..., 0] == n[0]) & (self.map[..., 1] == n[1]) & (self.map[..., 2] == n[2])

            if 0 <= n[0] < self.gw and 0 <= n[1] < self.gh and 0 <= n[2] < self.gd and ~(self.map[xmask].any()):
                ret.append(n)
        return ret
