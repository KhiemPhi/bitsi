import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

class OccupancyStruct:

    def __init__(self, bounding_box, point_cloud, resolution, expansion=3):
        self.bounding_box = bounding_box
        self.point_cloud = point_cloud 
        self.resolution = resolution
        self.expansion = expansion
        
        self.occupancy_grid = self.construct_occupancy_grid()
       
        self.occupied_graph =self.construct_occupancy_graph(occupied=1)
        self.empty_graph = self.construct_occupancy_graph(occupied=0)
        
        self.occupied_comps = self.count_connected_graphs(self.occupied_graph)
        self.empty_comps = self.count_connected_graphs(self.empty_graph)
        self.shape = self.occupancy_grid.shape

        # Calculate the area of the occupancy grid
        grid_area = self.shape[0] * self.shape[1] * self.shape[2]

        
        self.empty_factor = self.empty_comps / (self.empty_comps + self.occupied_comps)

        # Calculate the percentage adjustment factor based on grid area
        percentage_adjustment_factor = 1-(1 / grid_area)

        # Apply the adjustment to the empty factor
        self.weighted_empty_percentage = self.empty_factor
        
        
        # for i in range(3):
        #     grid_sum = self.occupancy_grid.sum(axis=i)
        #     grid_sum[grid_sum > np.mean(grid_sum)] = 1
        #     occupied_graph = self.construct_occupancy_graph_2D(grid_sum, occupied=1)
        #     empty_graph = self.construct_occupancy_graph_2D(grid_sum, occupied=0)
        #     occupied_comps = self.count_connected_graphs(occupied_graph)
        #     empty_comps = self.count_connected_graphs(empty_graph)
        #     self.occupied_empty_tuples.append((occupied_comps, empty_comps))
        

        #self.visualize_3d_occupancy_grid()

      


    def construct_occupancy_grid(self):
        # Calculate the dimensions of the grid
        
        x_min, y_min, z_min, x_max, y_max, z_max = self.bounding_box
        
        
        
        grid_width = int((x_max - x_min) / self.resolution)+self.expansion
        grid_height = int((y_max - y_min) / self.resolution)+self.expansion
        grid_depth = int((z_max - z_min) / self.resolution)+self.expansion
        
        # Initialize occupancy grid
        self.occupancy_grid = [[[0] * grid_depth for _ in range(grid_height)] for _ in range(grid_width)]
        
        # Populate occupancy grid based on point cloud
        for point in self.point_cloud:
            x, y, z = point
            # Calculate grid cell indices
            cell_x = int((x - x_min) / self.resolution)
            cell_y = int((y - y_min) / self.resolution)
            cell_z = int((z - z_min) / self.resolution)
            # Mark cell as occupied       
            # Check if the indices are within bounds
            if 0 <= cell_x < grid_width and 0 <= cell_y < grid_height and 0 <= cell_z < grid_depth:
                # Mark cell as occupied
                self.occupancy_grid[cell_x][cell_y][cell_z] = 1
           
        
    
        self.occupancy_grid = np.asarray(self.occupancy_grid)
        return self.occupancy_grid
        
    def construct_occupancy_graph(self, occupied=1): 
        occupancy_graph = nx.Graph()
      
        grid_width, grid_height, grid_depth = self.occupancy_grid.shape 
       
        for x in range(grid_width):
            for y in range(grid_height):
                for z in range(grid_depth):
                    # Check if the cell is occupied
                    if self.occupancy_grid[x][y][z] == occupied:
                        # Add node to the graph
                        node_id = (x, y, z)
                        occupancy_graph.add_node(node_id)
                        # Check neighboring cells
                        for dx in [-1, 0, 1]:
                            for dy in [-1, 0, 1]:
                                for dz in [-1, 0, 1]:
                                    if (dx, dy, dz) != (0, 0, 0):  # Exclude the current cell
                                        new_x, new_y, new_z = x + dx, y + dy, z + dz
                                        # Check if the neighboring cell is also occupied
                                        if (0 <= new_x < grid_width and 
                                            0 <= new_y < grid_height and 
                                            0 <= new_z < grid_depth and 
                                            self.occupancy_grid[new_x][new_y][new_z] == occupied):
                                            # Add edge between the current node and the neighbor
                                            neighbor_id = (new_x, new_y, new_z)
                                            occupancy_graph.add_edge(node_id, neighbor_id)

        return occupancy_graph
    


    def construct_occupancy_graph_2D(self, grid, occupied=1): 
        occupancy_graph = nx.Graph()
        grid_width, grid_height = grid.shape 
        
        for x in range(grid_width):
            for y in range(grid_height):
                # Check if the cell is occupied
                if grid[x][y] == occupied:
                    # Add node to the graph
                    node_id = (x, y)
                    occupancy_graph.add_node(node_id)
                    # Check neighboring cells
                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            if (dx, dy) != (0, 0):  # Exclude the current cell
                                new_x, new_y = x + dx, y + dy
                                # Check if the neighboring cell is also occupied
                                if (0 <= new_x < grid_width and 
                                    0 <= new_y < grid_height and 
                                    grid[new_x][new_y] == occupied):
                                    # Add edge between the current node and the neighbor
                                    neighbor_id = (new_x, new_y)
                                    occupancy_graph.add_edge(node_id, neighbor_id)

        return occupancy_graph
    
    def count_connected_graphs(self, graph):
        connected_components = nx.connected_components(graph)
        num_connected_graphs = sum([1 for component in connected_components])
        return num_connected_graphs
    
    def visualize_3d_occupancy_grid(self):
        # Convert the occupancy grid to a numpy array for visualization
        grid_array = self.occupancy_grid
        
        # Create subplots for each axis slice
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot slices along x, y, and z axes
        for i, ax in enumerate(axes):
            if i == 0:
                grid_sum = self.occupancy_grid.sum(axis=i)
                ax.imshow(grid_sum, cmap='binary', origin='upper')
                ax.set_xlabel('y')
                ax.set_ylabel('z')
                ax.set_title('Slice along X-axis_occupied={}_empty={}'.format(self.occupied_empty_tuples[0][0], self.occupied_empty_tuples[0][1]))
                ax.set_xlim(0, self.occupancy_grid.shape[1])
                ax.set_ylim(0, self.occupancy_grid.shape[2])
            elif i == 1:
                grid_sum = self.occupancy_grid.sum(axis=i)
                ax.imshow(grid_sum, cmap='binary', origin='upper')
                
                ax.set_xlim(0, self.occupancy_grid.shape[0])
                ax.set_ylim(0, self.occupancy_grid.shape[2])
        
                

                ax.set_xlabel('x')
                ax.set_ylabel('z')
                ax.set_title('Slice along Y-axis_occupied={}_empty={}'.format(self.occupied_empty_tuples[1][0], self.occupied_empty_tuples[1][1]))


            else:
                grid_sum = self.occupancy_grid.sum(axis=i)
                ax.imshow(grid_sum, cmap='binary', origin='upper')
                
                ax.set_xlabel('x')
                ax.set_ylabel('y')
              
                ax.set_title('Slice along Z-axis_occupied={}_empty={}'.format(self.occupied_empty_tuples[2][0], self.occupied_empty_tuples[2][1]))
            ax.grid(False)
        
        plt.tight_layout()
        plt.savefig("occupancy_grid.jpg")