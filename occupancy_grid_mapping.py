#!/usr/bin/env python2
# -*- coding: utf-8 -*-

#!/usr/bin/env python

import rospy  # Biblioteca ROS para Python
import numpy as np  # Biblioteca NumPy para cálculos numéricos
import tf  # Biblioteca de Transformações do ROS
from sensor_msgs.msg import LaserScan  # Mensagem de leitura de laser
from nav_msgs.msg import OccupancyGrid  # Mensagem de grelha de ocupação
from geometry_msgs.msg import Pose  # Mensagem de pose
from tf.transformations import euler_from_quaternion  # Função para converter quaternion para ângulos de Euler

# Função para implementar o algoritmo de Bresenham, que determina os pontos numa linha entre dois pontos
def bresenham(x0, y0, x1, y1):
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while True:
        points.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy
    
    return points

# Modelo inverso do sensor para atualizar os valores log-odds da grelha de ocupação
def inverse_sensor_model(x, y, theta, z, grid_size, log_odds, z_max, alpha, beta, resolution):
    origin_x = grid_size // 2  # Centro da grelha em x
    origin_y = grid_size // 2  # Centro da grelha em y
    for i, r in enumerate(z):
        if 0 < r < z_max:  # Considerar apenas leituras válidas dentro do alcance máximo do laser
            # Calcular o ponto final do feixe laser
            angle = theta + (i - len(z) // 2) * beta
            x_end = x + r * np.cos(angle)
            y_end = y + r * np.sin(angle)

            # Converter coordenadas do mundo para coordenadas da grelha
            x_end_grid = int((x_end / resolution) + origin_x)
            y_end_grid = int((y_end / resolution) + origin_y)
            x_grid = int((x / resolution) + origin_x)
            y_grid = int((y / resolution) + origin_y)

            # Garantir que os índices estão dentro dos limites da grelha
            if 0 <= x_end_grid < grid_size and 0 <= y_end_grid < grid_size:
                cells = bresenham(x_grid, y_grid, x_end_grid, y_end_grid)
                for cell in cells[:-1]:  # Marcar células como livres ao longo do feixe
                    if 0 <= cell[0] < grid_size and 0 <= cell[1] < grid_size:
                        log_odds[cell[0], cell[1]] -= np.log(alpha / (1 - alpha))
                if 0 <= x_end_grid < grid_size and 0 <= y_end_grid < grid_size:
                    log_odds[x_end_grid, y_end_grid] += np.log(alpha / (1 - alpha))  # Marcar a célula final como ocupada
    return log_odds

# Função para atualizar a grelha de ocupação a partir dos valores log-odds
def update_occupancy_grid(log_odds, occupancy_threshold):
    log_odds = np.clip(log_odds, -50, 50)  # Limitar os valores log-odds para evitar overflow na função exp
    occupancy_grid = np.full(log_odds.shape, -1)  # Inicializar grelha de ocupação com -1 (desconhecido)
    for i in range(log_odds.shape[0]):
        for j in range(log_odds.shape[1]):
            p = 1 - 1 / (1 + np.exp(log_odds[i, j]))  # Converter log-odds para probabilidade
            if p > occupancy_threshold:
                occupancy_grid[i, j] = 100  # Ocupado
            elif p < 1 - occupancy_threshold:
                occupancy_grid[i, j] = 0  # Livre
            else:
                occupancy_grid[i, j] = -1  # Desconhecido
    return occupancy_grid

# Classe principal para o mapeamento de grelha de ocupação
class OccupancyGridMapping:
    def __init__(self):
        rospy.init_node('occupancy_grid_mapping', anonymous=True)  # Inicializar nó ROS

        self.grid_size = 200  # Tamanho da grelha
        self.resolution = 0.05  # Resolução da grelha (metros por célula)
        self.z_max = 3.5  # Alcance máximo do laser em metros
        self.alpha = 0.4  # Parâmetro do modelo inverso do sensor
        self.beta = np.radians(1)  # Resolução angular do laser (convertido para radianos)
        self.occupancy_threshold = 0.65  # Limite para considerar uma célula ocupada

        self.log_odds = np.zeros((self.grid_size, self.grid_size))  # Inicializar log-odds como zero (desconhecido)

        self.tf_listener = tf.TransformListener()  # Inicializar listener de transformações TF

        self.occupancy_grid_pub = rospy.Publisher('/occupancy_grid', OccupancyGrid, queue_size=1)  # Publisher para a grelha de ocupação
        rospy.Subscriber('/scan', LaserScan, self.scan_callback)  # Subscrever aos dados do laser

    # Callback para processar os dados do laser
    def scan_callback(self, scan):
        pose = self.get_robot_pose()
        if pose is not None:
            x, y, theta = pose
            self.log_odds = inverse_sensor_model(x, y, theta, scan.ranges, self.grid_size, self.log_odds, self.z_max, self.alpha, self.beta, self.resolution)
            occupancy_grid = update_occupancy_grid(self.log_odds, self.occupancy_threshold)
            self.publish_occupancy_grid(occupancy_grid)
        else:
            rospy.logwarn("Pose is None, skipping scan callback.")

    # Função para obter a pose atual do robô
    def get_robot_pose(self):
        try:
            self.tf_listener.waitForTransform('/odom', '/base_footprint', rospy.Time(0), rospy.Duration(1.0))
            (trans, rot) = self.tf_listener.lookupTransform('/odom', '/base_footprint', rospy.Time(0))
            x = trans[0]
            y = trans[1]
            theta = euler_from_quaternion(rot)[2]
            return x, y, theta
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            rospy.logerr("TF Exception: %s" % str(e))
            return None

    # Função para publicar a grelha de ocupação
    def publish_occupancy_grid(self, grid):
        occupancy_msg = OccupancyGrid()
        occupancy_msg.header.stamp = rospy.Time.now()
        occupancy_msg.header.frame_id = "map"
        occupancy_msg.info.resolution = self.resolution
        occupancy_msg.info.width = self.grid_size
        occupancy_msg.info.height = self.grid_size
        occupancy_msg.info.origin = Pose()
        occupancy_msg.data = grid.flatten().tolist()  # Converter a grelha de ocupação para uma lista
        self.occupancy_grid_pub.publish(occupancy_msg)

if __name__ == '__main__':
    try:
        node = OccupancyGridMapping()  # Inicializar a classe principal
        rospy.spin()  # Manter o nó ativo
    except rospy.ROSInterruptException:
        pass

