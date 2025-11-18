"""
交通规划模块
负责规划城内交通路线，包括步行、地铁、出租车等
"""

import pandas as pd
import random
from typing import List, Dict, Optional, Tuple
from pathlib import Path


class TransportPlanner:
    """交通规划器"""
    
    def __init__(self, data_loader=None):
        """
        初始化交通规划器
        
        Args:
            data_loader: 数据加载器实例
        """
        self.data_loader = data_loader
        
        # 交通方式的基础配置
        self.transport_config = {
            'walk': {
                'speed_kmh': 5,  # 步行速度 km/h
                'cost_per_km': 0,
                'max_distance': 2.0  # 最大步行距离 km
            },
            'metro': {
                'speed_kmh': 35,  # 地铁速度 km/h
                'base_price': 2,
                'price_per_km': 0.5,
                'max_distance': 30.0
            },
            'taxi': {
                'speed_kmh': 30,  # 出租车速度 km/h
                'base_price': 10,
                'price_per_km': 2.5,
                'max_passengers': 4
            }
        }
    
    def plan_route(self, from_location: str, to_location: str,
                   city: str, people_number: int = 1,
                   distance: float = None) -> List[Dict]:
        """
        规划从起点到终点的交通路线
        
        Args:
            from_location: 起点
            to_location: 终点
            city: 城市
            people_number: 人数
            distance: 距离（km），如果提供则直接使用
            
        Returns:
            交通列表
        """
        # 如果起止相同，不需要交通
        if from_location == to_location:
            return []
        
        # 获取距离
        if distance is None:
            distance = self._get_distance(from_location, to_location, city)
        
        # 如果距离很小（<100米），不需要交通
        if distance < 0.1:
            return []
        
        # 根据距离选择交通方式
        if distance <= 1.0:
            # 短距离：步行
            return self._plan_walk(from_location, to_location, distance)
        elif distance <= 5.0:
            # 中距离：步行到地铁站 + 地铁 + 步行
            return self._plan_metro_route(from_location, to_location, distance, people_number)
        else:
            # 长距离：出租车
            return self._plan_taxi(from_location, to_location, distance, people_number)
    
    def _get_distance(self, from_loc: str, to_loc: str, city: str) -> float:
        """
        获取两点之间的距离
        
        Args:
            from_loc: 起点
            to_loc: 终点
            city: 城市
            
        Returns:
            距离（km）
        """
        if self.data_loader:
            try:
                # 尝试从距离数据库查询
                distances_df = self.data_loader.load_distances(city)
                
                if not distances_df.empty:
                    # 查询距离
                    result = distances_df[
                        (distances_df['from'] == from_loc) & 
                        (distances_df['to'] == to_loc)
                    ]
                    
                    if not result.empty:
                        return float(result.iloc[0]['distance'])
            except Exception as e:
                pass
        
        # 如果查询失败，返回估计距离
        return random.uniform(2.0, 8.0)
    
    def _plan_walk(self, from_loc: str, to_loc: str, distance: float) -> List[Dict]:
        """规划步行路线"""
        config = self.transport_config['walk']
        
        # 计算步行时间（分钟）
        time_minutes = (distance / config['speed_kmh']) * 60
        
        return [{
            'start': from_loc,
            'end': to_loc,
            'mode': 'walk',
            'start_time': '',  # 将在上层填充
            'end_time': '',    # 将在上层填充
            'cost': 0,
            'distance': round(distance, 2),
            'price': 0
        }]
    
    def _plan_metro_route(self, from_loc: str, to_loc: str, 
                         distance: float, people_number: int) -> List[Dict]:
        """
        规划地铁路线
        
        返回：[步行到地铁站, 地铁, 步行到目的地]
        """
        config = self.transport_config['metro']
        
        # 步行到地铁站（假设0.5km）
        walk_to_station_distance = 0.5
        station_from = f"{from_loc}附近地铁站"
        
        # 地铁距离
        metro_distance = distance - 1.0  # 减去两端步行距离
        
        # 步行到目的地
        walk_to_dest_distance = 0.5
        station_to = f"{to_loc}附近地铁站"
        
        # 计算地铁价格
        metro_price = config['base_price'] + (metro_distance * config['price_per_km'])
        metro_price = max(metro_price, config['base_price'])
        metro_price = round(metro_price)
        
        route = [
            # 步行到地铁站
            {
                'start': from_loc,
                'end': station_from,
                'mode': 'walk',
                'start_time': '',
                'end_time': '',
                'cost': 0,
                'distance': walk_to_station_distance,
                'price': 0
            },
            # 乘地铁
            {
                'start': station_from,
                'end': station_to,
                'mode': 'metro',
                'start_time': '',
                'end_time': '',
                'cost': metro_price * people_number,
                'distance': round(metro_distance, 2),
                'tickets': people_number,
                'price': metro_price
            },
            # 步行到目的地
            {
                'start': station_to,
                'end': to_loc,
                'mode': 'walk',
                'start_time': '',
                'end_time': '',
                'cost': 0,
                'distance': walk_to_dest_distance,
                'price': 0
            }
        ]
        
        return route
    
    def _plan_taxi(self, from_loc: str, to_loc: str,
                   distance: float, people_number: int) -> List[Dict]:
        """规划出租车路线"""
        config = self.transport_config['taxi']
        
        # 计算需要的出租车数量
        cars = (people_number + config['max_passengers'] - 1) // config['max_passengers']
        
        # 计算价格（每辆车）
        price_per_car = config['base_price'] + (distance * config['price_per_km'])
        price_per_car = round(price_per_car, 2)
        
        # 总费用
        total_cost = price_per_car * cars
        
        return [{
            'start': from_loc,
            'end': to_loc,
            'mode': 'taxi',
            'start_time': '',
            'end_time': '',
            'cost': total_cost,
            'distance': round(distance, 2),
            'cars': cars,
            'price': price_per_car
        }]
    
    def calculate_travel_time(self, transports: List[Dict]) -> int:
        """
        计算交通总时间
        
        Args:
            transports: 交通列表
            
        Returns:
            总时间（分钟）
        """
        total_minutes = 0
        
        for trans in transports:
            mode = trans.get('mode')
            distance = trans.get('distance', 0)
            
            if mode in self.transport_config:
                speed = self.transport_config[mode]['speed_kmh']
                minutes = (distance / speed) * 60
                
                # 地铁需要额外的等待时间
                if mode == 'metro':
                    minutes += 5  # 平均等待5分钟
                
                total_minutes += minutes
        
        return int(total_minutes)
    
    def update_transport_times(self, transports: List[Dict],
                              start_time: str) -> List[Dict]:
        """
        更新交通时间
        
        Args:
            transports: 交通列表
            start_time: 起始时间（格式：HH:MM）
            
        Returns:
            更新后的交通列表
        """
        from datetime import datetime, timedelta
        
        current_time = datetime.strptime(start_time, "%H:%M")
        
        for trans in transports:
            trans['start_time'] = current_time.strftime("%H:%M")
            
            # 计算时间
            mode = trans.get('mode')
            distance = trans.get('distance', 0)
            
            if mode in self.transport_config:
                speed = self.transport_config[mode]['speed_kmh']
                minutes = (distance / speed) * 60
                
                if mode == 'metro':
                    minutes += 5
                
                current_time += timedelta(minutes=minutes)
            
            trans['end_time'] = current_time.strftime("%H:%M")
        
        return transports


class TimeScheduler:
    """时间调度器"""
    
    def __init__(self):
        # 活动默认时长（分钟）
        self.default_durations = {
            'airplane': 120,
            'train': 300,
            'attraction': 90,
            'breakfast': 60,
            'lunch': 60,
            'dinner': 60,
            'accommodation': 0  # 住宿不占用白天时间
        }
        
        # 推荐时间段
        self.recommended_times = {
            'breakfast': ('07:30', '08:30'),
            'lunch': ('12:00', '13:00'),
            'dinner': ('18:00', '19:00'),
            'attraction': ('09:00', '17:00')
        }
    
    def schedule_activities(self, activities: List[Dict],
                          start_time: str = "08:00") -> List[Dict]:
        """
        为活动安排时间
        
        Args:
            activities: 活动列表
            start_time: 开始时间
            
        Returns:
            安排好时间的活动列表
        """
        from datetime import datetime, timedelta
        
        current_time = datetime.strptime(start_time, "%H:%M")
        
        for activity in activities:
            activity_type = activity.get('type')
            
            # 特殊处理：交通类活动保持原有时间
            if activity_type in ['airplane', 'train']:
                if activity.get('start_time') and activity.get('end_time'):
                    # 已有时间，更新当前时间
                    current_time = datetime.strptime(
                        activity.get('end_time'), "%H:%M"
                    )
                    continue
            
            # 特殊处理：早餐、午餐、晚餐有推荐时间
            if activity_type in ['breakfast', 'lunch', 'dinner']:
                rec_start, rec_end = self.recommended_times[activity_type]
                activity['start_time'] = rec_start
                activity['end_time'] = rec_end
                current_time = datetime.strptime(rec_end, "%H:%M")
                continue
            
            # 处理交通时间
            if activity.get('transports'):
                planner = TransportPlanner()
                activity['transports'] = planner.update_transport_times(
                    activity['transports'],
                    current_time.strftime("%H:%M")
                )
                
                # 更新当前时间到交通结束
                if activity['transports']:
                    last_trans = activity['transports'][-1]
                    current_time = datetime.strptime(
                        last_trans['end_time'], "%H:%M"
                    )
            
            # 设置活动时间
            activity['start_time'] = current_time.strftime("%H:%M")
            
            # 计算活动时长
            duration = self.default_durations.get(activity_type, 90)
            current_time += timedelta(minutes=duration)
            
            activity['end_time'] = current_time.strftime("%H:%M")
        
        return activities


# 测试代码
if __name__ == "__main__":
    print("="*60)
    print("测试交通规划器")
    print("="*60)
    
    planner = TransportPlanner()
    
    # 测试1：短距离步行
    print("\n1. 短距离步行 (0.8km)")
    route = planner.plan_route("景点A", "餐厅B", "北京", 2, distance=0.8)
    for trans in route:
        print(f"   {trans['mode']}: {trans['start']} → {trans['end']}, "
              f"{trans['distance']}km, ¥{trans['cost']}")
    
    # 测试2：中距离地铁
    print("\n2. 中距离地铁 (4.5km)")
    route = planner.plan_route("景点A", "景点B", "北京", 2, distance=4.5)
    for trans in route:
        print(f"   {trans['mode']}: {trans['start']} → {trans['end']}, "
              f"{trans['distance']}km, ¥{trans['cost']}")
    
    # 测试3：长距离出租车
    print("\n3. 长距离出租车 (10km, 4人)")
    route = planner.plan_route("酒店", "机场", "北京", 4, distance=10.0)
    for trans in route:
        print(f"   {trans['mode']}: {trans['start']} → {trans['end']}, "
              f"{trans['distance']}km, ¥{trans['cost']}, {trans.get('cars', 0)}辆车")
    
    # 测试4：时间调度
    print("\n" + "="*60)
    print("测试时间调度器")
    print("="*60)
    
    scheduler = TimeScheduler()
    
    test_activities = [
        {'type': 'breakfast'},
        {'type': 'attraction', 'transports': route},
        {'type': 'lunch'},
        {'type': 'attraction', 'transports': []},
    ]
    
    scheduled = scheduler.schedule_activities(test_activities, start_time="07:30")
    
    print("\n调度结果:")
    for i, activity in enumerate(scheduled, 1):
        print(f"{i}. {activity['type']}: "
              f"{activity.get('start_time', 'N/A')} - {activity.get('end_time', 'N/A')}")
